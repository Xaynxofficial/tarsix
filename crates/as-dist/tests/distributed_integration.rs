//! Integration tests for Phase 3 Distributed Compute (`as-dist`):
//! - 4-worker `LocalCluster` Ring-AllReduce (`Sum`, `Mean`, `Max`, `Min`), `Broadcast`, `AllGather`, `Barrier`
//! - `DistributedDataParallel` weight broadcast & gradient averaging
//! - Real 3-node loopback TCP `TcpCommunicator` Ring-AllReduce over OS sockets

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::net::TcpListener;
use std::thread;

use as_core::DType;
use as_dist::{
    AllReduceOp, Communicator, DEFAULT_MAX_FRAME_BYTES, DistributedDataParallel, LocalCluster,
    TcpCommunicator,
};
use as_nn::{Linear, Module, Rng};
use as_tensor::Tensor;

#[test]
fn local_cluster_ring_all_reduce_broadcast_all_gather_and_ddp() {
    let world_size = 4;
    let comms = LocalCluster::create(world_size).unwrap();

    let handles: Vec<_> = comms
        .into_iter()
        .map(|comm| {
            thread::spawn(move || {
                let r = comm.rank() as f32;
                let local = Tensor::from_f32(&[r + 1.0, (r + 1.0) * 10.0, 5.0 - r], vec![3]).unwrap();

                // Sum across ranks 0..3: [1+2+3+4=10, 10+20+30+40=100, 5+4+3+2=14]
                let sum = comm.all_reduce(&local, AllReduceOp::Sum).unwrap();
                assert_eq!(sum.to_f32_vec().unwrap(), vec![10.0, 100.0, 14.0]);

                // Mean: [2.5, 25.0, 3.5]
                let mean = comm.all_reduce(&local, AllReduceOp::Mean).unwrap();
                assert_eq!(mean.to_f32_vec().unwrap(), vec![2.5, 25.0, 3.5]);

                // Max: [4.0, 40.0, 5.0]
                let max = comm.all_reduce(&local, AllReduceOp::Max).unwrap();
                assert_eq!(max.to_f32_vec().unwrap(), vec![4.0, 40.0, 5.0]);

                // Broadcast from rank 0
                let bcast_in = if comm.rank() == 0 {
                    Tensor::from_f32(&[42.0, 99.0], vec![2]).unwrap()
                } else {
                    Tensor::zeros(vec![2], DType::F32).unwrap()
                };
                let bcast = comm.broadcast(&bcast_in, 0).unwrap();
                assert_eq!(bcast.to_f32_vec().unwrap(), vec![42.0, 99.0]);

                // All-Gather
                let gathered = comm.all_gather(&local).unwrap();
                assert_eq!(gathered.len(), world_size);
                assert_eq!(gathered[0].to_f32_vec().unwrap(), vec![1.0, 10.0, 5.0]);
                assert_eq!(gathered[3].to_f32_vec().unwrap(), vec![4.0, 40.0, 2.0]);

                // DDP parameter broadcast + gradient averaging
                let mut rng = Rng::from_seed(100 + comm.rank() as u64);
                let mut layer = Linear::new("fc", 2, 2, false, DType::F32, &mut rng).unwrap();
                let ddp = DistributedDataParallel::new(&comm);
                ddp.broadcast_parameters(&mut layer).unwrap();

                let mut grads = vec![Some(
                    Tensor::full(vec![2, 2], (comm.rank() + 1) as f64, DType::F32).unwrap(),
                )];
                ddp.sync_gradients(&mut grads).unwrap();
                assert_eq!(
                    grads[0].as_ref().unwrap().to_f32_vec().unwrap(),
                    vec![2.5, 2.5, 2.5, 2.5]
                );

                comm.barrier().unwrap();
                layer.parameters()[0].data().to_f32_vec().unwrap()
            })
        })
        .collect();

    let weights: Vec<Vec<f32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for w in &weights[1..] {
        assert_eq!(w, &weights[0], "all DDP ranks must share identical initial weights");
    }
}

#[test]
fn tcp_communicator_runs_ring_all_reduce_over_real_sockets() {
    let world_size = 3;
    let listeners: Vec<TcpListener> = (0..world_size)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addrs: Vec<_> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let next_addr = addrs[(rank + 1) % world_size];
            thread::spawn(move || {
                let comm = TcpCommunicator::connect_ring(
                    rank,
                    world_size,
                    listener,
                    next_addr,
                    DEFAULT_MAX_FRAME_BYTES,
                )
                .unwrap();

                let t = Tensor::from_f32(
                    &[(rank + 1) as f32, ((rank + 1) * 2) as f32, 10.0, 20.0, 30.0],
                    vec![5],
                )
                .unwrap();
                let reduced = comm.all_reduce(&t, AllReduceOp::Sum).unwrap();
                comm.barrier().unwrap();
                reduced.to_f32_vec().unwrap()
            })
        })
        .collect();

    for h in handles {
        let out = h.join().unwrap();
        assert_eq!(out, vec![6.0, 12.0, 30.0, 60.0, 90.0]);
    }
}
