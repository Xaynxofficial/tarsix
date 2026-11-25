//! Command-line interface for the Independent AI Computing Infrastructure (`as-cli`).
//!
//! Subcommands:
//! - `info`: hardware & security diagnostics
//! - `demo-train`: reproducible XOR neural network training demo
//! - `demo-llm [prompt]`: tiny RoPE + KV-Cache + SwiGLU LLM streaming demo
//! - `inspect-safetensors <path>`: safe inspection of untrusted `.safetensors` files

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

use as_accel::discover_backends;
use as_autograd::Tape;
use as_core::{DType, Error, MemoryLimits, Result};
use as_io::deserialize_safetensors;
use as_llm::{
    ByteTokenizer, LlmConfig, LlmModel, SamplingConfig, Tokenizer, load_hf_model_dir,
};
use as_nn::{Embedding, Linear, Module, Rng, TransformerBlock, cross_entropy_loss, gelu};
use as_optim::{AdamW, CosineAnnealingLr, LrScheduler, Optimizer, clip_grad_norm};
use as_tensor::Tensor;
use as_train::{
    DataLoader, Dataset, InMemoryDataset, Sample, classification_accuracy, format_csv_dataset,
    generate_synthetic_classification_dataset, inspect_checkpoint_shapes, load_checkpoint,
    parse_csv_dataset, save_checkpoint, train_val_split,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map_or("help", String::as_str);

    match cmd {
        "info" => cmd_info(),
        "demo-train" => cmd_demo_train(),
        "demo-llm" => {
            let prompt = args.get(2).map_or("Hello AI", String::as_str);
            cmd_demo_llm(prompt)
        }
        "demo-chat" => {
            let prompt = args.get(2).map_or("S: merhaba\nC:", String::as_str);
            cmd_demo_chat(prompt)
        }
        "run-llm" => {
            let dir = args.get(2).ok_or_else(|| Error::InvalidArgument {
                context: "as-cli run-llm",
                reason: "usage: as-cli run-llm <hf_model_dir> <prompt> [max_tokens] [temp]"
                    .to_owned(),
            })?;
            let prompt = args.get(3).map_or("Hello", String::as_str);
            let max_tokens: usize = args
                .get(4)
                .map_or(Ok(32), |s| s.parse())
                .map_err(|_| Error::InvalidArgument {
                    context: "as-cli run-llm",
                    reason: "max_tokens must be a positive integer".to_owned(),
                })?;
            let temp: f64 = args
                .get(5)
                .map_or(Ok(0.7), |s| s.parse())
                .map_err(|_| Error::InvalidArgument {
                    context: "as-cli run-llm",
                    reason: "temperature must be a valid float".to_owned(),
                })?;
            cmd_run_hf_llm(dir, prompt, max_tokens, temp)
        }
        "make-dataset" => {
            let out_path = args.get(2).ok_or_else(|| Error::InvalidArgument {
                context: "as-cli make-dataset",
                reason: "usage: as-cli make-dataset <out.csv> [num_samples]".to_owned(),
            })?;
            let count: usize = args
                .get(3)
                .map_or(Ok(200), |s| s.parse())
                .map_err(|_| Error::InvalidArgument {
                    context: "as-cli make-dataset",
                    reason: "num_samples must be a positive integer".to_owned(),
                })?;
            cmd_make_dataset(out_path, count)
        }
        "train" => {
            let csv_path = args.get(2).ok_or_else(|| Error::InvalidArgument {
                context: "as-cli train",
                reason: "usage: as-cli train <data.csv> <out.asck> [epochs] [hidden]".to_owned(),
            })?;
            let out_path = args.get(3).ok_or_else(|| Error::InvalidArgument {
                context: "as-cli train",
                reason: "usage: as-cli train <data.csv> <out.asck> [epochs] [hidden]".to_owned(),
            })?;
            let epochs: usize = args
                .get(4)
                .map_or(Ok(80), |s| s.parse())
                .map_err(|_| Error::InvalidArgument {
                    context: "as-cli train",
                    reason: "epochs must be a positive integer".to_owned(),
                })?;
            let hidden: usize = args
                .get(5)
                .map_or(Ok(24), |s| s.parse())
                .map_err(|_| Error::InvalidArgument {
                    context: "as-cli train",
                    reason: "hidden must be a positive integer".to_owned(),
                })?;
            cmd_train_csv(csv_path, out_path, epochs, hidden)
        }
        "predict" => {
            let ckpt_path = args.get(2).ok_or_else(|| Error::InvalidArgument {
                context: "as-cli predict",
                reason: "usage: as-cli predict <model.asck> <f0,f1,...>".to_owned(),
            })?;
            let input_csv = args.get(3).ok_or_else(|| Error::InvalidArgument {
                context: "as-cli predict",
                reason: "usage: as-cli predict <model.asck> <f0,f1,...>".to_owned(),
            })?;
            cmd_predict(ckpt_path, input_csv)
        }
        "inspect-safetensors" => {
            let path = args.get(2).ok_or_else(|| Error::InvalidArgument {
                context: "as-cli inspect-safetensors",
                reason: "usage: as-cli inspect-safetensors <file.safetensors>".to_owned(),
            })?;
            cmd_inspect_safetensors(path)
        }
        _ => {
            print_help();
            Ok(())
        }
    }
}

fn print_help() {
    println!("as-cli v{} — Independent AI Computing Infrastructure", env!("CARGO_PKG_VERSION"));
    println!();
    println!("USAGE:");
    println!("  as-cli info                                      Show hardware backends & security posture");
    println!("  as-cli make-dataset <out.csv> [samples]          Generate a synthetic 3-class CSV dataset");
    println!("  as-cli train <data.csv> <out.asck> [ep] [hid]    Train an MLP on any CSV & save checkpoint");
    println!("  as-cli predict <model.asck> <f0,f1,...>          Run live inference from a saved checkpoint");
    println!("  as-cli demo-train                                Run reproducible XOR MLP training demo");
    println!("  as-cli demo-llm [prompt]                         Run tiny RoPE/GQA/KV-Cache LLM demo");
    println!("  as-cli demo-chat [prompt]                        Train a mini Causal Transformer from scratch & chat");
    println!("  as-cli run-llm <model_dir> <prompt> [tok] [t]    Run any Llama/Qwen2/Mistral/SmolLM Safetensors model");
    println!("  as-cli inspect-safetensors <path>                Safely inspect a .safetensors archive");
}

fn cmd_info() -> Result<()> {
    println!("=== Independent AI Computing Infrastructure ===");
    println!("Version:          {}", env!("CARGO_PKG_VERSION"));
    println!("Security Posture: #![forbid(unsafe_code)], panic-free Result<T>, overflow-checks=true");
    println!("External Crates:  0 (zero third-party runtime dependencies)");
    println!();
    println!("Discovered Backends:");
    for b in discover_backends() {
        let status = if b.ready_for_execution { "ACTIVE" } else { "KERNEL-READY" };
        println!("  - {:<10} [{}] {}", b.device.to_string(), status, b.name);
    }
    Ok(())
}

fn cmd_demo_train() -> Result<()> {
    let dataset = InMemoryDataset::new(vec![
        Sample { features: vec![0.0, 0.0], label: 0 },
        Sample { features: vec![0.0, 1.0], label: 1 },
        Sample { features: vec![1.0, 0.0], label: 1 },
        Sample { features: vec![1.0, 1.0], label: 0 },
    ])?;
    let loader = DataLoader::new(4, true, false, DType::F32)?;
    let mut rng = Rng::from_seed(42);
    let mut fc1 = Linear::new("fc1", 2, 16, true, DType::F32, &mut rng)?;
    let mut fc2 = Linear::new("fc2", 16, 2, true, DType::F32, &mut rng)?;
    let mut optim = AdamW::new(0.05, 1e-4);
    let scheduler = CosineAnnealingLr {
        initial_lr: 0.05,
        min_lr: 0.005,
        warmup_steps: 5,
        total_steps: 120,
    };

    println!("Training 2-layer GELU MLP on XOR (120 epochs, AdamW + CosineAnnealingLr)...");
    for epoch in 0..120 {
        optim.set_learning_rate(scheduler.lr_at(epoch));
        let batches = loader.epoch_batches(&dataset, &mut rng)?;
        for batch in &batches {
            let mut tape = Tape::new();
            let x = tape.var(batch.inputs.clone(), false);
            let (h1, mut pvars) = fc1.forward(&mut tape, x)?;
            let act = gelu(&mut tape, h1)?;
            let (logits, p2) = fc2.forward(&mut tape, act)?;
            pvars.extend(p2);

            let loss = cross_entropy_loss(&mut tape, logits, &batch.labels)?;
            let loss_val = tape.value(loss)?.to_f32_vec()?.first().copied().unwrap_or(0.0);
            if epoch == 0 || epoch == 59 || epoch == 119 {
                let acc = classification_accuracy(tape.value(logits)?, &batch.labels)?;
                println!("  epoch {:>3} | loss = {:.4} | acc = {:.1}%", epoch + 1, loss_val, acc * 100.0);
            }

            let grads = tape.backward(loss)?;
            let mut grad_tensors = Vec::with_capacity(pvars.len());
            for &v in &pvars {
                let g = grads.get(v).cloned().ok_or(Error::InternalInvariant {
                    message: "missing gradient during demo-train",
                })?;
                grad_tensors.push(g);
            }
            let _ = clip_grad_norm(&mut grad_tensors, 5.0)?;
            let grad_refs: Vec<Option<&Tensor>> = grad_tensors.iter().map(Some).collect();
            let mut params = fc1.parameters_mut();
            params.extend(fc2.parameters_mut());
            optim.step(&mut params, &grad_refs)?;
        }
    }
    Ok(())
}

fn cmd_demo_llm(prompt: &str) -> Result<()> {
    let tok = ByteTokenizer;
    let prompt_ids = tok.encode(prompt, true);
    let cfg = LlmConfig {
        vocab_size: ByteTokenizer::VOCAB_SIZE,
        hidden_size: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 64,
        max_seq_len: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
    };
    let model = LlmModel::new(cfg, 42)?;
    let mut rng = Rng::from_seed(7);
    let sampling = SamplingConfig {
        temperature: 0.7,
        top_k: 16,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };

    println!("Prompt: \"{prompt}\" ({} tokens)", prompt_ids.len());
    print!("Generated token IDs (streaming): ");
    let generated = model.generate_stream(&prompt_ids, 12, tok.eos_id(), sampling, &mut rng, |id| {
        print!("{id} ");
        true
    })?;
    println!("\nCompleted {} autoregressive KV-Cache steps.", generated.len());
    Ok(())
}

fn cmd_inspect_safetensors(path: &str) -> Result<()> {
    let bytes = fs::read(path).map_err(|e| Error::InvalidArgument {
        context: "inspect-safetensors",
        reason: format!("failed to read `{path}`: {e}"),
    })?;
    let archive = deserialize_safetensors(&bytes, MemoryLimits::default())?;
    println!("Safetensors archive: {path}");
    println!("Metadata entries: {}", archive.metadata.len());
    for (k, v) in &archive.metadata {
        println!("  meta[{k}] = {v}");
    }
    println!("Tensors ({}):", archive.tensors.len());
    for (name, t) in &archive.tensors {
        println!("  - {:<40} dtype={:<4} shape={:?}", name, t.dtype(), t.dims());
    }
    Ok(())
}

fn cmd_make_dataset(out_path: &str, num_samples: usize) -> Result<()> {
    let ds = generate_synthetic_classification_dataset(num_samples, 42)?;
    let csv_text = format_csv_dataset(&ds, true)?;
    fs::write(out_path, csv_text).map_err(|e| Error::InvalidArgument {
        context: "make-dataset",
        reason: format!("failed to write `{out_path}`: {e}"),
    })?;
    println!(
        "Created dataset `{out_path}` ({} samples, {} features, 3 classes).",
        ds.len(),
        ds.feature_dim()
    );
    Ok(())
}

fn cmd_train_csv(csv_path: &str, out_path: &str, epochs: usize, hidden: usize) -> Result<()> {
    let text = fs::read_to_string(csv_path).map_err(|e| Error::InvalidArgument {
        context: "train",
        reason: format!("failed to read `{csv_path}`: {e}"),
    })?;
    let first_line = text.lines().next().unwrap_or("");
    let has_header = first_line
        .split(',')
        .next()
        .is_some_and(|tok| tok.trim().parse::<f64>().is_err());

    let (full_ds, num_classes) = parse_csv_dataset(&text, has_header)?;
    let in_features = full_ds.feature_dim();
    let mut rng = Rng::from_seed(42);
    let (train_ds, val_ds) = train_val_split(&full_ds, 0.2, &mut rng)?;

    println!(
        "Loaded `{csv_path}`: {} train / {} val samples | features={in_features}, classes={num_classes}",
        train_ds.len(),
        val_ds.len()
    );

    let batch_size = train_ds.len().min(32).max(1);
    let loader = DataLoader::new(batch_size, true, false, DType::F32)?;
    let val_loader = DataLoader::new(val_ds.len(), false, false, DType::F32)?;

    let mut fc1 = Linear::new("fc1", in_features, hidden, true, DType::F32, &mut rng)?;
    let mut fc2 = Linear::new("fc2", hidden, num_classes, true, DType::F32, &mut rng)?;
    let mut optim = AdamW::new(0.03, 1e-4);
    let scheduler = CosineAnnealingLr {
        initial_lr: 0.03,
        min_lr: 0.002,
        warmup_steps: 5,
        total_steps: epochs.max(1),
    };

    for epoch in 0..epochs {
        optim.set_learning_rate(scheduler.lr_at(epoch));
        let batches = loader.epoch_batches(&train_ds, &mut rng)?;
        let mut loss_sum = 0.0f64;

        for batch in &batches {
            let mut tape = Tape::new();
            let x = tape.var(batch.inputs.clone(), false);
            let (h1, mut pvars) = fc1.forward(&mut tape, x)?;
            let act = gelu(&mut tape, h1)?;
            let (logits, p2) = fc2.forward(&mut tape, act)?;
            pvars.extend(p2);

            let loss = cross_entropy_loss(&mut tape, logits, &batch.labels)?;
            loss_sum += f64::from(tape.value(loss)?.to_f32_vec()?.first().copied().unwrap_or(0.0));

            let grads = tape.backward(loss)?;
            let mut grad_tensors = Vec::with_capacity(pvars.len());
            for &v in &pvars {
                let g = grads.get(v).cloned().ok_or(Error::InternalInvariant {
                    message: "missing gradient during CSV training",
                })?;
                grad_tensors.push(g);
            }
            let _ = clip_grad_norm(&mut grad_tensors, 5.0)?;
            let grad_refs: Vec<Option<&Tensor>> = grad_tensors.iter().map(Some).collect();
            let mut params = fc1.parameters_mut();
            params.extend(fc2.parameters_mut());
            optim.step(&mut params, &grad_refs)?;
        }

        if epoch == 0 || (epoch + 1) % (epochs / 4).max(1) == 0 || epoch + 1 == epochs {
            let val_batches = val_loader.epoch_batches(&val_ds, &mut rng)?;
            let mut val_acc = 0.0;
            if let Some(vb) = val_batches.first() {
                let mut eval_tape = Tape::new();
                eval_tape.set_grad_enabled(false);
                let vx = eval_tape.var(vb.inputs.clone(), false);
                let (h1, _) = fc1.forward(&mut eval_tape, vx)?;
                let act = gelu(&mut eval_tape, h1)?;
                let (logits, _) = fc2.forward(&mut eval_tape, act)?;
                val_acc = classification_accuracy(eval_tape.value(logits)?, &vb.labels)?;
            }
            println!(
                "  epoch {:>3}/{epochs} | train_loss = {:.4} | val_acc = {:.1}%",
                epoch + 1,
                loss_sum / (batches.len().max(1) as f64),
                val_acc * 100.0
            );
        }
    }

    let mut params = fc1.parameters();
    params.extend(fc2.parameters());
    let ckpt_bytes = save_checkpoint(&params)?;
    fs::write(out_path, &ckpt_bytes).map_err(|e| Error::InvalidArgument {
        context: "train",
        reason: format!("failed to write checkpoint `{out_path}`: {e}"),
    })?;
    println!(
        "Saved verified ASCK checkpoint `{out_path}` ({} bytes).",
        ckpt_bytes.len()
    );
    Ok(())
}

fn cmd_predict(ckpt_path: &str, input_csv: &str) -> Result<()> {
    let bytes = fs::read(ckpt_path).map_err(|e| Error::InvalidArgument {
        context: "predict",
        reason: format!("failed to read `{ckpt_path}`: {e}"),
    })?;
    let specs = inspect_checkpoint_shapes(&bytes, MemoryLimits::default())?;
    let (fc1_shape, _) = specs.get("fc1.weight").ok_or_else(|| Error::InvalidArgument {
        context: "predict",
        reason: "checkpoint missing `fc1.weight`".to_owned(),
    })?;
    let (fc2_shape, _) = specs.get("fc2.weight").ok_or_else(|| Error::InvalidArgument {
        context: "predict",
        reason: "checkpoint missing `fc2.weight`".to_owned(),
    })?;

    let hidden = fc1_shape.dim(0)?;
    let in_features = fc1_shape.dim(1)?;
    let num_classes = fc2_shape.dim(0)?;

    let mut vals = Vec::new();
    for part in input_csv.split(',') {
        let v: f32 = part.trim().parse().map_err(|_| Error::InvalidArgument {
            context: "predict",
            reason: format!("invalid float input `{part}`"),
        })?;
        vals.push(v);
    }
    if vals.len() != in_features {
        return Err(Error::BufferLengthMismatch {
            expected: in_features,
            actual: vals.len(),
        });
    }

    let mut rng = Rng::from_seed(0);
    let mut fc1 = Linear::new("fc1", in_features, hidden, true, DType::F32, &mut rng)?;
    let mut fc2 = Linear::new("fc2", hidden, num_classes, true, DType::F32, &mut rng)?;
    {
        let mut params = fc1.parameters_mut();
        params.extend(fc2.parameters_mut());
        load_checkpoint(&bytes, &mut params, MemoryLimits::default())?;
    }

    let mut tape = Tape::new();
    tape.set_grad_enabled(false);
    let x = tape.var(Tensor::from_f32(&vals, vec![1, in_features])?, false);
    let (h1, _) = fc1.forward(&mut tape, x)?;
    let act = gelu(&mut tape, h1)?;
    let (logits_var, _) = fc2.forward(&mut tape, act)?;
    let logits = tape.value(logits_var)?.to_f32_vec()?;

    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_l).exp()).collect();
    let sum_e: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum_e).collect();

    let mut best_cls = 0usize;
    let mut best_p = -1.0f32;
    for (c, &p) in probs.iter().enumerate() {
        if p > best_p {
            best_p = p;
            best_cls = c;
        }
    }

    println!("Input:           {vals:?}");
    println!("Predicted class: {best_cls} (confidence: {:.1}%)", best_p * 100.0);
    println!("Probabilities:   {probs:?}");
    Ok(())
}

/// Trains a real character-level Causal Transformer Language Model from scratch
/// on a compact Turkish dialogue corpus using our own `as-autograd` + `as-nn` + `AdamW`,
/// then autoregressively generates a reply to `prompt`.
fn cmd_demo_chat(prompt: &str) -> Result<()> {
    let corpus = [
        "S: merhaba\nC: merhaba! ben tarsix, rust ile yazilmis yapay zeka motoruyum.\n",
        "S: adin ne\nC: adim tarsix. sifirdan gelistirilmis guvenli bir ai cekirdegiyim.\n",
        "S: ne ise yararsin\nC: tensor, autograd, egitim ve llm cikarimi yaparim.\n",
        "S: nasilsin\nC: harikayim! tum testlerim yesil ve calismaya hazirim.\n",
    ];

    // Build deterministic character vocabulary
    let mut chars: Vec<char> = Vec::new();
    for line in &corpus {
        for ch in line.chars() {
            if !chars.contains(&ch) {
                chars.push(ch);
            }
        }
    }
    for ch in prompt.chars() {
        if !chars.contains(&ch) {
            chars.push(ch);
        }
    }
    let vocab_size = chars.len();
    let char_to_id = |c: char| -> usize { chars.iter().position(|&x| x == c).unwrap_or(0) };

    let mut rng = Rng::from_seed(42);
    let embed_dim = 32;
    let max_len = 96;
    let mut emb = Embedding::new("emb", vocab_size, embed_dim, DType::F32, &mut rng)?;
    let mut pos_emb = Embedding::new("pos_emb", max_len, embed_dim, DType::F32, &mut rng)?;
    let mut block = TransformerBlock::new("blk0", embed_dim, 4, 64, DType::F32, &mut rng)?;
    let mut lm_head = Linear::new("lm_head", embed_dim, vocab_size, true, DType::F32, &mut rng)?;
    let mut optim = AdamW::new(0.02, 1e-4);

    let seqs: Vec<Vec<usize>> = corpus
        .iter()
        .map(|s| s.chars().map(char_to_id).collect())
        .collect();

    println!(
        "Sifirdan Causal Transformer Dil Modeli egitiliyor (vocab={vocab_size}, dim={embed_dim}, pos_emb, 120 adim)..."
    );

    let steps = 120usize;
    for step in 0..steps {
        let mut step_loss = 0.0f64;
        for seq in &seqs {
            let t_len = seq.len().saturating_sub(1);
            if t_len == 0 {
                continue;
            }
            let inp = &seq[..t_len];
            let tgt = &seq[1..];
            let positions: Vec<usize> = (0..t_len).collect();

            let mut tape = Tape::new();
            let (tok_h, mut pvars) = emb.forward(&mut tape, inp, 1, t_len)?;
            let (pos_h, p_pos) = pos_emb.forward(&mut tape, &positions, 1, t_len)?;
            pvars.extend(p_pos);
            let h_in = tape.add(tok_h, pos_h)?;
            let (h_blk, p_blk) = block.forward(&mut tape, h_in, true)?;
            pvars.extend(p_blk);
            let (logits_3d, p_head) = lm_head.forward(&mut tape, h_blk)?;
            pvars.extend(p_head);
            let logits_2d = tape.reshape(logits_3d, vec![t_len, vocab_size])?;

            let loss = cross_entropy_loss(&mut tape, logits_2d, tgt)?;
            step_loss += f64::from(tape.value(loss)?.to_f32_vec()?.first().copied().unwrap_or(0.0));

            let grads = tape.backward(loss)?;
            let mut g_tensors = Vec::with_capacity(pvars.len());
            for &v in &pvars {
                let g = grads.get(v).cloned().ok_or(Error::InternalInvariant {
                    message: "missing gradient in demo-chat",
                })?;
                g_tensors.push(g);
            }
            let _ = clip_grad_norm(&mut g_tensors, 2.0)?;
            let g_refs: Vec<Option<&Tensor>> = g_tensors.iter().map(Some).collect();
            let mut params = emb.parameters_mut();
            params.extend(pos_emb.parameters_mut());
            params.extend(block.parameters_mut());
            params.extend(lm_head.parameters_mut());
            optim.step(&mut params, &g_refs)?;
        }

        if step == 0 || step == 59 || step == steps - 1 {
            println!(
                "  adim {:>3}/{steps} | loss = {:.4}",
                step + 1,
                step_loss / (seqs.len() as f64)
            );
        }
    }

    // Autoregressive greedy generation from prompt
    let mut ids: Vec<usize> = prompt.chars().map(char_to_id).collect();
    let mut reply = String::new();
    for _ in 0..80 {
        if ids.len() >= max_len {
            break;
        }
        let mut tape = Tape::new();
        tape.set_grad_enabled(false);
        let t_len = ids.len();
        let positions: Vec<usize> = (0..t_len).collect();
        let (tok_h, _) = emb.forward(&mut tape, &ids, 1, t_len)?;
        let (pos_h, _) = pos_emb.forward(&mut tape, &positions, 1, t_len)?;
        let h_in = tape.add(tok_h, pos_h)?;
        let (h_blk, _) = block.forward(&mut tape, h_in, true)?;
        let (logits_3d, _) = lm_head.forward(&mut tape, h_blk)?;
        let all = tape.value(logits_3d)?.to_f32_vec()?;
        let last_start = (t_len - 1) * vocab_size;
        let last_slice = all.get(last_start..last_start + vocab_size).unwrap_or(&[]);

        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &v) in last_slice.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = idx;
            }
        }
        ids.push(best_id);
        let ch = chars.get(best_id).copied().unwrap_or(' ');
        if ch == '\n' {
            break;
        }
        reply.push(ch);
    }

    println!("\nGirdi : {}", prompt.replace('\n', " | "));
    println!("Tarsix:{reply}");
    Ok(())
}

fn cmd_run_hf_llm(dir: &str, prompt: &str, max_tokens: usize, temperature: f64) -> Result<()> {
    println!("Loading HuggingFace Safetensors model from `{dir}`...");
    let loaded = load_hf_model_dir(dir, 2048, MemoryLimits::default())?;
    let cfg = loaded.model.config();
    println!(
        "Loaded architecture `{}` | layers={} hidden={} heads={} kv_heads={} vocab={}",
        loaded.model_type,
        cfg.num_layers,
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.vocab_size
    );

    let prompt_ids = loaded.tokenizer.encode(prompt, true);
    let mut rng = Rng::from_seed(42);
    let sampling = SamplingConfig {
        temperature,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.1,
    };

    print!("{prompt}");
    let _ = io::stdout().flush();
    let _ = loaded.model.generate_stream(
        &prompt_ids,
        max_tokens,
        loaded.tokenizer.eos_id(),
        sampling,
        &mut rng,
        |tok_id| {
            if let Ok(piece) = loaded.tokenizer.decode(&[tok_id]) {
                print!("{piece}");
                let _ = io::stdout().flush();
            }
            true
        },
    )?;
    println!();
    Ok(())
}
