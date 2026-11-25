use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/metal_bridge.m");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let obj_path = out_dir.join("metal_bridge.o");
    let lib_path = out_dir.join("libas_metal_bridge.a");

    let mut clang = Command::new("clang");
    clang.args([
        "-c",
        "-O3",
        "-fobjc-arc",
        "-Wall",
        "-Wextra",
        "native/metal_bridge.m",
        "-o",
    ]);
    clang.arg(&obj_path);

    let status = clang.status().expect("failed to invoke clang");
    assert!(status.success(), "clang failed to compile metal_bridge.m");

    let ar_status = Command::new("ar")
        .args(["rcs"])
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
        .expect("failed to invoke ar");
    assert!(ar_status.success(), "ar failed to create libas_metal_bridge.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=as_metal_bridge");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
