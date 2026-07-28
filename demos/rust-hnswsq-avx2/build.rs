use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=FAISS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=FAISS_C_API_LIB_DIR");
    println!("cargo:rerun-if-env-changed=FAISS_FAISS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=FAISS_EXTRA_LINK_SEARCH");
    println!("cargo:rerun-if-env-changed=FAISS_OPENMP_LIB");
    println!("cargo:rerun-if-env-changed=FAISS_EXTRA_LIBS");
    println!("cargo:rerun-if-env-changed=FAISS_EXTRA_STATIC_LIBS");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("demo should live under <repo>/demos/rust-hnswsq-avx2");

    if let Some(dir) = env::var_os("FAISS_LIB_DIR") {
        add_link_search(PathBuf::from(dir));
    } else {
        add_env_or_default("FAISS_C_API_LIB_DIR", repo_root.join("build/c_api"));
        add_env_or_default("FAISS_FAISS_LIB_DIR", repo_root.join("build/faiss"));
    }

    if let Some(paths) = env::var_os("FAISS_EXTRA_LINK_SEARCH") {
        for path in env::split_paths(&paths) {
            add_link_search(path);
        }
    }
    add_gnu_runtime_link_searches();
    add_common_static_dependency_searches();

    println!("cargo:rustc-link-lib=static=faiss_c_avx2");
    println!("cargo:rustc-link-lib=static=faiss_avx2");

    link_cxx_runtime();
    link_openmp();

    for lib in extra_static_libs() {
        println!("cargo:rustc-link-lib=static={lib}");
    }
}

fn add_env_or_default(env_name: &str, default: PathBuf) {
    let path = env::var_os(env_name).map(PathBuf::from).unwrap_or(default);
    add_link_search(path);
}

fn add_link_search(path: PathBuf) {
    println!("cargo:rustc-link-search=native={}", path.display());
}

fn add_gnu_runtime_link_searches() {
    if cfg!(target_os = "macos") {
        return;
    }

    for (compiler, library) in [
        ("g++", "libstdc++.a"),
        ("gcc", "libgomp.a"),
        ("gfortran", "libgfortran.a"),
        ("gcc", "libquadmath.a"),
    ] {
        if let Some(path) = compiler_library_path(compiler, library) {
            if let Some(parent) = path.parent() {
                add_link_search(parent.to_path_buf());
            }
        }
    }
}

fn compiler_library_path(compiler: &str, library: &str) -> Option<PathBuf> {
    let output = Command::new(compiler)
        .arg(format!("-print-file-name={library}"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(path.trim());
    path.is_absolute().then_some(path)
}

fn add_common_static_dependency_searches() {
    if cfg!(target_os = "macos") {
        return;
    }

    for path in [
        "/usr/lib/x86_64-linux-gnu/openblas-pthread",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        let path = PathBuf::from(path);
        if path.exists() {
            add_link_search(path);
        }
    }
}

fn link_cxx_runtime() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=static=stdc++");
    }
}

fn link_openmp() {
    let default_openmp = if cfg!(target_os = "macos") {
        "omp"
    } else {
        "gomp"
    };
    let openmp = env::var("FAISS_OPENMP_LIB").unwrap_or_else(|_| default_openmp.to_owned());
    if !openmp.trim().is_empty() {
        println!("cargo:rustc-link-lib=static={}", openmp.trim());
    }
}

fn extra_static_libs() -> impl Iterator<Item = String> {
    let libs = env::var("FAISS_EXTRA_STATIC_LIBS")
        .or_else(|_| env::var("FAISS_EXTRA_LIBS"))
        .unwrap_or_else(|_| "openblas,gfortran,quadmath".to_owned());
    split_libs(libs)
}

fn split_libs(libs: String) -> impl Iterator<Item = String> {
    libs.split(|c| matches!(c, ',' | ';' | ':' | ' '))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>()
        .into_iter()
}
