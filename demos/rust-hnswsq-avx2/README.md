# Rust HNSWSQ AVX2 demo

This demo shows how to call Faiss from Rust with Cargo while producing a
fully-static Linux binary. It links the AVX2 static C API and Faiss archives:

- `libfaiss_c_avx2.a`
- `libfaiss_avx2.a`

The Rust code uses the C API directly and creates an `IndexHNSWSQ` through the
factory string `HNSW32,SQ8`. The crate also sets
`-C target-feature=+avx2,+crt-static` and `-C link-arg=-static` in
`.cargo/config.toml` so the Rust binary is built for AVX2-capable x86-64 CPUs
and links the C runtime statically.

## Build Faiss

From the repository root:

```sh
cmake -B build \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_C_API=ON \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF \
  .

cmake --build build --target faiss_c_avx2 -j
```

## Build and run the Cargo example

From this directory:

```sh
cargo run --release
```

By default, `build.rs` searches for the CMake build-tree archives in:

- `../../build/c_api`
- `../../build/faiss`

If you installed Faiss elsewhere, point Cargo at the install library directory:

```sh
FAISS_LIB_DIR=/path/to/faiss/lib cargo run --release
```

For separate build or install directories, use:

```sh
FAISS_C_API_LIB_DIR=/path/to/c_api/libdir \
FAISS_FAISS_LIB_DIR=/path/to/faiss/libdir \
cargo run --release
```

The resulting binary should be static:

```sh
file target/release/faiss-hnswsq-avx2-demo
ldd target/release/faiss-hnswsq-avx2-demo
```

`ldd` should print `statically linked` or `not a dynamic executable`.

## Static BLAS, LAPACK, and OpenMP dependencies

The demo statically links the Faiss archives and asks Cargo to statically link
the remaining native dependencies too.

`build.rs` links the GNU C++ runtime and `libgomp` statically by default on
Linux. If your Faiss build used a different OpenMP runtime, override it:

```sh
FAISS_OPENMP_LIB=omp cargo run --release
```

For BLAS/LAPACK, the default is static OpenBLAS plus the GNU Fortran runtime
libraries that OpenBLAS' static LAPACK objects may need:

```sh
FAISS_EXTRA_STATIC_LIBS=openblas,gfortran,quadmath cargo run --release
```

If your Faiss build used a different static BLAS/LAPACK backend, replace the
library list. For example:

```sh
FAISS_EXTRA_STATIC_LIBS="mkl_intel_lp64 mkl_sequential mkl_core" cargo run --release
```

For non-standard library locations, add search paths with
`FAISS_EXTRA_LINK_SEARCH`:

```sh
FAISS_EXTRA_LINK_SEARCH=/opt/openblas/lib \
FAISS_EXTRA_STATIC_LIBS=openblas,gfortran,quadmath \
cargo run --release
```

`FAISS_EXTRA_LIBS` is accepted as a backwards-compatible alias for
`FAISS_EXTRA_STATIC_LIBS`; both are emitted as static link directives.
