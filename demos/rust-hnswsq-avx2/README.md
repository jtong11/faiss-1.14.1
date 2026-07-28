# Rust HNSWSQ AVX2 demo

This demo shows how to call Faiss from Rust with Cargo while linking the AVX2
static C API and Faiss archives:

- `libfaiss_c_avx2.a`
- `libfaiss_avx2.a`

The Rust code uses the C API directly and creates an `IndexHNSWSQ` through the
factory string `HNSW32,SQ8`. The crate also sets
`-C target-feature=+avx2` in `.cargo/config.toml` so the Rust binary is built
for AVX2-capable x86-64 CPUs.

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

## Run the Cargo example

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

## BLAS, LAPACK, and OpenMP dependencies

The demo statically links the Faiss archives, but the final Rust executable also
needs the same system libraries used when Faiss was built.

`build.rs` links the C++ runtime and `libgomp` by default on Linux. If your Faiss
build used a different OpenMP runtime, override it:

```sh
FAISS_OPENMP_LIB=omp cargo run --release
```

If your linker reports missing BLAS or LAPACK symbols, add the required library
names with `FAISS_EXTRA_LIBS`. For example, with OpenBLAS:

```sh
FAISS_EXTRA_LIBS=openblas cargo run --release
```

For non-standard library locations, add search paths with
`FAISS_EXTRA_LINK_SEARCH`:

```sh
FAISS_EXTRA_LINK_SEARCH=/opt/openblas/lib FAISS_EXTRA_LIBS=openblas cargo run --release
```
