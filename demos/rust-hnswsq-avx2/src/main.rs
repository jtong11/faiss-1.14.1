/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::ptr::NonNull;

type FaissIdx = i64;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

const D: usize = 64;
const NB: usize = 20_000;
const NQ: usize = 5;
const K: usize = 4;
const HNSW_M: usize = 32;
const EF_SEARCH: c_int = 64;
const METRIC_L2: c_int = 1;

#[repr(C)]
struct FaissIndex {
    _private: [u8; 0],
}

#[repr(C)]
struct FaissSearchParameters {
    _private: [u8; 0],
}

#[repr(C)]
struct FaissSearchParametersHNSW {
    _private: [u8; 0],
}

extern "C" {
    fn faiss_get_last_error() -> *const c_char;

    fn faiss_index_factory(
        p_index: *mut *mut FaissIndex,
        d: c_int,
        description: *const c_char,
        metric: c_int,
    ) -> c_int;

    fn faiss_Index_free(index: *mut FaissIndex);
    fn faiss_Index_is_trained(index: *const FaissIndex) -> c_int;
    fn faiss_Index_ntotal(index: *const FaissIndex) -> FaissIdx;
    fn faiss_Index_train(index: *mut FaissIndex, n: FaissIdx, x: *const f32) -> c_int;
    fn faiss_Index_add(index: *mut FaissIndex, n: FaissIdx, x: *const f32) -> c_int;
    fn faiss_Index_search_with_params(
        index: *const FaissIndex,
        n: FaissIdx,
        x: *const f32,
        k: FaissIdx,
        params: *const FaissSearchParameters,
        distances: *mut f32,
        labels: *mut FaissIdx,
    ) -> c_int;

    fn faiss_SearchParametersHNSW_new(p_sp: *mut *mut FaissSearchParametersHNSW) -> c_int;
    fn faiss_SearchParametersHNSW_free(sp: *mut FaissSearchParametersHNSW);
    fn faiss_SearchParametersHNSW_set_efSearch(sp: *mut FaissSearchParametersHNSW, value: c_int);
}

#[derive(Debug)]
struct FaissError {
    code: c_int,
    message: String,
}

impl fmt::Display for FaissError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Faiss error {}: {}", self.code, self.message)
    }
}

impl Error for FaissError {}

fn check(code: c_int) -> std::result::Result<(), FaissError> {
    if code == 0 {
        return Ok(());
    }

    let message = unsafe {
        let ptr = faiss_get_last_error();
        if ptr.is_null() {
            "unknown error".to_owned()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Err(FaissError { code, message })
}

struct Index {
    ptr: NonNull<FaissIndex>,
}

impl Index {
    fn factory(d: usize, description: &str) -> Result<Self> {
        let description = CString::new(description)?;
        let mut ptr = std::ptr::null_mut();

        check(unsafe {
            faiss_index_factory(&mut ptr, d as c_int, description.as_ptr(), METRIC_L2)
        })?;

        let ptr = NonNull::new(ptr).ok_or("faiss_index_factory returned a null index")?;
        Ok(Self { ptr })
    }

    fn is_trained(&self) -> bool {
        unsafe { faiss_Index_is_trained(self.ptr.as_ptr()) != 0 }
    }

    fn ntotal(&self) -> FaissIdx {
        unsafe { faiss_Index_ntotal(self.ptr.as_ptr()) }
    }

    fn train(&mut self, vectors: &[f32], d: usize) -> Result<()> {
        assert_eq!(vectors.len() % d, 0);
        check(unsafe {
            faiss_Index_train(
                self.ptr.as_ptr(),
                (vectors.len() / d) as FaissIdx,
                vectors.as_ptr(),
            )
        })?;
        Ok(())
    }

    fn add(&mut self, vectors: &[f32], d: usize) -> Result<()> {
        assert_eq!(vectors.len() % d, 0);
        check(unsafe {
            faiss_Index_add(
                self.ptr.as_ptr(),
                (vectors.len() / d) as FaissIdx,
                vectors.as_ptr(),
            )
        })?;
        Ok(())
    }

    fn search_with_params(
        &self,
        queries: &[f32],
        d: usize,
        k: usize,
        params: &HnswSearchParameters,
    ) -> Result<(Vec<f32>, Vec<FaissIdx>)> {
        assert_eq!(queries.len() % d, 0);
        let nq = queries.len() / d;
        let mut distances = vec![0.0; nq * k];
        let mut labels = vec![-1; nq * k];

        check(unsafe {
            faiss_Index_search_with_params(
                self.ptr.as_ptr(),
                nq as FaissIdx,
                queries.as_ptr(),
                k as FaissIdx,
                params.as_ptr(),
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })?;

        Ok((distances, labels))
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            faiss_Index_free(self.ptr.as_ptr());
        }
    }
}

struct HnswSearchParameters {
    ptr: NonNull<FaissSearchParametersHNSW>,
}

impl HnswSearchParameters {
    fn new(ef_search: c_int) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { faiss_SearchParametersHNSW_new(&mut ptr) })?;
        let ptr =
            NonNull::new(ptr).ok_or("faiss_SearchParametersHNSW_new returned a null pointer")?;

        unsafe {
            faiss_SearchParametersHNSW_set_efSearch(ptr.as_ptr(), ef_search);
        }

        Ok(Self { ptr })
    }

    fn as_ptr(&self) -> *const FaissSearchParameters {
        self.ptr.as_ptr().cast()
    }
}

impl Drop for HnswSearchParameters {
    fn drop(&mut self) {
        unsafe {
            faiss_SearchParametersHNSW_free(self.ptr.as_ptr());
        }
    }
}

fn main() -> Result<()> {
    let xb = make_data(NB, D);
    let xq = xb[..NQ * D].to_vec();

    let description = format!("HNSW{HNSW_M},SQ8");
    let mut index = Index::factory(D, &description)?;

    println!("index factory description: {description}");
    println!("trained before train(): {}", index.is_trained());

    index.train(&xb, D)?;
    println!("trained after train(): {}", index.is_trained());

    index.add(&xb, D)?;
    println!("ntotal after add(): {}", index.ntotal());

    let params = HnswSearchParameters::new(EF_SEARCH)?;
    let (distances, labels) = index.search_with_params(&xq, D, K, &params)?;

    println!("searching the first {NQ} database vectors with efSearch={EF_SEARCH}");
    for query in 0..NQ {
        print!("query {query}:");
        for rank in 0..K {
            let offset = query * K + rank;
            print!(" {:>6} (d={:.4})", labels[offset], distances[offset]);
        }
        println!();
    }

    Ok(())
}

fn make_data(n: usize, d: usize) -> Vec<f32> {
    let mut rng = Lcg::new(0x5eed_f00d_cafe_f00d);
    let mut vectors = vec![0.0; n * d];

    for i in 0..n {
        for j in 0..d {
            vectors[i * d + j] = rng.next_f32();
        }
        vectors[i * d] += i as f32 / 1000.0;
    }

    vectors
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.state >> 40) as u32;
        bits as f32 / (1_u32 << 24) as f32
    }
}
