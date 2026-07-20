/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Minimal Rust binding example for `c_api/IndexHNSW_c.h`.
//!
//! Build this example from a configured Faiss build tree by linking against
//! `faiss_c`, for example:
//!
//! ```text
//! rustc c_api/example_hnsw_sq_rust.rs \
//!     -L build/c_api -L build/faiss -l faiss_c -l faiss
//! ```

use std::error::Error;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;

#[allow(non_camel_case_types)]
type idx_t = i64;

#[repr(C)]
pub struct FaissIndex {
    _private: [u8; 0],
}

#[repr(C)]
pub struct FaissIndexHNSWSQ {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaissMetricType {
    METRIC_INNER_PRODUCT = 0,
    METRIC_L2 = 1,
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaissQuantizerType {
    QT_8bit = 0,
    QT_4bit = 1,
    QT_8bit_uniform = 2,
    QT_4bit_uniform = 3,
    QT_fp16 = 4,
    QT_8bit_direct = 5,
    QT_6bit = 6,
    QT_bf16 = 7,
    QT_8bit_direct_signed = 8,
}

unsafe extern "C" {
    fn faiss_get_last_error() -> *const c_char;

    fn faiss_Index_d(index: *const FaissIndex) -> c_int;
    fn faiss_Index_ntotal(index: *const FaissIndex) -> idx_t;
    fn faiss_Index_train(index: *mut FaissIndex, n: idx_t, x: *const c_float) -> c_int;
    fn faiss_Index_add(index: *mut FaissIndex, n: idx_t, x: *const c_float) -> c_int;
    fn faiss_Index_search(
        index: *const FaissIndex,
        n: idx_t,
        x: *const c_float,
        k: idx_t,
        distances: *mut c_float,
        labels: *mut idx_t,
    ) -> c_int;

    fn faiss_IndexHNSWSQ_new_with_metric(
        p_index: *mut *mut FaissIndexHNSWSQ,
        d: c_int,
        qtype: FaissQuantizerType,
        m: c_int,
        metric: FaissMetricType,
    ) -> c_int;
    fn faiss_IndexHNSWSQ_free(index: *mut FaissIndexHNSWSQ);
}

#[derive(Debug)]
pub struct FaissError(String);

impl fmt::Display for FaissError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for FaissError {}

type Result<T> = std::result::Result<T, FaissError>;

fn faiss_try(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }

    let message = unsafe {
        let error = faiss_get_last_error();
        if error.is_null() {
            format!("Faiss error code {code}")
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    };
    Err(FaissError(message))
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub distances: Vec<f32>,
    pub labels: Vec<idx_t>,
}

#[derive(Debug)]
pub struct HnswScalarQuantizerIndex {
    inner: *mut FaissIndexHNSWSQ,
}

unsafe impl Send for HnswScalarQuantizerIndex {}
unsafe impl Sync for HnswScalarQuantizerIndex {}

impl Drop for HnswScalarQuantizerIndex {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexHNSWSQ_free(self.inner);
        }
    }
}

impl HnswScalarQuantizerIndex {
    pub fn new_l2(d: u32, qtype: FaissQuantizerType, m: u32) -> Result<Self> {
        Self::new_with_metric(d, qtype, m, FaissMetricType::METRIC_L2)
    }

    pub fn new_ip(d: u32, qtype: FaissQuantizerType, m: u32) -> Result<Self> {
        Self::new_with_metric(d, qtype, m, FaissMetricType::METRIC_INNER_PRODUCT)
    }

    pub fn new_with_metric(
        d: u32,
        qtype: FaissQuantizerType,
        m: u32,
        metric: FaissMetricType,
    ) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexHNSWSQ_new_with_metric(
                &mut inner,
                d as c_int,
                qtype,
                m as c_int,
                metric,
            ))?;
            Ok(Self { inner })
        }
    }

    fn index_ptr(&self) -> *mut FaissIndex {
        self.inner.cast()
    }

    pub fn d(&self) -> u32 {
        unsafe { faiss_Index_d(self.index_ptr()) as u32 }
    }

    pub fn ntotal(&self) -> u64 {
        unsafe { faiss_Index_ntotal(self.index_ptr()) as u64 }
    }

    pub fn train(&mut self, x: &[f32]) -> Result<()> {
        let n = self.vector_count(x)?;
        unsafe { faiss_try(faiss_Index_train(self.index_ptr(), n, x.as_ptr())) }
    }

    pub fn add(&mut self, x: &[f32]) -> Result<()> {
        let n = self.vector_count(x)?;
        unsafe { faiss_try(faiss_Index_add(self.index_ptr(), n, x.as_ptr())) }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<SearchResult> {
        let nq = self.vector_count(query)?;
        let mut distances = vec![0.0f32; k * nq as usize];
        let mut labels = vec![-1; k * nq as usize];

        unsafe {
            faiss_try(faiss_Index_search(
                self.index_ptr(),
                nq,
                query.as_ptr(),
                k as idx_t,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            ))?;
        }

        Ok(SearchResult { distances, labels })
    }

    fn vector_count(&self, x: &[f32]) -> Result<idx_t> {
        let d = self.d() as usize;
        if d == 0 || x.len() % d != 0 {
            return Err(FaissError(format!(
                "input length {} is not a multiple of index dimension {}",
                x.len(),
                d
            )));
        }
        Ok((x.len() / d) as idx_t)
    }
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    const D: u32 = 4;

    let mut index =
        HnswScalarQuantizerIndex::new_l2(D, FaissQuantizerType::QT_fp16, 16)?;

    let xb = [
        0.0_f32, 0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ];

    index.add(&xb)?;
    let result = index.search(&[0.0; D as usize], 2)?;

    println!("ntotal = {}", index.ntotal());
    println!("labels = {:?}", result.labels);
    println!("distances = {:?}", result.distances);

    Ok(())
}
