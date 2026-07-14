//! Interface and implementation to RaBitQ index types.

use std::marker::PhantomData;
use std::os::raw::{c_int, c_uint};
use std::{mem, ptr};

use super::*;
use crate::error::Result;
use crate::faiss_try;
use crate::index::FlatIndexImpl;

/// Alias for the native implementation of an IVF RaBitQ index.
pub type IVFRaBitQIndex<Q> = IVFRaBitQIndexImpl<Q>;

/// Native implementation of an IVF RaBitQ index.
#[derive(Debug)]
pub struct IVFRaBitQIndexImpl<Q> {
    inner: *mut FaissIndexIVFRaBitQ,
    quantizer: PhantomData<Q>,
}

unsafe impl<Q: Send> Send for IVFRaBitQIndexImpl<Q> {}

unsafe impl<Q: Sync> Sync for IVFRaBitQIndexImpl<Q> {}

impl<Q> Drop for IVFRaBitQIndexImpl<Q> {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexIVFRaBitQ_free(self.inner);
        }
    }
}

impl<Q> IVFRaBitQIndexImpl<Q>
where
    Q: NativeIndex,
{
    /// Create a new IVF RaBitQ index with metric.
    /// The index owns the quantizer.
    pub fn new_with_metric(
        quantizer: Q,
        d: u32,
        nlist: u32,
        metric: MetricType,
        nb_bits: u8,
    ) -> Result<Self> {
        IVFRaBitQIndexImpl::new_owned(quantizer, d, nlist, metric, nb_bits)
    }

    fn new_owned(
        quantizer: Q,
        d: u32,
        nlist: u32,
        metric: MetricType,
        nb_bits: u8,
    ) -> Result<Self> {
        let index =
            IVFRaBitQIndexImpl::new_helper(&quantizer, d, nlist, metric, nb_bits, true, true)?;
        std::mem::forget(quantizer);

        Ok(index)
    }

    /// Create a new 1-bit IVF RaBitQ index with L2 metric.
    /// The index owns the quantizer.
    pub fn new_l2(quantizer: Q, d: u32, nlist: u32) -> Result<Self> {
        IVFRaBitQIndexImpl::new_l2_with_bits(quantizer, d, nlist, 1)
    }

    /// Create a new IVF RaBitQ index with L2 metric and explicit code bit width.
    /// The index owns the quantizer.
    pub fn new_l2_with_bits(quantizer: Q, d: u32, nlist: u32, nb_bits: u8) -> Result<Self> {
        IVFRaBitQIndexImpl::new_owned(quantizer, d, nlist, MetricType::L2, nb_bits)
    }

    /// Create a new 1-bit IVF RaBitQ index with IP metric.
    /// The index owns the quantizer.
    pub fn new_ip(quantizer: Q, d: u32, nlist: u32) -> Result<Self> {
        IVFRaBitQIndexImpl::new_ip_with_bits(quantizer, d, nlist, 1)
    }

    /// Create a new IVF RaBitQ index with IP metric and explicit code bit width.
    /// The index owns the quantizer.
    pub fn new_ip_with_bits(quantizer: Q, d: u32, nlist: u32, nb_bits: u8) -> Result<Self> {
        IVFRaBitQIndexImpl::new_owned(quantizer, d, nlist, MetricType::InnerProduct, nb_bits)
    }

    fn new_helper(
        quantizer: &Q,
        d: u32,
        nlist: u32,
        metric: MetricType,
        nb_bits: u8,
        own_fields: bool,
        own_invlists: bool,
    ) -> Result<Self> {
        unsafe {
            let metric_ = metric as c_uint;
            let mut inner = ptr::null_mut();
            let quantizer_ = quantizer.inner_ptr();
            faiss_try(faiss_IndexIVFRaBitQ_new_with_metric(
                &mut inner,
                quantizer_,
                d as usize,
                nlist as usize,
                metric_,
                c_int::from(own_invlists),
                nb_bits,
            ))?;

            faiss_IndexIVFRaBitQ_set_own_fields(inner, c_int::from(own_fields));
            Ok(IVFRaBitQIndexImpl {
                inner,
                quantizer: PhantomData,
            })
        }
    }

    /// Get number of possible key values.
    pub fn nlist(&self) -> u32 {
        unsafe { faiss_IndexIVFRaBitQ_nlist(self.inner_ptr()) as u32 }
    }

    /// Get number of probes at query time.
    pub fn nprobe(&self) -> u32 {
        unsafe { faiss_IndexIVFRaBitQ_nprobe(self.inner_ptr()) as u32 }
    }

    /// Set number of probes at query time.
    pub fn set_nprobe(&mut self, value: u32) {
        unsafe {
            faiss_IndexIVFRaBitQ_set_nprobe(self.inner_ptr(), value as usize);
        }
    }

    /// Get the default number of bits used to quantize queries.
    ///
    /// A value of `0` disables query quantization and uses raw fp32 values.
    pub fn qb(&self) -> u8 {
        unsafe { faiss_IndexIVFRaBitQ_qb(self.inner_ptr()) }
    }

    /// Set the default number of bits used to quantize queries.
    ///
    /// A value of `0` disables query quantization and uses raw fp32 values.
    pub fn set_qb(&mut self, value: u8) {
        unsafe {
            faiss_IndexIVFRaBitQ_set_qb(self.inner_ptr(), value);
        }
    }

    /// Get the number of bits per vector component in the RaBitQ encoding.
    pub fn nb_bits(&self) -> u8 {
        unsafe { faiss_IndexIVFRaBitQ_nb_bits(self.inner_ptr()) }
    }
}

impl<Q> NativeIndex for IVFRaBitQIndexImpl<Q> {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl<Q> FromInnerPtr for IVFRaBitQIndexImpl<Q> {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        IVFRaBitQIndexImpl {
            inner: inner_ptr as *mut FaissIndexIVFRaBitQ,
            quantizer: PhantomData,
        }
    }
}

impl<Q> Index for IVFRaBitQIndexImpl<Q> {
    fn is_trained(&self) -> bool {
        unsafe { faiss_Index_is_trained(self.inner_ptr()) != 0 }
    }

    fn ntotal(&self) -> u64 {
        unsafe { faiss_Index_ntotal(self.inner_ptr()) as u64 }
    }

    fn d(&self) -> u32 {
        unsafe { faiss_Index_d(self.inner_ptr()) as u32 }
    }

    fn metric_type(&self) -> MetricType {
        unsafe { MetricType::from_code(faiss_Index_metric_type(self.inner_ptr()) as u32).unwrap() }
    }

    fn add_with_ids(&mut self, x: &[f32], xids: &[Idx]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try(faiss_Index_add_with_ids(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
                xids.as_ptr() as *const _,
            ))?;
            Ok(())
        }
    }

    fn train(&mut self, x: &[f32]) -> Result<()> {
        unsafe {
            let n = x.len() / self.d() as usize;
            faiss_try(faiss_Index_train(self.inner_ptr(), n as i64, x.as_ptr()))?;
            Ok(())
        }
    }

    fn remove_ids(&mut self, sel: &IdSelector) -> Result<usize> {
        unsafe {
            let mut n_removed = 0;
            faiss_try(faiss_Index_remove_ids(
                self.inner_ptr(),
                sel.inner_ptr(),
                &mut n_removed,
            ))?;
            Ok(n_removed)
        }
    }

    fn search(&mut self, query: &[f32], k: usize) -> Result<SearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut distances = vec![0_f32; k * nq];
            let mut labels = vec![Idx::none(); k * nq];
            faiss_try(faiss_Index_search(
                self.inner_ptr(),
                nq as idx_t,
                query.as_ptr(),
                k as idx_t,
                distances.as_mut_ptr(),
                labels.as_mut_ptr() as *mut _,
            ))?;
            Ok(SearchResult { distances, labels })
        }
    }

    fn fetch_top_lists(
        &mut self,
        query: &[f32],
        nprobe: usize,
        list: bool,
    ) -> Result<CentroidResult> {
        unsafe {
            let mut keys = vec![Idx::none(); nprobe];
            let mut coarse_diss = vec![0.0_f32; nprobe];
            let mut label_lists = vec![vec![]; nprobe];

            if query.len() == self.d() as usize {
                if list {
                    let mut lens = vec![0; nprobe];
                    let max_size = (nprobe * 1_000_000).min(self.ntotal() as usize);
                    let mut labels = vec![Idx::none(); max_size + 65536];

                    faiss_try(faiss_Index_fetch_top_lists(
                        self.inner_ptr(),
                        query.as_ptr(),
                        nprobe as idx_t,
                        keys.as_mut_ptr() as *mut _,
                        coarse_diss.as_mut_ptr(),
                        lens.as_mut_ptr(),
                        labels.as_mut_ptr() as *mut _,
                    ))?;

                    let mut len = 0_usize;
                    let mut i = 0;
                    while i < keys.len() {
                        if keys[i].is_none() {
                            break;
                        }

                        if lens[i] > 0 {
                            label_lists[i].extend_from_slice(&labels[len..len + lens[i] as usize]);
                        }

                        len += lens[i] as usize;
                        i += 1;
                    }

                    keys.drain(i..);
                    coarse_diss.drain(i..);
                    label_lists.drain(i..);
                } else {
                    faiss_try(faiss_Index_fetch_top_lists(
                        self.inner_ptr(),
                        query.as_ptr(),
                        nprobe as idx_t,
                        keys.as_mut_ptr() as *mut _,
                        coarse_diss.as_mut_ptr(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    ))?;
                }
            }

            Ok(CentroidResult {
                keys,
                coarse_diss,
                label_lists,
            })
        }
    }

    fn search_with_filter(
        &mut self,
        query: &[f32],
        k: usize,
        centroids: &mut CentroidResult,
    ) -> Result<SearchResult> {
        assert_eq!(centroids.keys.len(), centroids.coarse_diss.len());
        assert_eq!(centroids.keys.len(), centroids.label_lists.len());

        unsafe {
            let mut labels = vec![Idx::none(); k];
            let mut distances = vec![0_f32; k];

            if query.len() == self.d() as usize {
                let mut total = 0;
                let mut lens = vec![0; centroids.label_lists.len()];
                for i in 0..centroids.label_lists.len() {
                    lens[i] = centroids.label_lists[i].len();
                    total += lens[i];
                }

                let mut curr = 0;
                let mut label_lists = vec![Idx::none(); total];
                for i in 0..centroids.label_lists.len() {
                    label_lists[curr..curr + lens[i]].copy_from_slice(&centroids.label_lists[i]);
                    curr += lens[i];
                }

                // key range [0, nlist)
                centroids.keys.push(Idx::none());

                faiss_try(faiss_Index_search_with_filter(
                    self.inner_ptr(),
                    query.as_ptr(),
                    k as idx_t,
                    centroids.keys.as_mut_ptr() as *mut _,
                    centroids.coarse_diss.as_mut_ptr(),
                    lens.as_mut_ptr() as *mut _,
                    label_lists.as_mut_ptr() as *mut _,
                    labels.as_mut_ptr() as *mut _,
                    distances.as_mut_ptr(),
                ))?;
            }
            Ok(SearchResult { distances, labels })
        }
    }
}

impl<Q> IVFRaBitQIndexImpl<Q> {
    /// Create an independent clone of this index.
    ///
    /// # Errors
    ///
    /// May result in a native error if the clone operation is not
    /// supported for the internal type of index.
    pub fn try_clone(&self) -> Result<Self> {
        unsafe {
            let mut new_index_ptr = ::std::ptr::null_mut();
            faiss_try(faiss_clone_index(self.inner_ptr(), &mut new_index_ptr))?;
            Ok(crate::index::FromInnerPtr::from_inner_ptr(new_index_ptr))
        }
    }
}

impl<Q> ConcurrentIndex for IVFRaBitQIndexImpl<Q>
where
    Q: ConcurrentIndex,
{
    fn search(&self, query: &[f32], k: usize) -> Result<SearchResult> {
        unsafe {
            let nq = query.len() / self.d() as usize;
            let mut distances = vec![0_f32; k * nq];
            let mut labels = vec![Idx::none(); k * nq];
            faiss_try(faiss_Index_search(
                self.inner,
                nq as idx_t,
                query.as_ptr(),
                k as idx_t,
                distances.as_mut_ptr(),
                labels.as_mut_ptr() as *mut _,
            ))?;
            Ok(SearchResult { distances, labels })
        }
    }

    fn fetch_top_lists(&self, query: &[f32], nprobe: usize, list: bool) -> Result<CentroidResult> {
        unsafe {
            let mut keys = vec![Idx::none(); nprobe];
            let mut coarse_diss = vec![0_f32; nprobe];
            let mut label_lists = vec![vec![]; nprobe];

            if query.len() == self.d() as usize {
                if list {
                    let mut lens = vec![0; nprobe];
                    let max_size = (nprobe * 1_000_000).min(self.ntotal() as usize);
                    let mut labels = vec![Idx::none(); max_size + 65536];

                    faiss_try(faiss_Index_fetch_top_lists(
                        self.inner_ptr(),
                        query.as_ptr(),
                        nprobe as idx_t,
                        keys.as_mut_ptr() as *mut _,
                        coarse_diss.as_mut_ptr(),
                        lens.as_mut_ptr(),
                        labels.as_mut_ptr() as *mut _,
                    ))?;

                    let mut len = 0_usize;
                    let mut i = 0;
                    while i < keys.len() {
                        if keys[i].is_none() {
                            break;
                        }

                        if lens[i] > 0 {
                            label_lists[i].extend_from_slice(&labels[len..len + lens[i] as usize]);
                        }

                        len += lens[i] as usize;
                        i += 1;
                    }

                    keys.drain(i..);
                    coarse_diss.drain(i..);
                    label_lists.drain(i..);
                } else {
                    faiss_try(faiss_Index_fetch_top_lists(
                        self.inner_ptr(),
                        query.as_ptr(),
                        nprobe as idx_t,
                        keys.as_mut_ptr() as *mut _,
                        coarse_diss.as_mut_ptr(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    ))?;
                }
            }

            Ok(CentroidResult {
                keys,
                coarse_diss,
                label_lists,
            })
        }
    }

    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        centroids: &mut CentroidResult,
    ) -> Result<SearchResult> {
        assert_eq!(centroids.keys.len(), centroids.coarse_diss.len());
        assert_eq!(centroids.keys.len(), centroids.label_lists.len());

        unsafe {
            let mut labels = vec![Idx::none(); k];
            let mut distances = vec![0_f32; k];

            if query.len() == self.d() as usize {
                let mut total = 0;
                let mut lens = vec![0; centroids.label_lists.len()];
                for i in 0..centroids.label_lists.len() {
                    lens[i] = centroids.label_lists[i].len();
                    total += lens[i];
                }

                let mut curr = 0;
                let mut label_lists = vec![Idx::none(); total];
                for i in 0..centroids.label_lists.len() {
                    label_lists[curr..curr + lens[i]].copy_from_slice(&centroids.label_lists[i]);
                    curr += lens[i];
                }

                // key range [0, nlist)
                centroids.keys.push(Idx::none());

                faiss_try(faiss_Index_search_with_filter(
                    self.inner_ptr(),
                    query.as_ptr(),
                    k as idx_t,
                    centroids.keys.as_mut_ptr() as *mut _,
                    centroids.coarse_diss.as_mut_ptr(),
                    lens.as_mut_ptr() as *mut _,
                    label_lists.as_mut_ptr() as *mut _,
                    labels.as_mut_ptr() as *mut _,
                    distances.as_mut_ptr(),
                ))?;
            }
            Ok(SearchResult { distances, labels })
        }
    }
}

impl IndexImpl {
    /// Attempt a dynamic cast of an index to the IVF RaBitQ index type.
    pub fn into_ivf_rabitq(self) -> Result<IVFRaBitQIndexImpl<FlatIndexImpl>> {
        unsafe {
            let new_inner = faiss_IndexIVFRaBitQ_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(IVFRaBitQIndexImpl {
                    inner: new_inner,
                    quantizer: PhantomData,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IVFRaBitQIndexImpl;
    use crate::index::{index_factory, ConcurrentIndex, FlatIndex, Index, UpcastIndex};
    use crate::MetricType;

    const D: u32 = 8;

    fn some_data() -> Vec<f32> {
        vec![
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ]
    }

    #[test]
    fn ivf_rabitq_index_nlist() {
        let quantizer = FlatIndex::new_l2(D).unwrap();
        let index = IVFRaBitQIndexImpl::new_l2(quantizer, D, 1).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        assert_eq!(index.nlist(), 1);
        assert_eq!(index.nb_bits(), 1);
    }

    #[test]
    fn ivf_rabitq_index_nprobe() {
        let quantizer = FlatIndex::new_l2(D).unwrap();
        let mut index = IVFRaBitQIndexImpl::new_l2(quantizer, D, 1).unwrap();
        assert_eq!(index.nprobe(), 1);

        index.set_nprobe(10);
        assert_eq!(index.nprobe(), 10);
    }

    #[test]
    fn ivf_rabitq_index_qb() {
        let quantizer = FlatIndex::new_l2(D).unwrap();
        let mut index = IVFRaBitQIndexImpl::new_l2_with_bits(quantizer, D, 1, 2).unwrap();
        assert_eq!(index.nb_bits(), 2);
        assert_eq!(index.qb(), 4);

        index.set_qb(0);
        assert_eq!(index.qb(), 0);
    }

    #[test]
    fn ivf_rabitq_index_search() {
        let quantizer = FlatIndex::new_l2(D).unwrap();
        let mut index = IVFRaBitQIndexImpl::new_l2(quantizer, D, 1).unwrap();
        let data = some_data();
        index.train(&data).unwrap();
        index.add(&data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let query = [0.; D as usize];
        let result = index.search(&query, 5).unwrap();
        assert_eq!(result.labels.len(), 5);
        assert_eq!(result.distances.len(), 5);

        let query = [100.; D as usize];
        let result = (&index).search(&query, 5).unwrap();
        assert_eq!(result.labels.len(), 5);
        assert_eq!(result.distances.len(), 5);
    }

    #[test]
    fn ivf_rabitq_index_from_cast() {
        let mut index = index_factory(8, "IVF1,RaBitQ", MetricType::L2).unwrap();
        let data = some_data();
        index.train(&data).unwrap();
        index.add(&data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let index = index.into_ivf_rabitq().unwrap();
        assert_eq!(index.is_trained(), true);
        assert_eq!(index.ntotal(), 5);
    }

    #[test]
    fn ivf_rabitq_index_upcast() {
        let quantizer = FlatIndex::new_l2(D).unwrap();
        let index = IVFRaBitQIndexImpl::new_l2(quantizer, D, 1).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let index_impl = index.upcast();
        assert_eq!(index_impl.d(), D);
    }
}
