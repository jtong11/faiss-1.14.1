/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_HNSW_C_H
#define FAISS_INDEX_HNSW_C_H

#include "Index_c.h"
#include "IndexScalarQuantizer_c.h"
#include "faiss_c.h"
#include "impl/AuxIndexStructures_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS_INHERITED(SearchParametersHNSW, SearchParameters)
FAISS_DECLARE_DESTRUCTOR(SearchParametersHNSW)
FAISS_DECLARE_SEARCH_PARAMETERS_DOWNCAST(SearchParametersHNSW)

int faiss_SearchParametersHNSW_new(FaissSearchParametersHNSW** p_sp);
int faiss_SearchParametersHNSW_new_with(
        FaissSearchParametersHNSW** p_sp,
        FaissIDSelector* sel,
        int efSearch);

FAISS_DECLARE_GETTER(SearchParametersHNSW, const FaissIDSelector*, sel)
FAISS_DECLARE_GETTER_SETTER(SearchParametersHNSW, int, efSearch)
FAISS_DECLARE_GETTER_SETTER(SearchParametersHNSW, int, check_relative_distance)
FAISS_DECLARE_GETTER_SETTER(SearchParametersHNSW, int, bounded_queue)

/** Opaque type for IndexHNSW */
FAISS_DECLARE_CLASS_INHERITED(IndexHNSW, Index)

FAISS_DECLARE_DESTRUCTOR(IndexHNSW)
FAISS_DECLARE_INDEX_DOWNCAST(IndexHNSW)

int faiss_IndexHNSW_new(FaissIndexHNSW** p_index);
int faiss_IndexHNSW_new_with(
        FaissIndexHNSW** p_index,
        int d,
        int M,
        FaissMetricType metric);
int faiss_IndexHNSW_new_with_storage(
        FaissIndexHNSW** p_index,
        FaissIndex* storage,
        int M);

/// sequential storage index used by HNSW
FAISS_DECLARE_GETTER(IndexHNSW, FaissIndex*, storage)
/// whether object owns the storage index
FAISS_DECLARE_GETTER_SETTER(IndexHNSW, int, own_fields)

/// expansion factor at construction time
FAISS_DECLARE_GETTER_SETTER(IndexHNSW, int, efConstruction)
/// expansion factor at search time
FAISS_DECLARE_GETTER_SETTER(IndexHNSW, int, efSearch)
/// pruning headroom applied when shrinking neighbors
FAISS_DECLARE_GETTER_SETTER(IndexHNSW, float, prune_headroom)
/// whether search checks if the next best distance can improve results
FAISS_DECLARE_GETTER_SETTER(IndexHNSW, int, check_relative_distance)
/// whether search uses a bounded candidate queue
FAISS_DECLARE_GETTER_SETTER(IndexHNSW, int, search_bounded_queue)

int faiss_IndexHNSW_nb_neighbors(
        const FaissIndexHNSW* index,
        int layer_no,
        int* n);
int faiss_IndexHNSW_set_nb_neighbors(
        FaissIndexHNSW* index,
        int layer_no,
        int n);

int faiss_IndexHNSW_shrink_level_0_neighbors(
        FaissIndexHNSW* index,
        int size);
int faiss_IndexHNSW_reorder_links(FaissIndexHNSW* index);
int faiss_IndexHNSW_link_singletons(FaissIndexHNSW* index);

/** Opaque type for IndexHNSWFlat */
FAISS_DECLARE_CLASS_INHERITED(IndexHNSWFlat, Index)

FAISS_DECLARE_DESTRUCTOR(IndexHNSWFlat)
FAISS_DECLARE_INDEX_DOWNCAST(IndexHNSWFlat)

int faiss_IndexHNSWFlat_new(FaissIndexHNSWFlat** p_index);
int faiss_IndexHNSWFlat_new_with(
        FaissIndexHNSWFlat** p_index,
        int d,
        int M);
int faiss_IndexHNSWFlat_new_with_metric(
        FaissIndexHNSWFlat** p_index,
        int d,
        int M,
        FaissMetricType metric);

/** Opaque type for IndexHNSWFlatPanorama */
FAISS_DECLARE_CLASS_INHERITED(IndexHNSWFlatPanorama, Index)

FAISS_DECLARE_DESTRUCTOR(IndexHNSWFlatPanorama)
FAISS_DECLARE_INDEX_DOWNCAST(IndexHNSWFlatPanorama)

int faiss_IndexHNSWFlatPanorama_new(FaissIndexHNSWFlatPanorama** p_index);
int faiss_IndexHNSWFlatPanorama_new_with(
        FaissIndexHNSWFlatPanorama** p_index,
        int d,
        int M,
        int num_panorama_levels);
int faiss_IndexHNSWFlatPanorama_new_with_metric(
        FaissIndexHNSWFlatPanorama** p_index,
        int d,
        int M,
        int num_panorama_levels,
        FaissMetricType metric);

/** Opaque type for IndexHNSWPQ */
FAISS_DECLARE_CLASS_INHERITED(IndexHNSWPQ, Index)

FAISS_DECLARE_DESTRUCTOR(IndexHNSWPQ)
FAISS_DECLARE_INDEX_DOWNCAST(IndexHNSWPQ)

int faiss_IndexHNSWPQ_new(FaissIndexHNSWPQ** p_index);
int faiss_IndexHNSWPQ_new_with(
        FaissIndexHNSWPQ** p_index,
        int d,
        int pq_m,
        int M);
int faiss_IndexHNSWPQ_new_with_metric(
        FaissIndexHNSWPQ** p_index,
        int d,
        int pq_m,
        int M,
        int pq_nbits,
        FaissMetricType metric);

/** Opaque type for IndexHNSWSQ */
FAISS_DECLARE_CLASS_INHERITED(IndexHNSWSQ, Index)

FAISS_DECLARE_DESTRUCTOR(IndexHNSWSQ)
FAISS_DECLARE_INDEX_DOWNCAST(IndexHNSWSQ)

int faiss_IndexHNSWSQ_new(FaissIndexHNSWSQ** p_index);
int faiss_IndexHNSWSQ_new_with(
        FaissIndexHNSWSQ** p_index,
        int d,
        FaissQuantizerType qtype,
        int M);
int faiss_IndexHNSWSQ_new_with_metric(
        FaissIndexHNSWSQ** p_index,
        int d,
        FaissQuantizerType qtype,
        int M,
        FaissMetricType metric);

#ifdef __cplusplus
}
#endif

#endif
