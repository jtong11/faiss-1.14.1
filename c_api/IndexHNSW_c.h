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
