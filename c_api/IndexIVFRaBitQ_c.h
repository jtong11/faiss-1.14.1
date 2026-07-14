/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_IVF_RABITQ_C_H
#define FAISS_INDEX_IVF_RABITQ_C_H

#include "Index_c.h"
#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS_INHERITED(IVFRaBitQSearchParameters, SearchParameters)
FAISS_DECLARE_DESTRUCTOR(IVFRaBitQSearchParameters)
FAISS_DECLARE_SEARCH_PARAMETERS_DOWNCAST(IVFRaBitQSearchParameters)

int faiss_IVFRaBitQSearchParameters_new(
        FaissIVFRaBitQSearchParameters** p_sp);

int faiss_IVFRaBitQSearchParameters_new_with(
        FaissIVFRaBitQSearchParameters** p_sp,
        FaissIDSelector* sel,
        size_t nprobe,
        size_t max_codes,
        uint8_t qb,
        int centered);

FAISS_DECLARE_GETTER(IVFRaBitQSearchParameters, const FaissIDSelector*, sel)
FAISS_DECLARE_GETTER_SETTER(IVFRaBitQSearchParameters, size_t, nprobe)
FAISS_DECLARE_GETTER_SETTER(IVFRaBitQSearchParameters, size_t, max_codes)
FAISS_DECLARE_GETTER_SETTER(IVFRaBitQSearchParameters, uint8_t, qb)
FAISS_DECLARE_GETTER_SETTER(IVFRaBitQSearchParameters, int, centered)

/** Inverted-file index using RaBitQ vector encoding. */
FAISS_DECLARE_CLASS_INHERITED(IndexIVFRaBitQ, Index)
FAISS_DECLARE_DESTRUCTOR(IndexIVFRaBitQ)
FAISS_DECLARE_INDEX_DOWNCAST(IndexIVFRaBitQ)

int faiss_IndexIVFRaBitQ_new(FaissIndexIVFRaBitQ** p_index);

int faiss_IndexIVFRaBitQ_new_with(
        FaissIndexIVFRaBitQ** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist);

int faiss_IndexIVFRaBitQ_new_with_metric(
        FaissIndexIVFRaBitQ** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        FaissMetricType metric,
        int own_invlists,
        uint8_t nb_bits);

/// number of possible key values
FAISS_DECLARE_GETTER(IndexIVFRaBitQ, size_t, nlist)
/// number of probes at query time
FAISS_DECLARE_GETTER_SETTER(IndexIVFRaBitQ, size_t, nprobe)
/// quantizer that maps vectors to inverted lists
FAISS_DECLARE_GETTER(IndexIVFRaBitQ, FaissIndex*, quantizer)
/**
 * = 0: use the quantizer as index in a kmeans training
 * = 1: just pass on the training set to the train() of the quantizer
 * = 2: kmeans training on a flat index + add the centroids to the quantizer
 */
FAISS_DECLARE_GETTER(IndexIVFRaBitQ, char, quantizer_trains_alone)

/// whether object owns the quantizer
FAISS_DECLARE_GETTER_SETTER(IndexIVFRaBitQ, int, own_fields)

/// default number of bits used to quantize queries, or 0 for fp32 queries
FAISS_DECLARE_GETTER_SETTER(IndexIVFRaBitQ, uint8_t, qb)

/// number of bits per vector component in the RaBitQ encoding
FAISS_DECLARE_GETTER(IndexIVFRaBitQ, uint8_t, nb_bits)

int faiss_IndexIVFRaBitQ_add_core(
        FaissIndexIVFRaBitQ* index,
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx);

#ifdef __cplusplus
}
#endif

#endif
