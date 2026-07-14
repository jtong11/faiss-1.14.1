/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexIVFRaBitQ_c.h"
#include <faiss/IndexIVFRaBitQ.h>
#include "Index_c.h"
#include "macros_impl.h"

using faiss::IVFRaBitQSearchParameters;
using faiss::Index;
using faiss::IndexIVFRaBitQ;
using faiss::MetricType;

/// IVFRaBitQSearchParameters definitions

DEFINE_DESTRUCTOR(IVFRaBitQSearchParameters)
DEFINE_SEARCH_PARAMETERS_DOWNCAST(IVFRaBitQSearchParameters)

int faiss_IVFRaBitQSearchParameters_new(
        FaissIVFRaBitQSearchParameters** p_sp) {
    try {
        IVFRaBitQSearchParameters* sp = new IVFRaBitQSearchParameters;
        *p_sp = reinterpret_cast<FaissIVFRaBitQSearchParameters*>(sp);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IVFRaBitQSearchParameters_new_with(
        FaissIVFRaBitQSearchParameters** p_sp,
        FaissIDSelector* sel,
        size_t nprobe,
        size_t max_codes,
        uint8_t qb,
        int centered) {
    try {
        IVFRaBitQSearchParameters* sp = new IVFRaBitQSearchParameters;
        sp->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        sp->nprobe = nprobe;
        sp->max_codes = max_codes;
        sp->qb = qb;
        sp->centered = static_cast<bool>(centered);
        *p_sp = reinterpret_cast<FaissIVFRaBitQSearchParameters*>(sp);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER_PERMISSIVE(
        IVFRaBitQSearchParameters,
        const FaissIDSelector*,
        sel)

DEFINE_GETTER(IVFRaBitQSearchParameters, size_t, nprobe)
DEFINE_SETTER(IVFRaBitQSearchParameters, size_t, nprobe)

DEFINE_GETTER(IVFRaBitQSearchParameters, size_t, max_codes)
DEFINE_SETTER(IVFRaBitQSearchParameters, size_t, max_codes)

DEFINE_GETTER(IVFRaBitQSearchParameters, uint8_t, qb)
DEFINE_SETTER(IVFRaBitQSearchParameters, uint8_t, qb)

DEFINE_GETTER(IVFRaBitQSearchParameters, int, centered)
DEFINE_SETTER(IVFRaBitQSearchParameters, int, centered)

/// IndexIVFRaBitQ definitions

DEFINE_DESTRUCTOR(IndexIVFRaBitQ)
DEFINE_INDEX_DOWNCAST(IndexIVFRaBitQ)

/// number of possible key values
DEFINE_GETTER(IndexIVFRaBitQ, size_t, nlist)
/// number of probes at query time
DEFINE_GETTER(IndexIVFRaBitQ, size_t, nprobe)
DEFINE_SETTER(IndexIVFRaBitQ, size_t, nprobe)

/// quantizer that maps vectors to inverted lists
DEFINE_GETTER_PERMISSIVE(IndexIVFRaBitQ, FaissIndex*, quantizer)

/**
 * = 0: use the quantizer as index in a kmeans training
 * = 1: just pass on the training set to the train() of the quantizer
 * = 2: kmeans training on a flat index + add the centroids to the quantizer
 */
DEFINE_GETTER(IndexIVFRaBitQ, char, quantizer_trains_alone)

/// whether object owns the quantizer
DEFINE_GETTER(IndexIVFRaBitQ, int, own_fields)
DEFINE_SETTER(IndexIVFRaBitQ, int, own_fields)

/// default number of bits used to quantize queries, or 0 for fp32 queries
DEFINE_GETTER(IndexIVFRaBitQ, uint8_t, qb)
DEFINE_SETTER(IndexIVFRaBitQ, uint8_t, qb)

uint8_t faiss_IndexIVFRaBitQ_nb_bits(const FaissIndexIVFRaBitQ* index) {
    return reinterpret_cast<const IndexIVFRaBitQ*>(index)->rabitq.nb_bits;
}

int faiss_IndexIVFRaBitQ_new(FaissIndexIVFRaBitQ** p_index) {
    try {
        *p_index =
                reinterpret_cast<FaissIndexIVFRaBitQ*>(new IndexIVFRaBitQ());
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFRaBitQ_new_with(
        FaissIndexIVFRaBitQ** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist) {
    try {
        auto q = reinterpret_cast<Index*>(quantizer);
        *p_index = reinterpret_cast<FaissIndexIVFRaBitQ*>(
                new IndexIVFRaBitQ(q, d, nlist));
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFRaBitQ_new_with_metric(
        FaissIndexIVFRaBitQ** p_index,
        FaissIndex* quantizer,
        size_t d,
        size_t nlist,
        FaissMetricType metric,
        int own_invlists,
        uint8_t nb_bits) {
    try {
        auto q = reinterpret_cast<Index*>(quantizer);
        auto mt = static_cast<MetricType>(metric);
        *p_index = reinterpret_cast<FaissIndexIVFRaBitQ*>(
                new IndexIVFRaBitQ(
                        q,
                        d,
                        nlist,
                        mt,
                        static_cast<bool>(own_invlists),
                        nb_bits));
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_IndexIVFRaBitQ_add_core(
        FaissIndexIVFRaBitQ* index,
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx) {
    try {
        reinterpret_cast<IndexIVFRaBitQ*>(index)->add_core(
                n, x, xids, precomputed_idx);
        return 0;
    }
    CATCH_AND_HANDLE
}
