/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexHNSW_c.h"
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include "macros_impl.h"

using faiss::Index;
using faiss::IndexHNSW;
using faiss::IndexHNSWFlat;
using faiss::IndexHNSWFlatPanorama;
using faiss::IndexHNSWPQ;
using faiss::IndexHNSWSQ;
using faiss::SearchParametersHNSW;

/// SearchParametersHNSW definitions

DEFINE_DESTRUCTOR(SearchParametersHNSW)
DEFINE_SEARCH_PARAMETERS_DOWNCAST(SearchParametersHNSW)

int faiss_SearchParametersHNSW_new(FaissSearchParametersHNSW** p_sp) {
    try {
        SearchParametersHNSW* sp = new SearchParametersHNSW;
        *p_sp = reinterpret_cast<FaissSearchParametersHNSW*>(sp);
        return 0;
    }
    CATCH_AND_HANDLE
}

int faiss_SearchParametersHNSW_new_with(
        FaissSearchParametersHNSW** p_sp,
        FaissIDSelector* sel,
        int efSearch) {
    try {
        SearchParametersHNSW* sp = new SearchParametersHNSW;
        sp->sel = reinterpret_cast<faiss::IDSelector*>(sel);
        sp->efSearch = efSearch;
        *p_sp = reinterpret_cast<FaissSearchParametersHNSW*>(sp);
        return 0;
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER_PERMISSIVE(SearchParametersHNSW, const FaissIDSelector*, sel)

DEFINE_GETTER(SearchParametersHNSW, int, efSearch)
DEFINE_SETTER(SearchParametersHNSW, int, efSearch)

DEFINE_GETTER(SearchParametersHNSW, int, check_relative_distance)
DEFINE_SETTER(SearchParametersHNSW, int, check_relative_distance)

DEFINE_GETTER(SearchParametersHNSW, int, bounded_queue)
DEFINE_SETTER(SearchParametersHNSW, int, bounded_queue)

/// IndexHNSW definitions

DEFINE_DESTRUCTOR(IndexHNSW)
DEFINE_INDEX_DOWNCAST(IndexHNSW)

int faiss_IndexHNSW_new(FaissIndexHNSW** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSW*>(new IndexHNSW());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSW_new_with(
        FaissIndexHNSW** p_index,
        int d,
        int M,
        FaissMetricType metric) {
    try {
        IndexHNSW* index =
                new IndexHNSW(d, M, static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexHNSW*>(index);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSW_new_with_storage(
        FaissIndexHNSW** p_index,
        FaissIndex* storage,
        int M) {
    try {
        IndexHNSW* index = new IndexHNSW(reinterpret_cast<Index*>(storage), M);
        *p_index = reinterpret_cast<FaissIndexHNSW*>(index);
    }
    CATCH_AND_HANDLE
}

FaissIndex* faiss_IndexHNSW_storage(const FaissIndexHNSW* index) {
    return reinterpret_cast<FaissIndex*>(
            reinterpret_cast<const IndexHNSW*>(index)->storage);
}

DEFINE_GETTER(IndexHNSW, int, own_fields)
DEFINE_SETTER_STATIC(IndexHNSW, bool, int, own_fields)

int faiss_IndexHNSW_efConstruction(const FaissIndexHNSW* index) {
    return reinterpret_cast<const IndexHNSW*>(index)->hnsw.efConstruction;
}

void faiss_IndexHNSW_set_efConstruction(
        FaissIndexHNSW* index,
        int efConstruction) {
    reinterpret_cast<IndexHNSW*>(index)->hnsw.efConstruction = efConstruction;
}

int faiss_IndexHNSW_efSearch(const FaissIndexHNSW* index) {
    return reinterpret_cast<const IndexHNSW*>(index)->hnsw.efSearch;
}

void faiss_IndexHNSW_set_efSearch(FaissIndexHNSW* index, int efSearch) {
    reinterpret_cast<IndexHNSW*>(index)->hnsw.efSearch = efSearch;
}

float faiss_IndexHNSW_prune_headroom(const FaissIndexHNSW* index) {
    return reinterpret_cast<const IndexHNSW*>(index)->hnsw.prune_headroom;
}

void faiss_IndexHNSW_set_prune_headroom(
        FaissIndexHNSW* index,
        float prune_headroom) {
    reinterpret_cast<IndexHNSW*>(index)->hnsw.prune_headroom = prune_headroom;
}

int faiss_IndexHNSW_check_relative_distance(const FaissIndexHNSW* index) {
    return static_cast<int>(
            reinterpret_cast<const IndexHNSW*>(index)
                    ->hnsw.check_relative_distance);
}

void faiss_IndexHNSW_set_check_relative_distance(
        FaissIndexHNSW* index,
        int check_relative_distance) {
    reinterpret_cast<IndexHNSW*>(index)->hnsw.check_relative_distance =
            static_cast<bool>(check_relative_distance);
}

int faiss_IndexHNSW_search_bounded_queue(const FaissIndexHNSW* index) {
    return static_cast<int>(
            reinterpret_cast<const IndexHNSW*>(index)
                    ->hnsw.search_bounded_queue);
}

void faiss_IndexHNSW_set_search_bounded_queue(
        FaissIndexHNSW* index,
        int search_bounded_queue) {
    reinterpret_cast<IndexHNSW*>(index)->hnsw.search_bounded_queue =
            static_cast<bool>(search_bounded_queue);
}

int faiss_IndexHNSW_nb_neighbors(
        const FaissIndexHNSW* index,
        int layer_no,
        int* n) {
    try {
        *n = reinterpret_cast<const IndexHNSW*>(index)->hnsw.nb_neighbors(
                layer_no);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSW_set_nb_neighbors(
        FaissIndexHNSW* index,
        int layer_no,
        int n) {
    try {
        reinterpret_cast<IndexHNSW*>(index)->hnsw.set_nb_neighbors(
                layer_no, n);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSW_shrink_level_0_neighbors(
        FaissIndexHNSW* index,
        int size) {
    try {
        reinterpret_cast<IndexHNSW*>(index)->shrink_level_0_neighbors(size);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSW_reorder_links(FaissIndexHNSW* index) {
    try {
        reinterpret_cast<IndexHNSW*>(index)->reorder_links();
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSW_link_singletons(FaissIndexHNSW* index) {
    try {
        reinterpret_cast<IndexHNSW*>(index)->link_singletons();
    }
    CATCH_AND_HANDLE
}

/// IndexHNSWFlat definitions

DEFINE_DESTRUCTOR(IndexHNSWFlat)
DEFINE_INDEX_DOWNCAST(IndexHNSWFlat)

int faiss_IndexHNSWFlat_new(FaissIndexHNSWFlat** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWFlat*>(
                new IndexHNSWFlat());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWFlat_new_with(
        FaissIndexHNSWFlat** p_index,
        int d,
        int M) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWFlat*>(
                new IndexHNSWFlat(d, M));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWFlat_new_with_metric(
        FaissIndexHNSWFlat** p_index,
        int d,
        int M,
        FaissMetricType metric) {
    try {
        IndexHNSWFlat* index =
                new IndexHNSWFlat(d, M, static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexHNSWFlat*>(index);
    }
    CATCH_AND_HANDLE
}

/// IndexHNSWFlatPanorama definitions

DEFINE_DESTRUCTOR(IndexHNSWFlatPanorama)
DEFINE_INDEX_DOWNCAST(IndexHNSWFlatPanorama)

int faiss_IndexHNSWFlatPanorama_new(
        FaissIndexHNSWFlatPanorama** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWFlatPanorama*>(
                new IndexHNSWFlatPanorama());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWFlatPanorama_new_with(
        FaissIndexHNSWFlatPanorama** p_index,
        int d,
        int M,
        int num_panorama_levels) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWFlatPanorama*>(
                new IndexHNSWFlatPanorama(d, M, num_panorama_levels));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWFlatPanorama_new_with_metric(
        FaissIndexHNSWFlatPanorama** p_index,
        int d,
        int M,
        int num_panorama_levels,
        FaissMetricType metric) {
    try {
        IndexHNSWFlatPanorama* index = new IndexHNSWFlatPanorama(
                d,
                M,
                num_panorama_levels,
                static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexHNSWFlatPanorama*>(index);
    }
    CATCH_AND_HANDLE
}

/// IndexHNSWPQ definitions

DEFINE_DESTRUCTOR(IndexHNSWPQ)
DEFINE_INDEX_DOWNCAST(IndexHNSWPQ)

int faiss_IndexHNSWPQ_new(FaissIndexHNSWPQ** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWPQ*>(new IndexHNSWPQ());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWPQ_new_with(
        FaissIndexHNSWPQ** p_index,
        int d,
        int pq_m,
        int M) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWPQ*>(
                new IndexHNSWPQ(d, pq_m, M));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWPQ_new_with_metric(
        FaissIndexHNSWPQ** p_index,
        int d,
        int pq_m,
        int M,
        int pq_nbits,
        FaissMetricType metric) {
    try {
        IndexHNSWPQ* index = new IndexHNSWPQ(
                d, pq_m, M, pq_nbits, static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexHNSWPQ*>(index);
    }
    CATCH_AND_HANDLE
}

/// IndexHNSWSQ definitions

DEFINE_DESTRUCTOR(IndexHNSWSQ)
DEFINE_INDEX_DOWNCAST(IndexHNSWSQ)

int faiss_IndexHNSWSQ_new(FaissIndexHNSWSQ** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWSQ*>(new IndexHNSWSQ());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWSQ_new_with(
        FaissIndexHNSWSQ** p_index,
        int d,
        FaissQuantizerType qtype,
        int M) {
    try {
        *p_index = reinterpret_cast<FaissIndexHNSWSQ*>(
                new IndexHNSWSQ(
                        d,
                        static_cast<faiss::ScalarQuantizer::QuantizerType>(
                                qtype),
                        M));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexHNSWSQ_new_with_metric(
        FaissIndexHNSWSQ** p_index,
        int d,
        FaissQuantizerType qtype,
        int M,
        FaissMetricType metric) {
    try {
        IndexHNSWSQ* index = new IndexHNSWSQ(
                d,
                static_cast<faiss::ScalarQuantizer::QuantizerType>(qtype),
                M,
                static_cast<faiss::MetricType>(metric));
        *p_index = reinterpret_cast<FaissIndexHNSWSQ*>(index);
    }
    CATCH_AND_HANDLE
}
