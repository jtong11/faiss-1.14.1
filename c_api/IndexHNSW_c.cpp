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
