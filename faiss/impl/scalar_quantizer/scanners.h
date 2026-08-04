/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Private implementation header for per-SIMD scalar quantizer TUs.
// Do not include in public APIs.

#pragma once

#include <cstring>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/simd_levels.h>

#include <faiss/impl/simd_dispatch.h>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/expanded_scanners.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

namespace faiss {

namespace scalar_quantizer {

using QuantizerType = ScalarQuantizer::QuantizerType;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

template <class DCClass>
float reconstructed_l2sqr(const DCClass& dc, const uint8_t* code) {
    if constexpr (requires(const DCClass& dc0, const uint8_t* code0, size_t i) {
                      dc0.quant.d;
                      dc0.quant.reconstruct_component(code0, i);
                  }) {
        float norm = 0;
        for (size_t i = 0; i < dc.quant.d; i++) {
            const float xi = dc.quant.reconstruct_component(code, i);
            norm += xi * xi;
        }
        return norm;
    } else {
        FAISS_THROW_MSG("residual norm correction is unsupported");
    }
}

/*******************************************************************
 * IVFSQScannerIP / IVFSQScannerL2 — moved from anonymous namespace
 * in ScalarQuantizer.cpp
 *******************************************************************/

template <class DCClass>
struct IVFSQScannerIP : InvertedListScanner {
    DCClass dc;
    bool by_residual;

    float accu0; /// added to all distances

    IVFSQScannerIP(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained), by_residual(by_residual), accu0(0) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
        this->keep_max = true;
    }

    void set_query(const float* query) override {
        dc.set_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        accu0 = by_residual ? coarse_dis : 0;
    }

    float distance_to_code(const uint8_t* code) const final {
        return accu0 + dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return run_scan_codes_fix_C<CMin<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

template <class DCClass>
struct IVFSQScannerL2 : InvertedListScanner {
    DCClass dc;

    bool by_residual;
    const Index* quantizer;
    const float* x; /// current query
    bool use_residual_norms;
    size_t residual_norm_offset;

    std::vector<float> tmp;

    IVFSQScannerL2(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            const Index* quantizer,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual,
            bool use_residual_norms = false,
            size_t residual_norm_offset = 0)
            : dc(d, trained),
              by_residual(by_residual),
              quantizer(quantizer),
              x(nullptr),
              use_residual_norms(use_residual_norms),
              residual_norm_offset(residual_norm_offset),
              tmp(d) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        x = query;
        if (!quantizer) {
            dc.set_query(query);
        }
    }

    void set_list(idx_t list_no, float /*coarse_dis*/) override {
        this->list_no = list_no;
        if (by_residual) {
            quantizer->compute_residual(x, tmp.data(), list_no);
            dc.set_query(tmp.data());
        } else {
            dc.set_query(x);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        const float dis = dc.query_to_code(code);
        if (!use_residual_norms) {
            return dis;
        }

        float stored_norm = 0;
        std::memcpy(
                &stored_norm, code + residual_norm_offset, sizeof(stored_norm));
        return dis + stored_norm - reconstructed_l2sqr(dc, code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return run_scan_codes_fix_C<CMax<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

/*******************************************************************
 * Forward declaration of inverts list scanner
 *******************************************************************/

template <SIMDLevel SL>
InvertedListScanner* sq_select_InvertedListScanner(
        QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual,
        bool use_residual_norms = false,
        size_t residual_norm_offset = 0);

/// Scanner for QT_0bit / centroid-only distance: always returns the
/// coarse distance that was set via set_list().
struct IVFCoarseDistanceScanner : InvertedListScanner {
    float coarse_dis = 0;

    IVFCoarseDistanceScanner(
            bool is_similarity,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel) {
        code_size = 0;
        keep_max = is_similarity;
    }

    void set_query(const float* /*query_vector*/) override {}

    void set_list(idx_t list_no_in, float coarse_dis_in) override {
        this->list_no = list_no_in;
        this->coarse_dis = coarse_dis_in;
    }

    float distance_to_code(const uint8_t* /*code*/) const override {
        return coarse_dis;
    }
};

} // namespace scalar_quantizer

} // namespace faiss
