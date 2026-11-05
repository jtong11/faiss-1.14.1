/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexScalarQuantizer.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <omp.h>

#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/expanded_scanners.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/utils.h>

namespace faiss {

namespace {

struct SQ8IPIntegerPreselector {
    bool enabled = false;
    bool by_residual = false;
    int dim = 0;
    float query_base = 0;
    float list_base = 0;
    float int_to_score = 0;
    float error_bound = 0;
    std::vector<int16_t> int_weights;

    void prepare(
            const ScalarQuantizer& sq,
            const float* query,
            bool by_residual_in) {
        enabled = false;
        by_residual = by_residual_in;
        dim = static_cast<int>(sq.d);
        query_base = 0;
        list_base = 0;
        int_to_score = 0;
        error_bound = 0;
        int_weights.clear();

        if (sq.qtype != ScalarQuantizer::QT_8bit &&
            sq.qtype != ScalarQuantizer::QT_8bit_uniform) {
            return;
        }

        const bool uniform = sq.qtype == ScalarQuantizer::QT_8bit_uniform;
        std::vector<float> weights(dim);
        float max_abs_weight = 0;
        double base = 0;

        for (int i = 0; i < dim; i++) {
            const float vmin = uniform ? sq.trained[0] : sq.trained[i];
            const float vdiff = uniform ? sq.trained[1] : sq.trained[dim + i];
            const float weight = query[i] * vdiff;
            weights[i] = weight;
            base += query[i] * vmin + weight * (0.5 / 255.0);
            max_abs_weight = std::max(max_abs_weight, std::abs(weight));
        }

        query_base = static_cast<float>(base);
        int_weights.assign(dim, 0);
        enabled = true;

        if (max_abs_weight == 0) {
            return;
        }

        const double scale = 32767.0 / max_abs_weight;
        int_to_score = static_cast<float>(1.0 / (scale * 255.0));

        double err = 0;
        for (int i = 0; i < dim; i++) {
            long long wi = std::llround(weights[i] * scale);
            wi = std::max<long long>(-32767, std::min<long long>(32767, wi));
            int_weights[i] = static_cast<int16_t>(wi);
            err += std::abs(weights[i] - static_cast<float>(wi / scale));
        }
        error_bound = static_cast<float>(err);
    }

    void set_list(float coarse_dis) {
        list_base = query_base + (by_residual ? coarse_dis : 0);
    }

    bool may_exceed(float threshold, const uint8_t* code) const {
        if (!enabled) {
            return true;
        }

        int64_t int_score = 0;
        for (int i = 0; i < dim; i++) {
            int_score += static_cast<int64_t>(int_weights[i]) * code[i];
        }

        const float approx_score = int_score * int_to_score;
        const float upper_bound = list_base + approx_score + error_bound;
        const float margin =
                1e-5f * (std::abs(list_base) + std::abs(approx_score) +
                          error_bound + 1.0f);
        return upper_bound + margin > threshold;
    }
};

} // namespace

/*******************************************************************
 * IndexScalarQuantizer implementation
 ********************************************************************/

IndexScalarQuantizer::IndexScalarQuantizer(
        int d_in,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric)
        : IndexFlatCodes(0, d_in, metric), sq(d_in, qtype) {
    is_trained = qtype == ScalarQuantizer::QT_fp16 ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_bf16 ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
    code_size = sq.code_size;
}

IndexScalarQuantizer::IndexScalarQuantizer()
        : IndexScalarQuantizer(0, ScalarQuantizer::QT_8bit) {}

void IndexScalarQuantizer::train(idx_t n, const float* x) {
    sq.train(n, x);
    is_trained = true;
}

void IndexScalarQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    const IDSelector* sel = params ? params->sel : nullptr;

    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);

#pragma omp parallel
    {
        std::unique_ptr<InvertedListScanner> scanner(
                sq.select_InvertedListScanner(metric_type, nullptr, true, sel));

        scanner->list_no = 0; // directly the list number

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            float* D = distances + k * i;
            idx_t* I = labels + k * i;
            // re-order heap
            if (metric_type == METRIC_L2) {
                maxheap_heapify(k, D, I);
            } else {
                minheap_heapify(k, D, I);
            }
            scanner->set_query(x + i * d);
            scanner->scan_codes(ntotal, codes.data(), nullptr, D, I, k);

            // re-order heap
            if (metric_type == METRIC_L2) {
                maxheap_reorder(k, D, I);
            } else {
                minheap_reorder(k, D, I);
            }
        }
    }
}

FlatCodesDistanceComputer* IndexScalarQuantizer::get_FlatCodesDistanceComputer()
        const {
    ScalarQuantizer::SQDistanceComputer* dc =
            sq.get_distance_computer(metric_type);
    dc->code_size = sq.code_size;
    dc->codes = codes.data();
    return dc;
}

/* Codec interface */

void IndexScalarQuantizer::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    sq.compute_codes(x, bytes, n);
}

void IndexScalarQuantizer::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    sq.decode(bytes, x, n);
}

/*******************************************************************
 * IndexIVFScalarQuantizer implementation
 ********************************************************************/

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer(
        Index* quantizer_,
        size_t d_,
        size_t nlist_,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool by_residual_,
        bool own_invlists_)
        : IndexIVF(quantizer_, d_, nlist_, 0, metric, own_invlists_),
          sq(d_, qtype) {
    code_size = sq.code_size;
    this->by_residual = by_residual_;
    if (invlists) {
        // was not known at construction time
        invlists->code_size = code_size;
    }
    is_trained = false;
    if (qtype == ScalarQuantizer::QT_0bit) {
        by_residual = false;
        is_trained = true; // no training needed
    }
    if (ScalarQuantizer::TurboQuantRefine::is_turboq_full(qtype)) {
        by_residual = false;
    }
}

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer() : IndexIVF() {
    by_residual = true;
}

void IndexIVFScalarQuantizer::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* /*assign*/) {
    sq.train(n, x);
}

idx_t IndexIVFScalarQuantizer::train_encoder_num_vectors() const {
    return 100000;
}

void IndexIVFScalarQuantizer::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    if (sq.code_size == 0) {
        // QT_0bit: nothing to encode, but handle coarse codes if needed
        if (include_listnos) {
            size_t coarse_size = coarse_code_size();
            for (idx_t i = 0; i < n; i++) {
                int64_t list_no = list_nos[i];
                if (list_no >= 0) {
                    encode_listno(list_no, codes + i * coarse_size);
                }
            }
        }
        return;
    }
    std::unique_ptr<ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());
    size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    memset(codes, 0, (code_size + coarse_size) * n);

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code = codes + i * (code_size + coarse_size);
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }
                if (coarse_size) {
                    encode_listno(list_no, code);
                }
                squant->encode_vector(xi, code + coarse_size);
            }
        }
    }
}

void IndexIVFScalarQuantizer::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* list_nos,
        float* x) const {
    if (sq.code_size == 0 && list_nos) {
        // QT_0bit: reconstruct centroids if list_nos provided
        for (idx_t i = 0; i < n; i++) {
            quantizer->reconstruct(list_nos[i], x + i * d);
        }
        return;
    }
    FAISS_THROW_IF_NOT(is_trained);
    sq.decode(codes, x, n);
    if (by_residual) {
        FAISS_THROW_IF_NOT_MSG(
                list_nos, "decode_vectors with by_residual requires list_nos");
#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; i++) {
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_nos[i], centroid.data());
            for (size_t j = 0; j < static_cast<size_t>(d); j++) {
                x[i * d + j] += centroid[j];
            }
        }
    }
}

void IndexIVFScalarQuantizer::sa_decode(idx_t n, const uint8_t* codes, float* x)
        const {
    if (sq.code_size == 0) {
        size_t coarse_size = coarse_code_size();
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * coarse_size;
            int64_t list_no = decode_listno(code);
            quantizer->reconstruct(list_no, x + i * d);
        }
        return;
    }
    std::unique_ptr<ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());
    size_t coarse_size = coarse_code_size();

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * (code_size + coarse_size);
            int64_t list_no = decode_listno(code);
            float* xi = x + i * d;
            squant->decode_vector(code + coarse_size, xi);
            if (by_residual) {
                quantizer->reconstruct(list_no, residual.data());
                for (size_t j = 0; j < static_cast<size_t>(d); j++) {
                    xi[j] += residual[j];
                }
            }
        }
    }
}

void IndexIVFScalarQuantizer::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    if (sq.code_size == 0) {
        // QT_0bit: just add IDs with empty codes
        uint8_t dummy_code = 0;
        DirectMapAdd dm_add(direct_map, n, xids);
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = coarse_idx[i];
            if (list_no >= 0) {
                int64_t id = xids ? xids[i] : ntotal + i;
                size_t ofs = invlists->add_entry(list_no, id, &dummy_code);
                dm_add.add(i, list_no, ofs);
            } else {
                dm_add.add(i, -1, 0);
            }
        }
        ntotal += n;
        return;
    }

    std::unique_ptr<ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());

    DirectMapAdd dm_add(direct_map, n, xids);

#pragma omp parallel
    {
        std::vector<float> residual(d);
        std::vector<uint8_t> one_code(code_size);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                int64_t id = xids ? xids[i] : ntotal + i;

                const float* xi = x + i * d;
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }

                memset(one_code.data(), 0, code_size);
                squant->encode_vector(xi, one_code.data());

                size_t ofs = invlists->add_entry(
                        list_no, id, one_code.data(), inverted_list_context);

                dm_add.add(i, list_no, ofs);

            } else if (rank == 0 && list_no == -1) {
                dm_add.add(i, -1, 0);
            }
        }
    }

    ntotal += n;
}

InvertedListScanner* IndexIVFScalarQuantizer::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* search_params) const {
    if (!ScalarQuantizer::TurboQuantRefine::is_turboq_full(sq.qtype)) {
        return sq.select_InvertedListScanner(
                metric_type, quantizer, store_pairs, sel, by_residual);
    }

    // TurboQ full types: create a TQ-specific scanner that supports
    // search params (qb, int_qjl) and pre-screening.
    uint8_t tq_qb = 0;
    bool tq_int_qjl = false;
    if (auto* tp = dynamic_cast<const IVFSQTurboQSearchParameters*>(
                search_params)) {
        tq_qb = tp->qb;
        tq_int_qjl = tp->int_qjl;
    }
    using TurboQDC = ScalarQuantizer::TurboQuantRefine::DistanceComputer;
    auto* dc = static_cast<TurboQDC*>(sq.get_distance_computer(metric_type));
    dc->configure(tq_qb, tq_int_qjl);

    struct TQScanner : InvertedListScanner {
        std::unique_ptr<TurboQDC> dc;

        explicit TQScanner(
                TurboQDC* dc_in,
                bool store_pairs_in,
                const IDSelector* sel_in,
                bool keep_max_in)
                : InvertedListScanner(store_pairs_in, sel_in), dc(dc_in) {
            this->keep_max = keep_max_in;
        }

        void set_query(const float* query) override {
            dc->set_query(query);
        }

        void set_list(idx_t list_no_in, float) override {
            this->list_no = list_no_in;
        }

        float distance_to_code(const uint8_t* code) const final {
            return dc->distance_to_code(code);
        }

        size_t scan_codes(
                size_t list_size,
                const uint8_t* codes,
                const idx_t* ids,
                ResultHandler& handler) const override {
            dc->set_prescreen_threshold(&handler.threshold, !keep_max);
            size_t nup = run_scan_codes(*this, list_size, codes, ids, handler);
            dc->clear_prescreen_threshold();
            return nup;
        }
    };

    auto* scanner = new TQScanner(
            dc, store_pairs, sel, is_similarity_metric(metric_type));
    scanner->code_size = code_size;
    return scanner;
}

void IndexIVFScalarQuantizer::search_with_filter(
        const float* x,
        idx_t k,
        idx_t* keys,
        float* coarse_diss,
        idx_t* lens,
        idx_t* label_lists,
        idx_t* labels,
        float* distances) const {
    if (metric_type != METRIC_INNER_PRODUCT ||
        (sq.qtype != ScalarQuantizer::QT_8bit &&
         sq.qtype != ScalarQuantizer::QT_8bit_uniform)) {
        IndexIVF::search_with_filter(
                x, k, keys, coarse_diss, lens, label_lists, labels, distances);
        return;
    }

    FAISS_THROW_IF_NOT(k > 0);

    idx_t nkeys = 0;
    while (nkeys < static_cast<idx_t>(nlist) && keys[nkeys] >= 0) {
        FAISS_THROW_IF_NOT_FMT(
                keys[nkeys] < static_cast<idx_t>(nlist),
                "Invalid key=%" PRId64 " nlist=%zd\n",
                keys[nkeys],
                nlist);
        nkeys += 1;
    }

    invlists->prefetch_lists(keys, static_cast<int>(nkeys));

    size_t nlistv = 0, ndis = 0, nheap = 0;
    float min_score = std::max(0.01f, distances[0]);
    float max_score = min_score;

    double t0 = getmillisecs();

    std::unique_ptr<InvertedListScanner> scanner(
            get_InvertedListScanner(false, nullptr, nullptr));
    scanner->set_query(x);

    SQ8IPIntegerPreselector preselector;
    preselector.prepare(sq, x, by_residual);

    using HeapForIP = CMin<float, idx_t>;
    heap_heapify<HeapForIP>(k, distances, labels);

    int pred = 0;
    int full = 0;
    int hit = 0;
    std::vector<int> hits(nkeys);
    int miss_centroid = 0;
    int stop_i = -1;
    bool collect_drops = false;
    std::vector<float> drops;

    for (idx_t i = 0; i < nkeys; i++) {
        scanner->set_list(keys[i], coarse_diss[i]);
        preselector.set_list(coarse_diss[i]);
        nlistv++;

        try {
            InvertedLists::ScopedCodes scodes(invlists, keys[i]);

            const uint8_t* codes = scodes.get();
            const size_t scanner_code_size = scanner->code_size;

            if (scanner_code_size > 0) {
                hits[i] = 0;
                for (idx_t n = 0; n < lens[i]; n++) {
                    const idx_t label = label_lists[ndis + n];
                    if (label == -1) {
                        codes += scanner_code_size;
                        continue;
                    }

                    pred++;

                    const float threshold = std::max(min_score, distances[0]);
                    if (!preselector.may_exceed(threshold, codes)) {
                        codes += scanner_code_size;
                        continue;
                    }

                    const float accu = scanner->distance_to_code(codes);
                    full++;

                    if (accu > min_score && accu > distances[0]) {
                        heap_replace_top<HeapForIP>(
                                k, distances, labels, accu, label);
                        nheap++;

                        min_score = std::max(min_score, distances[0]);
                        max_score = std::max(max_score, accu);
                        hit++;
                        hits[i]++;

                        if (collect_drops) {
                            drops.push_back(accu);
                        }
                    }

                    codes += scanner_code_size;
                }

                float hit_ratio = 1000 * hits[i] / static_cast<float>(lens[i]);
                if (hit_ratio < 1.0) {
                    miss_centroid++;
                }

                if (nkeys < nlist) {
                    if (coarse_diss[i] < min_score * 0.9 &&
                        miss_centroid >
                                std::max(200, static_cast<int>(nkeys)) *
                                        0.382) {
                        if (!collect_drops) {
                            collect_drops = true;
                            stop_i = i;
                        }

                        break;
                    }
                }
            } else {
                nheap += scanner->scan_codes(
                        lens[i],
                        scodes.get(),
                        label_lists + ndis,
                        distances,
                        labels,
                        k);
            }

            ndis += lens[i];
        } catch (const std::exception&) {
            break;
        }
    }

    heap_reorder<HeapForIP>(k, distances, labels);

    float sum = 0.0;
    for (idx_t i = 0; i < k; i++) {
        if (distances[i] >= min_score) {
            sum += distances[i];
        }
    }
    float avg_score = sum / k;
    float mean_score = distances[k / 2];

    int drop1 = 0;
    int drop2 = 0;
    int drop3 = 0;
    for (float drop_score : drops) {
        if (drop_score >= min_score) {
            drop1++;
        }
        if (drop_score >= avg_score) {
            drop2++;
        }
        if (drop_score >= mean_score) {
            drop3++;
        }
    }
    double t1 = getmillisecs();
    printf("cost %f min=%f avg=%f mean=%f max=%f stop=%d miss=%d %d %d %d %d %d %d\n",
           t1 - t0,
           min_score,
           avg_score,
           mean_score,
           max_score,
           stop_i,
           miss_centroid,
           pred,
           full,
           hit,
           drop1,
           drop2,
           drop3);
}

void IndexIVFScalarQuantizer::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    if (sq.code_size == 0) {
        // QT_0bit: reconstruct from centroid
        quantizer->reconstruct(list_no, recons);
        return;
    }
    InvertedLists::ScopedCodes sc(invlists, list_no, offset);

    if (by_residual) {
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        sq.decode(sc.get(), recons, 1);
        for (int i = 0; i < d; ++i) {
            recons[i] += centroid[i];
        }
    } else {
        sq.decode(sc.get(), recons, 1);
    }
}

} // namespace faiss
