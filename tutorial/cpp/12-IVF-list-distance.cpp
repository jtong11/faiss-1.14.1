/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/invlists/InvertedLists.h>

float squared_l2(const float* x, const float* y, int d) {
    float distance = 0;
    for (int i = 0; i < d; i++) {
        const float diff = x[i] - y[i];
        distance += diff * diff;
    }
    return distance;
}

int main() {
    int d = 8;      // dimension
    int nb = 2000;  // database size
    int nlist = 32; // number of IVF lists

    std::mt19937 rng(1234);
    std::uniform_real_distribution<> distrib;

    std::vector<float> xb(d * nb);
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) {
            xb[d * i + j] = distrib(rng);
        }
        xb[d * i] += i / 1000.;
    }

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFScalarQuantizer index(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_8bit);
    index.train(nb, xb.data());
    index.add(nb, xb.data());

    // Pick a non-empty inverted list to inspect. The largest list makes the
    // example print several vector IDs even for this small random dataset.
    size_t list_no = 0;
    size_t list_size = 0;
    for (size_t i = 0; i < index.nlist; i++) {
        const size_t current_size = index.invlists->list_size(i);
        if (current_size > list_size) {
            list_no = i;
            list_size = current_size;
        }
    }
    assert(list_size > 0);

    // IVF list numbers correspond to centroid IDs in the coarse quantizer.
    std::vector<float> centroid(d);
    index.quantizer->reconstruct(list_no, centroid.data());

    const faiss::InvertedLists* invlists = index.invlists;
    faiss::InvertedLists::ScopedIds ids(invlists, list_no);

    printf("IVF list %zu contains %zu vectors\n", list_no, list_size);
    printf("first 10 distances from the list centroid:\n");

    const size_t nprint = std::min<size_t>(10, list_size);
    std::vector<float> list_vector(d);
    for (size_t offset = 0; offset < nprint; offset++) {
        // IVF-SQ8 stores quantized residual codes. Reconstruct the approximate
        // vector for this list entry before comparing it to the centroid.
        index.reconstruct_from_offset(list_no, offset, list_vector.data());
        const float distance =
                squared_l2(list_vector.data(), centroid.data(), d);
        printf("id=%5zd squared_L2_to_centroid=%f\n", ids[offset], distance);
    }

    return 0;
}
