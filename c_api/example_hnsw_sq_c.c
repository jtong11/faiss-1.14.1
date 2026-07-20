/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "IndexHNSW_c.h"
#include "Index_c.h"
#include "error_c.h"

#define FAISS_TRY(C)                                       \
    {                                                      \
        if (C) {                                           \
            fprintf(stderr, "%s", faiss_get_last_error()); \
            exit(-1);                                      \
        }                                                  \
    }

int main(void) {
    enum {
        d = 4,
        nb = 5,
        k = 2,
    };

    const float xb[d * nb] = {
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    const float xq[d] = {0.0f, 0.0f, 0.0f, 0.0f};

    FaissIndexHNSWSQ* index = NULL;
    FAISS_TRY(faiss_IndexHNSWSQ_new_with_metric(
            &index, d, QT_fp16, 16, METRIC_L2));
    assert(faiss_Index_is_trained(index) != 0);
    assert(faiss_IndexHNSWSQ_cast(index) == index);

    FAISS_TRY(faiss_Index_add(index, nb, xb));
    assert(faiss_Index_ntotal(index) == nb);

    idx_t labels[k];
    float distances[k];
    FAISS_TRY(faiss_Index_search(index, 1, xq, k, distances, labels));
    assert(labels[0] == 0);
    assert(distances[0] == 0.0f);

    faiss_IndexHNSWSQ_free(index);

    printf("example_hnsw_sq_c: OK\n");
    return 0;
}
