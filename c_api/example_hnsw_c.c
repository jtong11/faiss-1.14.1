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

    FaissIndexHNSWFlat* index = NULL;
    FAISS_TRY(faiss_IndexHNSWFlat_new_with_metric(
            &index, d, 16, METRIC_L2));
    assert(faiss_Index_is_trained(index) != 0);

    FaissIndexHNSW* hnsw = faiss_IndexHNSW_cast(index);
    assert(hnsw != NULL);
    assert(faiss_IndexHNSWFlat_cast(index) == index);

    int nb_neighbors = 0;
    FAISS_TRY(faiss_IndexHNSW_nb_neighbors(hnsw, 0, &nb_neighbors));
    assert(nb_neighbors == 32);

    faiss_IndexHNSW_set_efSearch(hnsw, 32);
    assert(faiss_IndexHNSW_efSearch(hnsw) == 32);

    FAISS_TRY(faiss_Index_add(index, nb, xb));
    assert(faiss_Index_ntotal(index) == nb);

    idx_t labels[k];
    float distances[k];
    FAISS_TRY(faiss_Index_search(index, 1, xq, k, distances, labels));
    assert(labels[0] == 0);
    assert(distances[0] == 0.0f);

    faiss_IndexHNSWFlat_free(index);

    printf("example_hnsw_c: OK\n");
    return 0;
}
