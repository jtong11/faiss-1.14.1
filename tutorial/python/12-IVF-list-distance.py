# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
from faiss.contrib.inspect_tools import get_invlist


d = 8                            # dimension
nb = 2000                        # database size
nlist = 32                       # number of IVF lists
np.random.seed(1234)             # make reproducible

xb = np.random.random((nb, d)).astype("float32")
xb[:, 0] += np.arange(nb) / 1000.

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
index.train(xb)
index.add(xb)

# Pick a non-empty inverted list to inspect. The largest list makes the
# example print several vector IDs even for this small random dataset.
list_sizes = np.array([
    index.invlists.list_size(list_no)
    for list_no in range(index.nlist)
])
list_no = int(list_sizes.argmax())

# IVF list numbers correspond to centroid IDs in the coarse quantizer.
centroid = index.quantizer.reconstruct(list_no)

# For IndexIVFFlat, the inverted-list codes are the raw float32 vectors.
list_ids, list_codes = get_invlist(index.invlists, list_no)
list_vectors = list_codes.view("float32").reshape(-1, d)

# Faiss reports L2 distances as squared L2 distances.
squared_l2_distances = ((list_vectors - centroid) ** 2).sum(axis=1)

print(f"IVF list {list_no} contains {len(list_ids)} vectors")
print("first 10 distances from the list centroid:")
for doc_id, distance in zip(list_ids[:10], squared_l2_distances[:10]):
    print(f"id={doc_id:5d} squared_L2_to_centroid={distance:.6f}")
