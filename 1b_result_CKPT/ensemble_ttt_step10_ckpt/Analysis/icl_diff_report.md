# ICL checkpoint difference summary

Baseline: `tabicl-classifier-v1.1-20250506.ckpt`
Compared tensors: `icl_predictor.*`

## Dataset overview

| dataset   |   num_icl_tensors |    numel |   mean_rel_l2 |   max_rel_l2 |   mean_abs_delta |   max_abs_delta |   mean_changed_fraction_gt_1e_12 |   max_changed_fraction_gt_1e_12 |   min_cosine_similarity |
|:----------|------------------:|---------:|--------------:|-------------:|-----------------:|----------------:|---------------------------------:|--------------------------------:|------------------------:|
| Amazon    |               152 | 25775626 |    0.00088889 |   0.00266539 |      2.50233e-05 |     5.08092e-05 |                         0.994153 |                               1 |                0.999996 |
| BLE       |               152 | 25775626 |    0.00107009 |   0.00320656 |      3.10188e-05 |     5.08167e-05 |                         0.995192 |                               1 |                0.999995 |
| BNG       |               152 | 25775626 |    0.00110254 |   0.00349402 |      3.2169e-05  |     5.08092e-05 |                         0.993334 |                               1 |                0.999994 |

## Top changed tensors by dataset

| dataset   | key                                               | layer    | group         |   shape |     rel_l2 |   mean_abs_delta |   max_abs_delta |   changed_fraction_gt_1e_12 |   cosine_similarity |
|:----------|:--------------------------------------------------|:---------|:--------------|--------:|-----------:|-----------------:|----------------:|----------------------------:|--------------------:|
| Amazon    | icl_predictor.tf_icl.blocks.2.linear2.bias        | block_02 | ffn.linear2   |     512 | 0.00266539 |      2.51441e-05 |     5.04632e-05 |                           1 |            0.999996 |
| Amazon    | icl_predictor.tf_icl.blocks.11.linear2.bias       | block_11 | ffn.linear2   |     512 | 0.00220013 |      2.76482e-05 |     4.80004e-05 |                           1 |            0.999998 |
| Amazon    | icl_predictor.tf_icl.blocks.11.attn.out_proj.bias | block_11 | attn.out_proj |     512 | 0.00197296 |      2.69568e-05 |     4.75496e-05 |                           1 |            0.999998 |
| Amazon    | icl_predictor.tf_icl.blocks.2.attn.out_proj.bias  | block_02 | attn.out_proj |     512 | 0.00197124 |      2.36057e-05 |     4.94412e-05 |                           1 |            0.999998 |
| Amazon    | icl_predictor.tf_icl.blocks.1.attn.out_proj.bias  | block_01 | attn.out_proj |     512 | 0.00194496 |      2.25518e-05 |     5.04106e-05 |                           1 |            0.999998 |
| Amazon    | icl_predictor.tf_icl.blocks.1.linear2.bias        | block_01 | ffn.linear2   |     512 | 0.00186645 |      2.17385e-05 |     5.00968e-05 |                           1 |            0.999998 |
| Amazon    | icl_predictor.tf_icl.blocks.4.linear2.bias        | block_04 | ffn.linear2   |     512 | 0.00182479 |      2.38777e-05 |     4.98034e-05 |                           1 |            0.999998 |
| Amazon    | icl_predictor.tf_icl.blocks.3.linear2.bias        | block_03 | ffn.linear2   |     512 | 0.00181245 |      2.31269e-05 |     5.0406e-05  |                           1 |            0.999998 |
| Amazon    | icl_predictor.tf_icl.blocks.0.linear2.bias        | block_00 | ffn.linear2   |     512 | 0.00179842 |      2.31375e-05 |     5.0359e-05  |                           1 |            0.999998 |
| Amazon    | icl_predictor.ln.bias                             | ln       | ln            |     512 | 0.00178479 |      2.83739e-05 |     4.67049e-05 |                           1 |            0.999998 |
| BLE       | icl_predictor.tf_icl.blocks.2.linear2.bias        | block_02 | ffn.linear2   |     512 | 0.00320656 |      3.10779e-05 |     5.0731e-05  |                           1 |            0.999995 |
| BLE       | icl_predictor.tf_icl.blocks.11.linear2.bias       | block_11 | ffn.linear2   |     512 | 0.00279729 |      3.3463e-05  |     5.06341e-05 |                           1 |            0.999996 |
| BLE       | icl_predictor.tf_icl.blocks.1.linear2.bias        | block_01 | ffn.linear2   |     512 | 0.00251394 |      3.09574e-05 |     5.03054e-05 |                           1 |            0.999997 |
| BLE       | icl_predictor.tf_icl.blocks.1.attn.out_proj.bias  | block_01 | attn.out_proj |     512 | 0.00251222 |      3.03032e-05 |     5.02318e-05 |                           1 |            0.999997 |
| BLE       | icl_predictor.tf_icl.blocks.2.attn.out_proj.bias  | block_02 | attn.out_proj |     512 | 0.00249129 |      3.09938e-05 |     5.03456e-05 |                           1 |            0.999997 |
| BLE       | icl_predictor.tf_icl.blocks.11.attn.out_proj.bias | block_11 | attn.out_proj |     512 | 0.00242249 |      3.19917e-05 |     5.05352e-05 |                           1 |            0.999997 |
| BLE       | icl_predictor.tf_icl.blocks.0.linear2.bias        | block_00 | ffn.linear2   |     512 | 0.0022786  |      2.98741e-05 |     5.07043e-05 |                           1 |            0.999997 |
| BLE       | icl_predictor.tf_icl.blocks.4.linear2.bias        | block_04 | ffn.linear2   |     512 | 0.00225578 |      3.08234e-05 |     5.06602e-05 |                           1 |            0.999997 |
| BLE       | icl_predictor.tf_icl.blocks.3.linear2.bias        | block_03 | ffn.linear2   |     512 | 0.00223672 |      3.02921e-05 |     5.03259e-05 |                           1 |            0.999997 |
| BLE       | icl_predictor.ln.bias                             | ln       | ln            |     512 | 0.00220323 |      3.34627e-05 |     5.06251e-05 |                           1 |            0.999998 |
| BNG       | icl_predictor.tf_icl.blocks.2.linear2.bias        | block_02 | ffn.linear2   |     512 | 0.00349402 |      3.44451e-05 |     5.06155e-05 |                           1 |            0.999994 |
| BNG       | icl_predictor.tf_icl.blocks.1.attn.out_proj.bias  | block_01 | attn.out_proj |     512 | 0.00273867 |      3.33386e-05 |     5.04553e-05 |                           1 |            0.999996 |
| BNG       | icl_predictor.tf_icl.blocks.1.linear2.bias        | block_01 | ffn.linear2   |     512 | 0.00271079 |      3.32648e-05 |     5.06612e-05 |                           1 |            0.999996 |
| BNG       | icl_predictor.tf_icl.blocks.2.attn.out_proj.bias  | block_02 | attn.out_proj |     512 | 0.00267397 |      3.35361e-05 |     5.07063e-05 |                           1 |            0.999996 |
| BNG       | icl_predictor.tf_icl.blocks.0.linear2.bias        | block_00 | ffn.linear2   |     512 | 0.00250812 |      3.37754e-05 |     5.0582e-05  |                           1 |            0.999997 |
| BNG       | icl_predictor.tf_icl.blocks.3.linear2.bias        | block_03 | ffn.linear2   |     512 | 0.00246538 |      3.33127e-05 |     5.07175e-05 |                           1 |            0.999997 |
| BNG       | icl_predictor.tf_icl.blocks.4.linear2.bias        | block_04 | ffn.linear2   |     512 | 0.00240079 |      3.26937e-05 |     5.06979e-05 |                           1 |            0.999997 |
| BNG       | icl_predictor.tf_icl.blocks.0.attn.out_proj.bias  | block_00 | attn.out_proj |     512 | 0.00237131 |      3.48257e-05 |     5.07589e-05 |                           1 |            0.999997 |
| BNG       | icl_predictor.tf_icl.blocks.11.linear2.bias       | block_11 | ffn.linear2   |     512 | 0.00221795 |      2.52711e-05 |     4.96926e-05 |                           1 |            0.999998 |
| BNG       | icl_predictor.tf_icl.blocks.3.attn.out_proj.bias  | block_03 | attn.out_proj |     512 | 0.00214852 |      3.22191e-05 |     5.06975e-05 |                           1 |            0.999998 |

## Largest layer deltas

| dataset   | layer    |   num_tensors |   numel |     rel_l2 |   mean_abs_delta |   max_abs_delta |
|:----------|:---------|--------------:|--------:|-----------:|-----------------:|----------------:|
| Amazon    | block_11 |            12 | 2102784 | 0.00110747 |      2.8423e-05  |     5.0721e-05  |
| Amazon    | block_00 |            12 | 2102784 | 0.00109652 |      2.71906e-05 |     5.08092e-05 |
| Amazon    | block_05 |            12 | 2102784 | 0.00104625 |      2.75105e-05 |     5.07892e-05 |
| Amazon    | block_02 |            12 | 2102784 | 0.00104335 |      2.67877e-05 |     5.0772e-05  |
| Amazon    | block_04 |            12 | 2102784 | 0.0010268  |      2.68813e-05 |     5.07806e-05 |
| BLE       | block_11 |            12 | 2102784 | 0.00135116 |      3.49062e-05 |     5.07988e-05 |
| BLE       | block_00 |            12 | 2102784 | 0.00131907 |      3.38056e-05 |     5.08074e-05 |
| BLE       | block_02 |            12 | 2102784 | 0.00124689 |      3.34732e-05 |     5.08064e-05 |
| BLE       | block_03 |            12 | 2102784 | 0.00122714 |      3.25764e-05 |     5.07999e-05 |
| BLE       | block_04 |            12 | 2102784 | 0.00122256 |      3.33981e-05 |     5.08092e-05 |
| BNG       | block_00 |            12 | 2102784 | 0.00143937 |      3.79992e-05 |     5.07897e-05 |
| BNG       | block_02 |            12 | 2102784 | 0.00133598 |      3.62319e-05 |     5.07925e-05 |
| BNG       | block_03 |            12 | 2102784 | 0.00130978 |      3.52743e-05 |     5.07869e-05 |
| BNG       | block_04 |            12 | 2102784 | 0.0013017  |      3.59547e-05 |     5.07981e-05 |
| BNG       | block_05 |            12 | 2102784 | 0.00130165 |      3.59997e-05 |     5.07981e-05 |
