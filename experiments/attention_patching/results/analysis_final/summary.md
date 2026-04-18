| model                          | condition                  |      KL |     JS |   Wasserstein |   R_KL |   R_JS |    R_W |   n_rows |
|:-------------------------------|:---------------------------|--------:|-------:|--------------:|-------:|-------:|-------:|---------:|
| Qwen/Qwen2.5-3B [donor]        | A2 (patched)               |  3.5395 | 0.4685 |        0.6563 | 0.2703 | 0.7041 | 0.729  |    12700 |
| Qwen/Qwen2.5-3B [donor]        | A1 vs A' (random baseline) | 13.0955 | 0.6655 |        0.9003 | 1      | 1      | 1      |    12700 |
| Qwen/Qwen2.5-3B [permutation]  | A2 (patched)               |  9.1278 | 0.6429 |        0.8587 | 0.6831 | 0.9644 | 0.9498 |     7620 |
| Qwen/Qwen2.5-3B [permutation]  | A1 vs A' (random baseline) | 13.3625 | 0.6666 |        0.904  | 1      | 1      | 1      |     7620 |
| EleutherAI/pythia-410m [donor] | A2 (patched)               |  3.5352 | 0.45   |        0.6409 | 0.3734 | 0.7114 | 0.77   |     6350 |
| EleutherAI/pythia-410m [donor] | A1 vs A' (random baseline) |  9.4663 | 0.6326 |        0.8323 | 1      | 1      | 1      |     6350 |