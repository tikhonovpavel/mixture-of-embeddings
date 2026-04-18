# Attention-Patching Experiment: Report

Setup: for each FineWeb text $A$ we replace every post-softmax attention map at every layer with the corresponding map from an unrelated donor text $C$. We keep $Q, K, V$ projections of $A$ intact; only the mixing weights change. No finetuning, no prompt manipulation.

If the model is largely robust to this substitution, then 'mixed embeddings don't break attention' cannot be what makes superposition work in the main paper -- because attention was wholesale replaced with unrelated content here, and the model still tracked the right tokens.

## EleutherAI/pythia-410m

- rank(top-1 of A1) under A2: P(rank=1)=0.194, P(rank<=5)=0.423, P(rank<=10)=0.513, P(rank<=100)=0.719, median rank=9.0.

- Superposition Approximation Ratio (patched vs. unrelated-text baseline): R_KL=0.373, R_JS=0.711, R_Wasserstein=0.770.

- Entropy sanity: mean H(P_A1)=3.213 nats, mean H(P_A2)=6.123 nats (delta=+2.910; should not blow up if the patched distribution stays informative).

## Summary table

| model                  | condition                  |     KL |     JS |   Wasserstein |   R_KL |   R_JS |   R_W |   n_rows |
|:-----------------------|:---------------------------|-------:|-------:|--------------:|-------:|-------:|------:|---------:|
| EleutherAI/pythia-410m | A2 (patched)               | 3.5352 | 0.45   |        0.6409 | 0.3734 | 0.7114 |  0.77 |     6350 |
| EleutherAI/pythia-410m | A1 vs A' (random baseline) | 9.4663 | 0.6326 |        0.8323 | 1      | 1      |  1    |     6350 |


## Interpretation

- If rank-CDF(A2) sits far above chance and R_* << 1, the output distribution remains close to the baseline even with fully unrelated attention maps. This empirically contradicts the folk explanation that superposition works because mixed embeddings induce 'almost-correct' attention; since here attention is explicitly wrong, yet predictions survive, the mechanism must live elsewhere (residual stream + MLP, plausibly also OV-circuits that don't care which particular query-key interaction fired).

- If R_* approaches 1 and the rank CDF hugs the diagonal, the model relies heavily on content-specific attention and the superposition phenomenon cannot be explained without invoking it.
