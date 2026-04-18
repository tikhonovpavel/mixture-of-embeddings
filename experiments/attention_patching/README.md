# Attention-Patching Experiment

Control experiment for the Superposition Linearity result in
`icml_submission.tex`. We test how robust a pretrained decoder-only LLM is
when every post-softmax attention map (at every layer, every head) is
replaced in-place with the attention map computed on an unrelated donor text.

If the model is robust to this wholesale attention swap, then the common
explanation "superposition works because mixed embeddings don't break
attention" cannot be the real mechanism — attention can be fully wrong and
predictions still survive.

## Files

- `patching.py` — donor-attention capture (via `output_attentions=True`) and
  patched forward pass, implemented by registering a custom entry in
  `transformers.ALL_ATTENTION_FUNCTIONS` and flipping
  `config._attn_implementation` only for the patched forward. Works for
  Pythia / Llama / Qwen / any decoder-only model in transformers v5 that
  exposes `module.layer_idx` on attention submodules (i.e. all modern
  decoder-only HF models).
- `metrics_extra.py` — top-256 Wasserstein with cosine distance between
  token embeddings as the ground metric (POT backend). KL / JS / TVD /
  entropy are reused from the repo-level `metrics.py`.
- `run_patching.py` — end-to-end runner. Loads FineWeb via the
  streaming helper already used by `run_batch_tvd_temperature.py`,
  tokenizes to a fixed length T, and for each pair computes A1 (baseline
  on A), A2 (patched on A with donor C), and A' (baseline on an unrelated
  text — denominator of the Superposition Approximation Ratio).
- `analyze.py` — builds the rank-CDF plot, entropy histogram, summary
  table (tab:dist_metrics style), and a markdown report.

## Quick debug run (Pythia-410M, N=50)

```bash
cd $REPO_ROOT
python -m experiments.attention_patching.run_patching \
    --model_name EleutherAI/pythia-410m \
    --num_pairs 50 \
    --max_length 128
```

## Full run (Pythia-2.8B, N=1000)

```bash
python -m experiments.attention_patching.run_patching \
    --model_name EleutherAI/pythia-2.8b \
    --num_pairs 1000 \
    --max_length 128
```

## Aggregate / plots / report

```bash
python -m experiments.attention_patching.analyze \
    --inputs experiments/attention_patching/results/*.parquet
```

Outputs (default `experiments/attention_patching/results/analysis/`):

- `rank_cdf.png` — rank CDF in the style of `fig:rank_curve`.
- `entropy_hist.png` — entropy histogram of `P_{A1}` vs `P_{A2}`.
- `summary.csv`, `summary.md` — tab:dist_metrics-style table.
- `report.md` — human-readable summary.

## Design notes

- **Why `ALL_ATTENTION_FUNCTIONS` and not forward-hooks?** In transformers
  v5 every attention module dispatches through
  `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,
  eager_attention_forward)`. Registering a custom implementation and
  switching the attention implementation is the intended extension point
  and works uniformly across MHA/GQA/MLA, Pythia/Llama/Qwen/Gemma, without
  having to rewrite per-architecture `forward`s or monkey-patch
  `torch.nn.functional.softmax`.
- **Causal mask**: donor attention maps come from a standard forward pass
  on the donor text, so they are already post-softmax and causal
  (`A_{ij}=0` for `j>i`, rows sum to 1). No renormalization needed.
- **GQA (Llama, Qwen, Gemma)**: `output_attentions=True` returns
  attention after `repeat_kv`, i.e. with `num_query_heads` rows. When we
  matmul donor@V in the patched impl, we apply `repeat_kv` to V so the
  head counts match — mirroring `eager_attention_forward` in
  `modeling_llama.py`.
- **Value path unchanged**: the donor controls only mixing weights; the V
  projection is computed from the live hidden states of A exactly as in
  the unmodified forward (which is what the experiment spec asks for:
  "matrices mixture from C, values from A").
- **Position 0 skipped** — there's only one key to attend to there, so
  patching is a no-op.
- **Reproducibility**: `--seed` controls dataset streaming order, text
  pairing (A / C / A'), and numpy/torch RNG.
