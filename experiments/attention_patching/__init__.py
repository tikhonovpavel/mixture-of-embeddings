"""Attention-map patching experiment.

Measures robustness of pretrained decoder LLMs to wholesale replacement of
post-softmax attention weights with weights from an unrelated donor text.

Serves as a control for the Superposition Linearity result in the main paper
(icml_submission.tex): if the model is robust to fully replacing its attention
maps, then "mixed embeddings do not break attention" cannot be the mechanism
behind superposition.
"""
