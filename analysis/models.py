"""The trained spam models, by short name: (base model, adapter). Shared by the speed/budget analyses."""
S135, S360 = "mlx-community/SmolLM2-135M-Instruct", "mlx-community/SmolLM2-360M-Instruct"
P70, P14 = "EleutherAI/pythia-70m", "EleutherAI/pythia-14m"
MODELS = {
    "360m": (S360, "checkpoints/spamshape4_smollm2-360m"),
    "135m": (S135, "checkpoints/spamshape4_smollm2-135m"),
    "135m_distill_raw": (S135, "checkpoints/distill_t360_raw_smollm2-135m"),
    "135m_distill_pipeline": (S135, "checkpoints/distill_t360_pipeline_smollm2-135m"),
    "135m_drop10": ("checkpoints/spamshape4_135m_drop10", "checkpoints/heal_spamshape4_135m_drop10"),
    "pythia70m_distill_pipeline": (P70, "checkpoints/distill_t360_pipeline_pythia-70m"),
    "pythia14m": (P14, "checkpoints/spamshape4_pythia-14m"),
    "pythia14m_distill_raw": (P14, "checkpoints/distill_t360_raw_pythia-14m"),
    "pythia14m_distill_pipeline": (P14, "checkpoints/distill_t360_pipeline_pythia-14m"),
    # vocab-pruned (train/prune_vocab.py): plain model folders, no adapter
    "p14_vocab_all": ("checkpoints/vocab_all_pythia-14m", None),
    "p14_vocab8k_cut": ("checkpoints/vocab8k_pythia-14m", None),
    "p14_vocab8k": ("checkpoints/vocab8k_pythia-14m", "checkpoints/heal_vocab8k_pythia-14m"),
    "p14_vocab4k_cut": ("checkpoints/vocab4k_pythia-14m", None),
    "p14_vocab4k": ("checkpoints/vocab4k_pythia-14m", "checkpoints/heal_vocab4k_pythia-14m"),
    "p14_vocab2k": ("checkpoints/vocab2k_pythia-14m", "checkpoints/heal_vocab2k_pythia-14m"),
    "p14_vocab1k": ("checkpoints/vocab1k_pythia-14m", "checkpoints/heal_vocab1k_pythia-14m"),
    "p14_vocab512": ("checkpoints/vocab512_pythia-14m", "checkpoints/heal_vocab512_pythia-14m"),
}
