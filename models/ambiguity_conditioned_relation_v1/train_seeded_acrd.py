"""Seeded Fair-PSG launcher with a local ACRD model factory override."""

from __future__ import annotations

import os
import random
import runpy

import numpy as np
import torch

from models.ambiguity_conditioned_relation_v1.acrd import (
    AmbiguityConditionedRelationDecoder,
)


def main() -> None:
    seed = int(os.environ.get("ACRD_SEED", "0"))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    import fair_psgg.from_config as factory

    def get_model(config, num_node_outputs, num_rel_outputs):
        extractor = factory.get_extractor(config.extractor)
        with torch.inference_mode():
            sample = extractor(torch.rand(1, 3, 640, 640))
        arch = config.architecture
        return AmbiguityConditionedRelationDecoder(
            num_node_outputs=num_node_outputs,
            num_rel_outputs=num_rel_outputs,
            extractor=extractor,
            transformer_depth=1,
            embed_dim=arch.embed_dim,
            patch_size=arch.patch_size,
            feature_shape=sample.shape[1:],
            use_semantics=arch.use_semantics,
            use_masks=arch.use_masks,
            bg_ratio_strategy=arch.bg_ratio_strategy,
            encode_coords=arch.encode_coords,
            retrieval_mode=os.environ.get("ACRD_MODE", "predicate"),
            top_k=int(os.environ.get("ACRD_TOP_K", "16")),
            num_candidates=int(os.environ.get("ACRD_NUM_CANDIDATES", "3")),
        )

    factory.get_model = get_model
    runpy.run_module("fair_psgg", run_name="__main__")


if __name__ == "__main__":
    main()
