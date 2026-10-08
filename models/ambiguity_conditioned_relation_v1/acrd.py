"""Small shared-memory relation carrier for the E3--E5 profiling gate.

The pinned DaniFormer materialises a full visual token sequence once per
relation and then applies quadratic self-attention.  This carrier keeps the
backbone and patch memory image-shared and performs only K-token cross
attention per pair.  The three retrieval modes are deliberately simple:

``uniform``  : image-shared static top-K tokens (E4 control)
``mask``     : top-K tokens under the subject/object mask union (E5 control)
``predicate``: top-K tokens selected independently for each coarse predicate
               hypothesis (ACRD seed)

This is a profiling carrier, not a final paper model.  It preserves the
Fair-PSG trainer's (subject logits, object logits, relation logits) contract.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fair_psgg.models.building_blocks import CoordEncoder
from fair_psgg.models.daniformer import SbjObjBoxEncoder, SbjObjMaskEncoder
from fair_psgg.models.daniformer import make_sine_position_encoding
from fair_psgg.models.frequency_bias import FreqBias


class AmbiguityConditionedRelationDecoder(nn.Module):
    """Shared visual memory + pair query + sparse evidence decoder."""

    def __init__(
        self,
        num_node_outputs: int,
        num_rel_outputs: int,
        extractor: nn.Module,
        transformer_depth: int = 1,
        embed_dim: int = 384,
        patch_size: int = 8,
        feature_shape=(256, 128, 128),
        use_semantics: bool = True,
        use_masks: bool = True,
        bg_ratio_strategy: str = "sum",
        encode_coords: bool = True,
        retrieval_mode: str = "predicate",
        top_k: int = 16,
        num_candidates: int = 3,
    ):
        super().__init__()
        if retrieval_mode not in {"uniform", "mask", "predicate"}:
            raise ValueError(f"unknown retrieval mode: {retrieval_mode}")
        self.extractor = extractor
        self.embed_dim = int(embed_dim)
        self.patch_size = (patch_size, patch_size)
        self.feature_shape = feature_shape
        self.retrieval_mode = retrieval_mode
        self.top_k = int(top_k)
        self.num_candidates = int(num_candidates)
        self.use_masks = bool(use_masks)

        self.patch_embed = nn.Conv2d(
            feature_shape[0], self.embed_dim, kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        tokens = (feature_shape[-2] // patch_size) * (feature_shape[-1] // patch_size)
        self.register_buffer(
            "pos_encoding",
            make_sine_position_encoding(
                feature_shape[-2:], patch_size=self.patch_size, d_model=self.embed_dim
            ),
            persistent=False,
        )

        if use_masks:
            self.sbjobj_encoder = SbjObjMaskEncoder(
                self.embed_dim, patch_size, bg_ratio_strategy, bg_token=True
            )
        else:
            self.sbjobj_encoder = SbjObjBoxEncoder(
                self.embed_dim, patch_size, feature_shape[1:],
                bg_ratio_strategy, bg_token=True,
            )
        self.classification_token = nn.Parameter(torch.rand(self.embed_dim))
        self.freq_bias = (
            FreqBias(num_node_classes=num_node_outputs, num_rel_outputs=self.embed_dim)
            if use_semantics else None
        )
        self.coord_embed = CoordEncoder(self.embed_dim) if encode_coords else None

        self.coarse_rel = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, num_rel_outputs),
        )
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.memory_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.embed_dim, num_heads=8, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
        )
        self.node_head = nn.Linear(self.embed_dim, num_node_outputs * 2)
        self.relation_head = nn.Linear(self.embed_dim, num_rel_outputs)
        self.candidate_embeddings = nn.Parameter(torch.randn(num_rel_outputs, self.embed_dim) * 0.02)
        self.token_score = nn.Linear(self.embed_dim, 1)
        self.predicate_key = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Keep this visible in checkpoints and logs for the resource gate.
        self.num_memory_tokens = tokens

    def _pair_context(self, data, features, sbj_ids, obj_ids, img_shape):
        if self.use_masks:
            pair_tokens = self.sbjobj_encoder(
                data["segmentation"][sbj_ids], data["segmentation"][obj_ids],
                features.shape[-2:],
            )
            # The full spatial mask encoding is reduced to a single pair query;
            # it is never fed through pair-wise self-attention.
            pair_context = pair_tokens.mean(dim=1)
        else:
            h, w = img_shape[-2:]
            fh, fw = features.shape[-2:]
            coords = data["bboxes"].clone()
            coords[:, [0, 2]] *= fw / w
            coords[:, [1, 3]] *= fh / h
            pair_context = self.sbjobj_encoder(
                coords[sbj_ids], coords[obj_ids]
            ).mean(dim=1)

        extra = self.classification_token[None].expand(pair_context.size(0), -1)
        extra = extra + pair_context
        box_targets = data["box_categories"]
        if self.freq_bias is not None:
            extra = extra + self.freq_bias(
                box_targets[sbj_ids], box_targets[obj_ids]
            )
        if self.coord_embed is not None:
            extra = extra + self.coord_embed(
                data["bboxes"][sbj_ids].to(features.device),
                data["bboxes"][obj_ids].to(features.device), img_shape[-2:],
            )
        return extra

    def _gather(self, memory, indices):
        # memory: [R, T, D], indices: [R, K]
        return memory.gather(1, indices[..., None].expand(-1, -1, memory.size(-1)))

    def _attend(self, query, memory):
        q = self.query_norm(query)[:, None]
        k = self.memory_norm(memory)
        attended, _ = self.cross_attention(q, k, k, need_weights=False)
        attended = attended[:, 0] + query
        return attended + self.ffn(attended)

    def _forward_internal(self, data, features, memory, img_shape, sbj_ids, obj_ids):
        num_boxes = data["num_boxes"]
        img_ids = torch.repeat_interleave(num_boxes.to(sbj_ids.device))[sbj_ids]
        pair_query = self._pair_context(data, features, sbj_ids, obj_ids, img_shape)
        pair_memory = memory[img_ids]
        coarse = self.coarse_rel(pair_query)

        if self.retrieval_mode == "mask" and self.use_masks:
            sbj = data["segmentation"][sbj_ids].to(features.device)
            obj = data["segmentation"][obj_ids].to(features.device)
            _, sr, or_ = self.sbjobj_encoder.forward_ratios(
                sbj, obj, features.shape[-2:]
            )
            scores = sr + or_
            k = min(self.top_k, scores.size(1))
            idx = scores.topk(k, dim=1).indices
            attended = self._attend(pair_query, self._gather(pair_memory, idx))
            rel = coarse + self.relation_head(attended)
        elif self.retrieval_mode == "predicate":
            candidate_ids = coarse.sigmoid().topk(
                min(self.num_candidates, coarse.size(1)), dim=1
            ).indices
            queries = pair_query[:, None, :] + self.candidate_embeddings[candidate_ids]
            queries = queries.reshape(-1, self.embed_dim)
            candidate_keys = self.predicate_key(self.candidate_embeddings[candidate_ids])
            scores = torch.einsum("rtd,rmd->rmt", pair_memory, candidate_keys)
            k = min(self.top_k, scores.size(-1))
            idx = scores.topk(k, dim=-1).indices
            mem = pair_memory[:, None].expand(-1, candidate_ids.size(1), -1, -1)
            mem = mem.reshape(-1, mem.size(-2), mem.size(-1))
            mem = self._gather(mem, idx.reshape(-1, k))
            attended = self._attend(queries, mem).reshape(
                pair_query.size(0), candidate_ids.size(1), self.embed_dim
            )
            deltas = self.relation_head(attended).gather(
                2, candidate_ids[..., None]
            ).squeeze(-1)
            rel = coarse.scatter_add(1, candidate_ids, deltas)
        else:
            static_scores = self.token_score(pair_memory).squeeze(-1)
            k = min(self.top_k, static_scores.size(1))
            idx = static_scores.topk(k, dim=1).indices
            attended = self._attend(pair_query, self._gather(pair_memory, idx))
            rel = coarse + self.relation_head(attended)

        node = self.node_head(attended if self.retrieval_mode != "predicate" else attended.mean(1))
        half = node.size(-1) // 2
        return node[:, half:], node[:, :half], rel

    def forward(self, data: dict, return_attention=None, max_relations=None):
        del return_attention
        pair_ids = data["pair_ids"]
        sbj_ids, obj_ids = pair_ids[:, 0], pair_ids[:, 1]
        features = self.extractor(data["img"])
        assert features.shape[1:] == self.feature_shape, (features.shape, self.feature_shape)
        memory = self.patch_embed(features).flatten(2).transpose(1, 2)
        memory = memory + self.pos_encoding.to(memory.device)

        if max_relations is None:
            max_relations = len(sbj_ids)
        outputs = [[], [], []]
        for sb, ob in zip(sbj_ids.split(max_relations), obj_ids.split(max_relations)):
            so, oo, rr = self._forward_internal(
                data, features, memory, data["img"].shape, sb, ob
            )
            outputs[0].append(so)
            outputs[1].append(oo)
            outputs[2].append(rr)
        return tuple(torch.cat(x, dim=0) for x in outputs)
