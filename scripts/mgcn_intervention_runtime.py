"""Auditable MGCN forward decomposition with representation-level interventions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MGCNComponents:
    content: torch.Tensor
    image_items: torch.Tensor
    text_items: torch.Tensor


@torch.no_grad()
def decompose(model: torch.nn.Module) -> MGCNComponents:
    """Reproduce MGCN.forward up to the modality fuser."""
    image_features = model.image_trs(model.image_embedding.weight)
    text_features = model.text_trs(model.text_embedding.weight)
    image_items = torch.multiply(
        model.item_id_embedding.weight, model.gate_v(image_features)
    )
    text_items = torch.multiply(
        model.item_id_embedding.weight, model.gate_t(text_features)
    )

    user_items = torch.cat(
        [model.user_embedding.weight, model.item_id_embedding.weight], dim=0
    )
    layers = [user_items]
    for _ in range(model.n_ui_layers):
        user_items = torch.sparse.mm(model.norm_adj, user_items)
        layers.append(user_items)
    content = torch.stack(layers, dim=1).mean(dim=1)

    for _ in range(model.n_layers):
        image_items = torch.sparse.mm(model.image_original_adj, image_items)
        text_items = torch.sparse.mm(model.text_original_adj, text_items)
    return MGCNComponents(content, image_items, text_items)


def intervene_items(
    items: torch.Tensor,
    mode: str,
    *,
    seed: int,
) -> torch.Tensor:
    """Intervene on enriched item representations, preserving tensor shape."""
    if mode == "none":
        return items
    if mode == "zero":
        return torch.zeros_like(items)
    if mode == "mean":
        return items.mean(dim=0, keepdim=True).expand_as(items)
    if mode == "permutation":
        generator = torch.Generator(device=items.device)
        generator.manual_seed(seed)
        order = torch.randperm(
            items.shape[0], generator=generator, device=items.device
        )
        return items[order]
    raise ValueError(f"Unsupported intervention mode: {mode}")


def branch_from_items(model: torch.nn.Module, items: torch.Tensor) -> torch.Tensor:
    """Recompute user modality representations from intervened item branches."""
    users = torch.sparse.mm(model.R, items)
    return torch.cat([users, items], dim=0)


@torch.no_grad()
def fuse(
    model: torch.nn.Module,
    components: MGCNComponents,
    *,
    image_mode: str = "none",
    text_mode: str = "none",
    permutation_seed: int = 20260730,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_items = intervene_items(
        components.image_items, image_mode, seed=permutation_seed
    )
    text_items = intervene_items(
        components.text_items, text_mode, seed=permutation_seed + 1
    )
    image = branch_from_items(model, image_items)
    text = branch_from_items(model, text_items)

    attention = torch.cat(
        [model.query_common(image), model.query_common(text)], dim=-1
    )
    weights = model.softmax(attention)
    common = (
        weights[:, 0].unsqueeze(1) * image
        + weights[:, 1].unsqueeze(1) * text
    )
    separate_image = image - common
    separate_text = text - common
    image_preference = model.gate_image_prefer(components.content)
    text_preference = model.gate_text_prefer(components.content)
    separate_image = torch.multiply(image_preference, separate_image)
    separate_text = torch.multiply(text_preference, separate_text)
    side = (separate_image + separate_text + common) / 3
    combined = components.content + side
    return torch.split(combined, [model.n_users, model.n_items], dim=0)

