"""Shared legality checks and a guaranteed-legal fallback action."""

from __future__ import annotations

from typing import Any

import numpy as np
from jackdaw.env import FactoredAction

from jackhammer.selfplay.recorder import ACTION_NAMES


def is_action_legal(factored: Any, mask: Any) -> tuple[bool, str]:
    """Return ``(legal, reason)`` for a factored action and current mask."""
    action_type = int(factored.action_type)
    if not mask.type_mask[action_type]:
        return (
            False,
            f"Action type {action_type} ({ACTION_NAMES.get(action_type)}) is not legal right now.",
        )
    if action_type in (0, 1):  # play / discard MUST carry a card selection
        n = len(factored.card_target) if factored.card_target else 0
        need = max(1, int(mask.min_card_select))
        if n < need:
            legal_cards = np.nonzero(mask.card_mask)[0].tolist()
            return False, (
                f"{ACTION_NAMES.get(action_type)} requires the 'cards' parameter with at least "
                f"{need} card index/indices, but received {n}. Legal card indices: {legal_cards}."
            )
    if factored.card_target is not None:
        for idx in factored.card_target:
            if idx < 0 or idx >= len(mask.card_mask) or not mask.card_mask[idx]:
                return False, f"Card index {idx} is not legal to select."
        if not (mask.min_card_select <= len(factored.card_target) <= mask.max_card_select):
            return False, (
                f"Selected {len(factored.card_target)} cards, but min is "
                f"{mask.min_card_select} and max is {mask.max_card_select}."
            )
    if factored.entity_target is not None:
        if action_type not in mask.entity_masks:
            return False, f"Action type {action_type} does not accept an entity target."
        entity_mask = mask.entity_masks[action_type]
        idx = factored.entity_target
        if idx < 0 or idx >= len(entity_mask) or not entity_mask[idx]:
            action_name = ACTION_NAMES.get(action_type)
            return False, (
                f"Entity target {idx} is not legal for action {action_type} ({action_name})."
            )
    return True, ""


def get_fallback_action(mask: Any) -> FactoredAction:
    """Pick the first constructable legal action."""
    legal_types = np.nonzero(mask.type_mask)[0]
    if len(legal_types) == 0:
        raise RuntimeError("No legal actions available in mask!")
    for action_type in legal_types:
        action_type = int(action_type)
        card_target = None
        entity_target = None
        if action_type in (0, 1):
            legal_cards = np.nonzero(mask.card_mask)[0]
            card_target = (int(legal_cards[0]),) if len(legal_cards) > 0 else ()
        elif action_type in mask.entity_masks:
            legal_entities = np.nonzero(mask.entity_masks[action_type])[0]
            if len(legal_entities) > 0:
                entity_target = int(legal_entities[0])
        return FactoredAction(
            action_type=action_type,
            card_target=card_target,
            entity_target=entity_target,
        )
    raise RuntimeError("No constructable legal action.")  # pragma: no cover
