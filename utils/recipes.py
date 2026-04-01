from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


RECIPES_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "generation_recipes.json"


@dataclass(frozen=True)
class SettingRecipe:
    name: str
    generation_strategy: str
    prompt_template: str | None
    prompt_mode: str
    num_human: int
    num_model: int
    num_choices: int
    generated_count: int
    is_schedulable: bool
    conditioned_on: str | None = None
    prerequisite_setting: str | None = None


@lru_cache(maxsize=1)
def load_setting_recipes() -> tuple[SettingRecipe, ...]:
    payload = json.loads(RECIPES_CONFIG_PATH.read_text(encoding="utf-8"))
    return tuple(
        SettingRecipe(
            name=str(raw["name"]),
            generation_strategy=str(raw.get("generation_strategy") or raw["name"]),
            prompt_template=raw.get("prompt_template"),
            prompt_mode=str(raw.get("prompt_mode") or "qa"),
            num_human=int(raw.get("num_human", 0) or 0),
            num_model=int(raw.get("num_model", 0) or 0),
            num_choices=int(raw.get("num_choices", 0) or 0),
            generated_count=int(raw.get("generated_count", 0) or 0),
            is_schedulable=bool(raw.get("is_schedulable", False)),
            conditioned_on=raw.get("conditioned_on"),
            prerequisite_setting=raw.get("prerequisite_setting"),
        )
        for raw in payload.get("settings", [])
    )


@lru_cache(maxsize=1)
def _setting_recipe_map() -> dict[str, SettingRecipe]:
    return {recipe.name: recipe for recipe in load_setting_recipes()}


def get_setting_recipe(name: str) -> SettingRecipe:
    try:
        return _setting_recipe_map()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown setting recipe: {name}") from exc


def schedulable_generation_strategies() -> tuple[str, ...]:
    return tuple(recipe.generation_strategy for recipe in load_setting_recipes() if recipe.is_schedulable)


def setting_specs() -> dict[str, dict[str, int]]:
    return {
        recipe.name: {
            "num_human": recipe.num_human,
            "num_model": recipe.num_model,
            "num_choices": recipe.num_choices,
        }
        for recipe in load_setting_recipes()
    }
