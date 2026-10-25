"""Small regression tests for analytics utilities added after Part 1."""
import pytest
from vyndra.nutrition import normalize_food_name, find_food, calculate_nutrition
from vyndra.portions import portion_grams, serving_info
from vyndra.matching import match_class_to_food, load_approved_mapping
from vyndra.recommend import _tags

@pytest.mark.parametrize(("value", "expected"), [
    ("Hot tea (Garam Chai)", "hot tea"),
    ("  IDLI & SAMBAR ", "idli and sambar"),
    ("Paneer-Tikka", "paneer tikka"),
    ("", ""),
    (None, ""),
])
def test_food_name_normalization(value, expected):
    assert normalize_food_name(value) == expected

@pytest.mark.parametrize(("value", "expected"), [
    ("Vegan, Vegetarian", ["Vegan", "Vegetarian"]),
    ("Eggs|Nuts", ["Eggs", "Nuts"]),
    ("", []),
    (None, []),
    ("none", []),
])
def test_tag_splitting(value, expected):
    assert _tags(value) == expected

def test_unknown_food_uses_documented_default_portion():
    result = serving_info("food definitely not in workbook")
    assert result["grams_per_serving"] == 100.0
    assert result["default_portion"] is True

def test_explicit_grams_override_serving_estimate():
    result = portion_grams("idli", grams=175)
    assert result["quantity_grams"] == 175.0
    assert result["portion_source"] == "user_grams"

def test_servings_multiplier_is_applied():
    one = portion_grams("food definitely not in workbook", servings=1)
    two = portion_grams("food definitely not in workbook", servings=2)
    assert one["quantity_grams"] == 100.0
    assert two["quantity_grams"] == 200.0

def test_portion_defaults_when_no_input():
    result = portion_grams("food definitely not in workbook")
    assert result["quantity_grams"] == 100.0
    assert result["default_portion"] is True

@pytest.mark.parametrize("grams", [0, -1, 5001, True])
def test_invalid_explicit_grams_rejected(grams):
    with pytest.raises((ValueError, TypeError)):
        portion_grams("idli", grams=grams)

@pytest.mark.parametrize("servings", [0, -1, 21, True])
def test_invalid_servings_rejected(servings):
    with pytest.raises((ValueError, TypeError)):
        portion_grams("idli", servings=servings)

def test_known_nutrition_lookup_has_values():
    result = find_food("Hot tea (Garam Chai)")
    assert result["found"] is True
    assert result["energy_kcal"] >= 0

def test_unknown_nutrition_lookup_never_guesses():
    result = find_food("VYNDRA unknown fictional dish")
    assert result["found"] is False
    assert "energy_kcal" not in result

def test_nutrition_scales_with_grams():
    one = calculate_nutrition("Hot tea (Garam Chai)", 100)
    two = calculate_nutrition("Hot tea (Garam Chai)", 200)
    assert two["energy_kcal"] == pytest.approx(one["energy_kcal"] * 2)

def test_approved_mapping_is_read_without_modification():
    mapping = load_approved_mapping()
    assert mapping
    assert "allo gobi" in mapping
    assert mapping["allo gobi"]["match_type"] in {"exact", "family"}

def test_unmatched_class_is_not_given_nutrition():
    result = match_class_to_food("VYNDRA fictional dish class 9000")
    assert result.get("found") is False
