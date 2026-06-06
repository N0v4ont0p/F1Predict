"""Race-reference parser tests."""
from __future__ import annotations

import pandas as pd
import pytest

from f1predict.data_pipeline.racref import parse_racref


@pytest.fixture()
def schedule():
    # Minimal 2026 calendar slice covering the disambiguation cases.
    return pd.DataFrame([
        {"season": 2026, "round": 3, "circuit_id": "suzuka"},
        {"season": 2026, "round": 6, "circuit_id": "monaco"},
        {"season": 2026, "round": 7, "circuit_id": "catalunya"},
        {"season": 2026, "round": 14, "circuit_id": "madring"},
        {"season": 2026, "round": 22, "circuit_id": "yas_marina"},
    ])


def test_two_and_four_digit_years(schedule):
    assert parse_racref("suzuka26", schedule=schedule).season == 2026
    assert parse_racref("suzuka2026", schedule=schedule).season == 2026


def test_country_city_circuit_aliases(schedule):
    assert parse_racref("japan26", schedule=schedule).circuit_id == "suzuka"
    assert parse_racref("abudhabi2026", schedule=schedule).circuit_id == "yas_marina"
    assert parse_racref("uae2026", schedule=schedule).circuit_id == "yas_marina"
    assert parse_racref("monaco26", schedule=schedule).circuit_id == "monaco"


def test_madrid_vs_barcelona_disambiguation(schedule):
    # Madrid is the official 'Spanish GP' from 2026.
    assert parse_racref("madring26", schedule=schedule).circuit_id == "madring"
    assert parse_racref("spain26", schedule=schedule).circuit_id == "madring"
    assert parse_racref("barca26", schedule=schedule).circuit_id == "catalunya"
    assert parse_racref("catalunya26", schedule=schedule).circuit_id == "catalunya"


def test_round_resolution(schedule):
    assert parse_racref("madring26", schedule=schedule).round == 14
    assert parse_racref("suzuka26", schedule=schedule).round == 3


def test_unknown_circuit_raises(schedule):
    with pytest.raises(ValueError):
        parse_racref("notarealplace26", schedule=schedule)


def test_separators_and_case(schedule):
    assert parse_racref("Abu_Dhabi_2026", schedule=schedule).circuit_id == "yas_marina"
    assert parse_racref("ABUDHABI26", schedule=schedule).circuit_id == "yas_marina"
