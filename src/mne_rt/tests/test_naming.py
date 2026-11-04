"""Tests for instanced modality names (``base@label``)."""

from pathlib import Path

import pytest
import yaml

import mne_rt
from mne_rt._naming import (
    MODALITY_SEP,
    ModalitySpec,
    osc_address_name,
    parse_modality,
    resolve_by_base,
    split_modality,
)

CONFIG = Path(mne_rt.__file__).parent / "config_methods.yml"
KNOWN_MODALITIES = sorted(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["NF_modality"])


# ------------------------------------------------------------------
# split_modality / parse_modality
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, base, label",
    [
        ("sensor_power", "sensor_power", ""),
        ("sensor_power@alpha", "sensor_power", "alpha"),
        ("source_connectivity@theta", "source_connectivity", "theta"),
        ("sensor_power@band-1", "sensor_power", "band-1"),
        ("sensor_power@a_b", "sensor_power", "a_b"),
    ],
)
def test_parse_round_trip(name, base, label):
    spec = parse_modality(name)
    assert spec == ModalitySpec(name=name, base=base, label=label)
    assert spec.is_instanced is bool(label)
    assert split_modality(name) == (base, label)


def test_split_modality_never_raises_on_junk():
    """The plot path calls this every frame; a raise there kills the display."""
    assert split_modality("@@@") == ("", "@@")
    assert split_modality("") == ("", "")


@pytest.mark.parametrize("name", KNOWN_MODALITIES)
def test_plain_modality_names_are_unchanged(name):
    """Every shipped modality name must survive parsing untouched."""
    spec = parse_modality(name)
    assert (spec.base, spec.label, spec.is_instanced) == (name, "", False)


@pytest.mark.parametrize(
    "name, match",
    [
        ("a@b@c", "more than one"),
        ("@alpha", "empty base"),
        ("sensor_power@", "empty instance label"),
        ("sensor_power@has space", "may only contain"),
        ("sensor_power@dots.not.allowed", "may only contain"),
        ("sensor_power@slash/no", "may only contain"),
    ],
)
def test_parse_rejects(name, match):
    with pytest.raises(ValueError, match=match):
        parse_modality(name)


def test_parse_rejects_non_string():
    with pytest.raises(ValueError, match="must be strings"):
        parse_modality(None)


# ------------------------------------------------------------------
# resolve_by_base
# ------------------------------------------------------------------


def test_resolve_by_base_prefers_exact_then_base_then_default():
    mapping = {"sensor_power": 1.0, "sensor_power@alpha": 2.0}
    assert resolve_by_base(mapping, "sensor_power@alpha") == 2.0  # exact wins
    assert resolve_by_base(mapping, "sensor_power@theta") == 1.0  # inherits base
    assert resolve_by_base(mapping, "hjorth@x", 9.9) == 9.9  # neither present


def test_resolve_by_base_returns_falsy_entries():
    """A deliberate 0.0 must not fall through to the default."""
    assert resolve_by_base({"sensor_power": 0.0}, "sensor_power@a", 1.0) == 0.0


# ------------------------------------------------------------------
# osc_address_name
# ------------------------------------------------------------------


@pytest.mark.parametrize("name", KNOWN_MODALITIES + ["combined", "snr_db"])
def test_osc_address_name_is_identity_for_plain_names(name):
    """Existing OSC subscriptions must keep matching."""
    assert osc_address_name(name) == name


def test_osc_address_name_flattens_the_separator():
    # Deliberately not "/" — an OSC wildcard does not match across a path
    # separator, so "/ant/*" would stop matching instanced modalities.
    assert osc_address_name("source_connectivity@theta") == "source_connectivity_theta"
    assert "/" not in osc_address_name(f"sensor_power{MODALITY_SEP}alpha")


@pytest.mark.parametrize("char", ["#", "*", ",", "?", "[", "]", "{", "}", " "])
def test_osc_address_name_strips_reserved_characters(char):
    """OSC pattern metacharacters must never reach an address."""
    assert char not in osc_address_name(f"sensor_power{char}x")
