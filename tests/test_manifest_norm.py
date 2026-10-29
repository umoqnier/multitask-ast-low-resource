"""Self-check for manifest normalization in scripts/train_nemo_multitask.py.

Run directly: python tests/test_manifest_norm.py
"""

import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The training script only needs click for its CLI; stub it so this check runs
# without the training environment installed.
if "click" not in sys.modules:
    _click = types.ModuleType("click")
    _click.command = lambda *a, **k: lambda f: f
    _click.option = lambda *a, **k: lambda f: f
    _click.Choice = lambda choices: choices
    sys.modules["click"] = _click

from scripts.train_nemo_multitask import langs_for_task, normalize_manifests

assert langs_for_task("asr", "nah") == ["nah"]
assert langs_for_task("ast", "nah") == ["es"]
assert langs_for_task("multitask", "nah") == ["nah", "es"]

with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "manifests"
    audio = Path(tmp) / "a.wav"
    audio.write_bytes(b"RIFF")
    # Mimic what build_datasets.py writes for Nahuatl (placeholder langs, task set).
    rows = {
        "train_manifest.json": [
            {
                "text": "tlahtoz",
                "duration": 1.0,
                "task": "asr",
                "source_lang": "es",
                "target_lang": "es",
                "audio_filepath": str(audio),
            },
            {
                "text": "hola",
                "duration": 1.0,
                "task": "ast",
                "source_lang": "en",
                "target_lang": "es",
                "audio_filepath": str(audio),
            },
        ]
    }
    for split in ("train", "validation", "test"):
        name = (
            f"{split}_manifest.json" if split != "validation" else "val_manifest.json"
        )
        path = src / "azz" / "multitask" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows["train_manifest.json"]))

    out_base, counts, durations = normalize_manifests(
        src, "azz", "multitask", Path(tmp) / "norm"
    )
    assert counts == {"train": 2, "validation": 2, "test": 2}
    assert durations["train"] == 2.0

    entries = [
        json.loads(line)
        for line in (Path(out_base) / "train_manifest.json")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    asr, ast = (
        [e for e in entries if e["task"] == "asr"][0],
        [e for e in entries if e["task"] == "ast"][0],
    )
    assert (asr["source_lang"], asr["target_lang"]) == ("azz", "azz")
    assert (ast["source_lang"], ast["target_lang"]) == ("azz", "es")
    assert asr["pnc"] == "no"

    # Originals are never modified.
    first_orig = json.loads(
        (src / "azz" / "multitask" / "train_manifest.json").read_text().splitlines()[0]
    )
    assert first_orig["source_lang"] == "es"

    # Missing audio must fail fast, not starve training.
    bad = src / "azz" / "asr"
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "train_manifest.json").write_text(
        json.dumps(
            {"audio_filepath": "/nonexistent.wav", "duration": 1.0, "task": "asr"}
        )
    )
    for name in ("val_manifest.json", "test_manifest.json"):
        (bad / name).write_text("")
    try:
        normalize_manifests(src, "azz", "asr", Path(tmp) / "norm_bad")
        raise AssertionError("expected SystemExit for missing audio")
    except SystemExit:
        pass

print("OK")
