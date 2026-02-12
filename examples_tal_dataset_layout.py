"""Builds a minimal fake TaL layout and validates metadata parsing/splits."""

from __future__ import annotations

from pathlib import Path
import tempfile

from utitospeech.data import (
    TAL1_NAME,
    TAL80_NAME,
    discover_tal_utterances,
    parse_spontaneous_lab,
    prompt_disjoint_split,
    speaker_disjoint_split,
)


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_utterance(base: Path, stem: str, with_lab: bool = False) -> None:
    _touch(base / f"{stem}.txt", "dummy prompt")
    _touch(base / f"{stem}.wav")
    _touch(base / f"{stem}.sync")
    _touch(base / f"{stem}.ult")
    _touch(base / f"{stem}.param")
    _touch(base / f"{stem}.mp4")
    if with_lab:
        _touch(base / f"{stem}.lab", "0.00 2.50 hello world\n2.50 5.00 next segment")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        tal80 = root / "TaL80"
        _make_utterance(tal80 / "core" / "01fi", "001_xaud")
        _make_utterance(tal80 / "core" / "01fi", "029_xsil")
        _make_utterance(tal80 / "core" / "03mn", "002_xaud")

        tal1 = root / "TaL1"
        _make_utterance(tal1 / "core" / "day1", "038_sil")
        _make_utterance(tal1 / "core" / "day2", "028_spo", with_lab=True)

        tal80_records = discover_tal_utterances(tal80, TAL80_NAME)
        tal1_records = discover_tal_utterances(tal1, TAL1_NAME)

        train_spk, test_spk = speaker_disjoint_split(tal80_records, train_ratio=0.5, seed=13)
        train_prompt, test_prompt = prompt_disjoint_split(
            tal80_records + tal1_records,
            train_tags={"xaud", "xsil"},
            test_tags={"sil", "spo"},
        )

        spo_rec = [r for r in tal1_records if r.utterance_id.prompt_tag == "spo"][0]
        segments = parse_spontaneous_lab(spo_rec.lab)

        print("TaL80 records:", len(tal80_records))
        print("TaL1 records:", len(tal1_records))
        print("Speaker-disjoint split:", len(train_spk), len(test_spk))
        print("Prompt-disjoint split:", len(train_prompt), len(test_prompt))
        print("SPO segments:", len(segments), segments[0].text)


if __name__ == "__main__":
    main()
