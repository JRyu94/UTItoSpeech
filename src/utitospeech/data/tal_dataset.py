from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable


TAL1_NAME = "TaL1"
TAL80_NAME = "TaL80"

PROMPT_TAGS = {"swa", "cal", "aud", "sil", "whi", "spo", "xaud", "xsil", "xwhi"}


@dataclass(frozen=True)
class Tal80Speaker:
    raw_id: str
    index: int
    gender: str
    country: str


@dataclass(frozen=True)
class TalUtteranceId:
    number: int
    prompt_tag: str


@dataclass(frozen=True)
class TalSpontaneousSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class TalUtterance:
    dataset: str
    session_or_speaker: str
    utterance_id: TalUtteranceId
    prompt_txt: Path
    wav: Path
    sync: Path
    ult: Path
    param: Path
    lip_mp4: Path
    lab: Path | None

    @property
    def utterance_key(self) -> str:
        return f"{self.utterance_id.number:03d}_{self.utterance_id.prompt_tag}"

    @property
    def speaker_id(self) -> str | None:
        return self.session_or_speaker if self.dataset == TAL80_NAME else None

    @property
    def session_id(self) -> str | None:
        return self.session_or_speaker if self.dataset == TAL1_NAME else None


_TAL80_SPK_PATTERN = re.compile(r"^(\d{2})([fm])([esino])$")
_UTT_PATTERN = re.compile(r"^(\d{3})_([a-z]+)$")


def parse_tal80_speaker_id(value: str) -> Tal80Speaker:
    """Parse TaL80 speaker ID like 01fi or 03mn."""
    m = _TAL80_SPK_PATTERN.match(value)
    if not m:
        raise ValueError(f"Invalid TaL80 speaker ID: {value}")

    country_code_to_name = {
        "e": "England",
        "s": "Scotland",
        "i": "Ireland",
        "n": "Northern Ireland",
        "o": "Other",
    }
    index = int(m.group(1))
    gender = "female" if m.group(2) == "f" else "male"
    country = country_code_to_name[m.group(3)]
    return Tal80Speaker(raw_id=value, index=index, gender=gender, country=country)


def parse_utterance_id(stem: str) -> TalUtteranceId:
    """Parse utterance stem like 002_cal or 029_xsil."""
    m = _UTT_PATTERN.match(stem)
    if not m:
        raise ValueError(f"Invalid utterance ID format: {stem}")

    tag = m.group(2)
    if tag not in PROMPT_TAGS:
        raise ValueError(f"Unknown prompt tag '{tag}' in utterance ID: {stem}")
    return TalUtteranceId(number=int(m.group(1)), prompt_tag=tag)


def parse_spontaneous_lab(path: Path) -> list[TalSpontaneousSegment]:
    """Parse .lab file format: start end transcript."""
    segments: list[TalSpontaneousSegment] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            raise ValueError(f"Invalid .lab line in {path}: {line}")
        start, end, text = parts
        segments.append(TalSpontaneousSegment(float(start), float(end), text))
    return segments


def discover_tal_utterances(tal_root: Path, dataset_name: str) -> list[TalUtterance]:
    """Discover TaL utterances from TaL1 or TaL80 root path."""
    if dataset_name not in {TAL1_NAME, TAL80_NAME}:
        raise ValueError(f"dataset_name must be {TAL1_NAME} or {TAL80_NAME}: {dataset_name}")

    core_root = tal_root / "core"
    if not core_root.exists():
        raise FileNotFoundError(f"Missing core directory: {core_root}")

    records: list[TalUtterance] = []
    for sess_or_spk_dir in sorted([p for p in core_root.iterdir() if p.is_dir()]):
        sess_or_spk = sess_or_spk_dir.name
        if dataset_name == TAL80_NAME:
            parse_tal80_speaker_id(sess_or_spk)

        txt_files = sorted(sess_or_spk_dir.glob("*.txt"))
        for txt in txt_files:
            utt_id = parse_utterance_id(txt.stem)
            stem = txt.stem
            wav = sess_or_spk_dir / f"{stem}.wav"
            sync = sess_or_spk_dir / f"{stem}.sync"
            ult = sess_or_spk_dir / f"{stem}.ult"
            param = sess_or_spk_dir / f"{stem}.param"
            mp4 = sess_or_spk_dir / f"{stem}.mp4"
            lab = sess_or_spk_dir / f"{stem}.lab"

            required = [wav, sync, ult, param, mp4]
            missing = [p for p in required if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing required modality files for {stem}: {missing}")

            records.append(
                TalUtterance(
                    dataset=dataset_name,
                    session_or_speaker=sess_or_spk,
                    utterance_id=utt_id,
                    prompt_txt=txt,
                    wav=wav,
                    sync=sync,
                    ult=ult,
                    param=param,
                    lip_mp4=mp4,
                    lab=lab if lab.exists() else None,
                )
            )
    return records


def filter_by_prompt_tags(records: Iterable[TalUtterance], include_tags: set[str]) -> list[TalUtterance]:
    unknown = include_tags - PROMPT_TAGS
    if unknown:
        raise ValueError(f"Unknown tags: {sorted(unknown)}")
    return [r for r in records if r.utterance_id.prompt_tag in include_tags]


def speaker_disjoint_split(
    records: list[TalUtterance],
    train_ratio: float,
    seed: int = 7,
) -> tuple[list[TalUtterance], list[TalUtterance]]:
    """Speaker-disjoint split (for TaL80)."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    speakers = sorted({r.session_or_speaker for r in records if r.dataset == TAL80_NAME})
    if len(speakers) < 2:
        raise ValueError("Need at least two speakers for speaker-disjoint split")

    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_train = max(1, int(len(speakers) * train_ratio))
    n_train = min(n_train, len(speakers) - 1)
    train_spk = set(speakers[:n_train])

    train = [r for r in records if r.dataset == TAL80_NAME and r.session_or_speaker in train_spk]
    test = [r for r in records if r.dataset == TAL80_NAME and r.session_or_speaker not in train_spk]
    return train, test


def prompt_disjoint_split(
    records: list[TalUtterance],
    train_tags: set[str],
    test_tags: set[str],
) -> tuple[list[TalUtterance], list[TalUtterance]]:
    """Prompt-tag disjoint split to reduce prompt leakage."""
    overlap = train_tags & test_tags
    if overlap:
        raise ValueError(f"train_tags and test_tags overlap: {sorted(overlap)}")

    train = filter_by_prompt_tags(records, train_tags)
    test = filter_by_prompt_tags(records, test_tags)
    return train, test
