"""Artifact description (manifest) -- make "using the wrong artifact" a **loud failure**.

See `docs/architecture.md` §4.2. Three loading contracts:

1. manifest missing -> raise (no silent recompile, no silent use);
2. any field does not match the current :class:`BuildKey` -> raise;
3. artifact digest mismatch -> raise.

Better to crash than to be wrong: using the wrong artifact shows up as numeric errors
far downstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..errors import ManifestError
from .fingerprint import CACHE_SCHEMA_VERSION, BuildKey

_REQUIRED_FIELDS = (
    "format_version",
    "source_fingerprint",
    "target_fingerprint",
    "namespace",
    "contract",
    "entry_id",
    "export_symbol",
    "artifact_digest",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    format_version: int
    source_fingerprint: str
    target_fingerprint: str
    namespace: str
    contract: str
    entry_id: str
    export_symbol: str
    artifact_digest: str

    @classmethod
    def create(cls, key: BuildKey, *, artifact_digest: str) -> "Manifest":
        return cls(
            format_version=CACHE_SCHEMA_VERSION,  # the manifest's projection of CACHE_SCHEMA_VERSION
            source_fingerprint=key.source_fingerprint,
            target_fingerprint=key.target_fingerprint,
            namespace=key.namespace,
            contract=key.contract.name,
            entry_id=key.entry_id,
            export_symbol=key.export_symbol,
            artifact_digest=artifact_digest,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def loads(cls, text: str) -> "Manifest":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ManifestError("manifest must be a JSON object")
        missing = [f for f in _REQUIRED_FIELDS if f not in raw]
        if missing:
            raise ManifestError(f"manifest is missing fields: {missing}")
        return cls(**{f: raw[f] for f in _REQUIRED_FIELDS})

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(f"manifest is missing or unreadable: {path}: {exc}") from exc
        return cls.loads(text)

    def verify(self, key: BuildKey, artifact_path: Path) -> None:
        """Verify fingerprints, format version, and artifact digest; any mismatch raises
        :class:`ManifestError`."""
        expected = {
            "format_version": CACHE_SCHEMA_VERSION,
            "source_fingerprint": key.source_fingerprint,
            "target_fingerprint": key.target_fingerprint,
            "namespace": key.namespace,
            "contract": key.contract.name,
            "entry_id": key.entry_id,
            "export_symbol": key.export_symbol,
        }
        actual = asdict(self)
        mismatches = {
            name: {"expected": value, "actual": actual.get(name)}
            for name, value in expected.items()
            if actual.get(name) != value
        }
        if mismatches:
            raise ManifestError(
                f"incompatible artifact at {artifact_path}: {mismatches}"
            )
        digest = file_sha256(Path(artifact_path))
        if digest != self.artifact_digest:
            raise ManifestError(
                f"artifact digest mismatch at {artifact_path}: "
                f"expected {self.artifact_digest}, got {digest}"
            )

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(path)
