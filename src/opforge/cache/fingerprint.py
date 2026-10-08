"""Two-level fingerprints -- the cornerstone of cache correctness.

See `docs/architecture.md` §4.1. Two hard rules:

* **everything that shapes the generated code goes in**: operator source, DSL /
  dependency versions, target architecture, platform and SOABI;
* **the shapes of this run never go in**: token counts, sequence lengths, batch sizes
  are all runtime parameters, masked / guarded inside the kernel. Putting them in the
  fingerprint = cache explosion + repeated recompilation in production (invariant I5).
"""

from __future__ import annotations

import functools
import hashlib
import os
import platform
import sys
import sysconfig
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping

from ..contract import Contract

#: Versions the whole on-disk cache schema: path layout, manifest keys, and the
#: hash inputs. Bump it whenever any of those change (it lands in the source fingerprint
#: and in every artifact path, so old entries are cleanly orphaned).
CACHE_SCHEMA_VERSION = 1

CACHE_ROOT_ENV = "OPFORGE_CACHE_DIR"
TARGET_ARCH_OVERRIDE_ENV = "OPFORGE_TARGET_ARCH"

#: Dependencies that affect the generated code. When present, their "version + RECORD
#: digest" goes into the source fingerprint; the RECORD digest distinguishes images
#: with the same version number but different binary contents.
SOURCE_PACKAGES: tuple[str, ...] = (
    "torch",
    "triton",
    "nvidia-cutlass-dsl",
    "cuda-python",
)


def function_source(fn: Callable) -> str:
    """Get a function's source text; returns a **stable** placeholder when unavailable
    (no exception)."""
    try:
        import inspect

        return inspect.getsource(fn)
    except (OSError, TypeError):
        return f"<source-unavailable:{getattr(fn, '__qualname__', repr(fn))}>"


def _version_of(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "absent"


def _record_digest(package: str) -> str:
    try:
        dist = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return "absent"
    text = dist.read_text("RECORD") or dist.read_text("METADATA") or ""
    if not text:
        return "absent"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def source_fingerprint(
    sources: Mapping[str, str],
    *,
    namespace: str = "",
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Source fingerprint = operator source + dependency identity + namespace.

    ``sources`` is ``{name: source text}``; keys are hashed after sorting by name,
    guaranteeing "same content, same fingerprint".
    """
    digest = hashlib.sha256()
    digest.update(f"schema={CACHE_SCHEMA_VERSION}".encode())
    digest.update(f"python={sys.version_info.major}.{sys.version_info.minor}".encode())
    for package in SOURCE_PACKAGES:
        digest.update(f"{package}={_version_of(package)}".encode())
        digest.update(f"{package}.record={_record_digest(package)}".encode())
    digest.update(f"namespace={namespace}".encode())
    for name in sorted(sources):
        digest.update(name.encode())
        digest.update(sources[name].encode())
    for name in sorted(extra or {}):
        digest.update(f"extra.{name}={extra[name]}".encode())
    return digest.hexdigest()


def detected_target_arch() -> str:
    """Target device architecture: environment variable first, then torch, finally
    ``unknown``."""
    forced = os.environ.get(TARGET_ARCH_OVERRIDE_ENV)
    if forced:
        return forced
    try:
        import torch  # optional dependency, imported lazily
    except Exception:
        return "unknown"
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"sm_{major}{minor}"
        cuda = getattr(getattr(torch, "version", None), "cuda", None)
        return f"cpu+torch{cuda}" if cuda else "cpu"
    except Exception:
        return "unknown"


@functools.lru_cache(maxsize=1)
def target_fingerprint() -> str:
    """Target fingerprint.

    Deliberately cached once (``lru_cache(maxsize=1)``): the hot path must not
    repeatedly probe the device and driver.
    """
    digest = hashlib.sha256()
    libc_name, libc_version = platform.libc_ver()
    fields = (
        ("arch", detected_target_arch()),
        ("machine", platform.machine()),
        ("libc", f"{libc_name}:{libc_version}"),
        ("soabi", str(sysconfig.get_config_var("SOABI") or "unknown")),
    )
    for key, value in fields:
        digest.update(f"{key}={value}".encode())
    return digest.hexdigest()


def reset_target_fingerprint_cache() -> None:
    """Clear the target fingerprint cache (for tests; also callable after the in-process
    environment changes)."""
    target_fingerprint.cache_clear()


@dataclass(frozen=True)
class BuildKey:
    """Cache key = two-level fingerprints + namespace + contract.

    **The contract goes into the key**: the `_fast` and `_det` variants of the same
    operator are two distinct artifacts and must never hit each other.
    """

    source_fingerprint: str
    target_fingerprint: str
    namespace: str
    contract: Contract

    @property
    def entry_id(self) -> str:
        """Stable id of this cache entry: a hash over both fingerprints, the namespace
        and the contract."""
        digest = hashlib.sha256()
        for part in (
            self.source_fingerprint,
            self.target_fingerprint,
            self.namespace,
            self.contract.name,
        ):
            digest.update(part.encode())
            digest.update(b"\0")
        return digest.hexdigest()

    @property
    def export_symbol(self) -> str:
        """The export symbol name is derived from namespace + entry id, avoiding symbol
        collisions when one process loads several artifacts."""
        stem = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in (self.namespace or "op")
        )
        return f"opforge_{stem}_{self.entry_id[:16]}"

    def directory(self, root: Path) -> Path:
        """``<root>/v<schema>/<target_fp>/<namespace>/<source_fp>``.

        Target outermost: "drop every artifact built for this accelerator" is then a
        single recursive delete. Source innermost: it is the level that changes on every
        source edit, so the churn stays as deep as possible.
        """
        return (
            Path(root)
            / f"v{CACHE_SCHEMA_VERSION}"
            / self.target_fingerprint
            / (self.namespace or "_")
            / self.source_fingerprint
        )

    def artifact_path(self, root: Path) -> Path:
        return self.directory(root) / f"{self.entry_id}.artifact"

    def manifest_path(self, root: Path) -> Path:
        return self.directory(root) / f"{self.entry_id}.manifest.json"
