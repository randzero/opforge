"""L2 · cache layer: build / persist / load / sealed compilation.

Division of labor with the launch layer (`launch.launcher`):

* what this saves is **compilation**: expensive, persisted to disk, reused across
  processes;
* what that saves is **the per-call binding overhead**: cheap, in-process.

Sealed compilation (`seal`) is the switch that turns "silently getting slower" into
"failing on the spot": call it once after prewarming, and any subsequent build that
was not prewarmed raises :class:`~opforge.errors.CompilationSealed`.
"""

from __future__ import annotations

import getpass
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Protocol

from ..errors import CompilationSealed
from .fingerprint import CACHE_ROOT_ENV, BuildKey
from .manifest import Manifest, file_sha256


class PersistableArtifact(Protocol):
    """An artifact that the cache layer can persist: just provide ``export(path, key)
    -> None``.

    Non-persistable backends (e.g. the pure Python reference backend) do not implement
    it -- the cache layer then degrades to a purely in-process cache, with all other
    semantics (hit/miss, sealed compilation) identical.

    ``key`` is passed along so the artifact can record identity information such as the
    **export symbol**: at load time
    :func:`~opforge.cache.fingerprint.BuildKey.export_symbol` is the only lookup basis,
    so export and load must use the same name (see the CuTeDSL backend's ``export_to_c``
    + ``load_module``).
    """

    def export(self, path: Path, key: BuildKey) -> None: ...


def default_cache_root() -> Path:
    """Cache root: ``OPFORGE_CACHE_DIR``, otherwise ``<tmp>/<user>/opforge-cache``."""
    env = os.environ.get(CACHE_ROOT_ENV)
    if env:
        return Path(env).expanduser()
    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USER") or "unknown"
    return Path(tempfile.gettempdir()) / user / "opforge-cache"


class KernelCache:
    """Two-level cache: in-process dict (fast) + on-disk artifacts (cross-process).

    Args:
        root: cache root; :func:`default_cache_root` when ``None``.
        persist: whether persisting to disk is allowed. When off, only an in-process
            cache is used.
        namespace: ledger namespace, fed into :class:`BuildKey`.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        persist: bool = True,
        namespace: str = "",
    ) -> None:
        self.root = Path(root) if root is not None else default_cache_root()
        self.persist = persist
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        self._built: dict[str, Any] = {}
        self._sealed = False
        self._compiled_since_seal: list[str] = []
        self._lock = threading.Lock()

    # -- read path ---------------------------------------------------------- #

    def get_or_build(
        self,
        key: BuildKey,
        build: Callable[[], Any],
        *,
        load: Callable[[BuildKey, Path], Any] | None = None,
    ) -> Any:
        """Get the artifact for this key: memory -> disk -> build.

        Args:
            key: cache key made of two-level fingerprints + namespace + contract.
            build: the build function used on a miss.
            load: optional on-disk load function; only when provided is a disk hit
                attempted. Note: when ``load`` exists but the manifest is **missing or
                mismatched**, a :class:`~opforge.errors.ManifestError` is raised and it
                does **not** fall back to build.
        """
        entry_id = key.entry_id
        with self._lock:
            cached = self._built.get(entry_id)
            if cached is not None:
                self.hits += 1
                return cached

        from_disk = self._load_from_disk(key, load)
        if from_disk is not None:
            with self._lock:
                self._built[entry_id] = from_disk
                self.hits += 1
            return from_disk

        with self._lock:
            cached = self._built.get(entry_id)
            if cached is not None:  # another thread just put it in
                self.hits += 1
                return cached
            if self._sealed:
                label = key.namespace or entry_id[:12]
                self._compiled_since_seal.append(label)
                raise CompilationSealed(
                    f"compilation is sealed: {label} ({key.contract.name}) "
                    f"was not prebuilt; refusing to compile at runtime"
                )
            self.misses += 1

        artifact = build()

        with self._lock:
            self._built[entry_id] = artifact
        self._export_to_disk(key, artifact)
        return artifact

    # -- disk --------------------------------------------------------------- #

    def _load_from_disk(
        self,
        key: BuildKey,
        load: Callable[[BuildKey, Path], Any] | None,
    ) -> Any:
        if not (self.persist and load is not None):
            return None
        artifact_path = key.artifact_path(self.root)
        if not artifact_path.exists():
            return None
        manifest = Manifest.load(key.manifest_path(self.root))
        manifest.verify(key, artifact_path)
        return load(key, artifact_path)

    def _export_to_disk(self, key: BuildKey, artifact: Any) -> None:
        if not (self.persist and hasattr(artifact, "export")):
            return
        artifact_path = key.artifact_path(self.root)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact.export(artifact_path, key)
        Manifest.create(key, artifact_digest=file_sha256(artifact_path)).write(
            key.manifest_path(self.root)
        )

    # -- sealed compilation ------------------------------------------------- #

    def seal(self) -> None:
        """Seal compilation: after this, any build on a miss raises."""
        with self._lock:
            self._sealed = True

    @property
    def sealed(self) -> bool:
        with self._lock:
            return self._sealed

    @property
    def compiled_since_seal(self) -> tuple[str, ...]:
        """Operator labels that tried to build after sealing (non-empty means an
        unprewarmed path exists)."""
        with self._lock:
            return tuple(self._compiled_since_seal)

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "cached": len(self._built),
            }

    def clear_memory(self) -> None:
        """Clear the in-process cache (does not affect disk); for tests."""
        with self._lock:
            self._built.clear()
            self.hits = 0
            self.misses = 0


_default_cache = KernelCache()


def get_cache() -> KernelCache:
    """The process-wide default cache."""
    return _default_cache
