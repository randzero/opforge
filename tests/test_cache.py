"""L2 cache and launch layers: fingerprints, manifest, sealed compilation, launch signatures."""

from __future__ import annotations

import tempfile
from pathlib import Path

from opforge import (
    BuildKey,
    Contract,
    KernelCache,
    Manifest,
    source_fingerprint,
    target_fingerprint,
)
from opforge.cache.manifest import file_sha256
from opforge.errors import CompilationSealed, ManifestError
from opforge.launch import LaunchSpec, Launcher
from support import FakeArtifact, FakeTensor


def _key(namespace: str = "t", contract: Contract = Contract.HIGH_PRECISION) -> BuildKey:
    return BuildKey(
        source_fingerprint=source_fingerprint({"op": f"def {namespace}(): pass"}),
        target_fingerprint=target_fingerprint(),
        namespace=namespace,
        contract=contract,
    )


def test_source_fingerprint_is_content_addressed():
    a = source_fingerprint({"op": "body"})
    assert a == source_fingerprint({"op": "body"}), "equal content must give equal fingerprints"
    assert a != source_fingerprint({"op": "body "}), "changed content must change the fingerprint"
    assert a != source_fingerprint({"op": "body"}, namespace="ns"), (
        "the namespace must affect the fingerprint"
    )


def test_source_fingerprint_ignores_key_insertion_order():
    assert source_fingerprint({"a": "1", "b": "2"}) == source_fingerprint({"b": "2", "a": "1"})


def test_build_key_separates_contract_variants():
    """Different contracts of the same op must be different cache entries (otherwise
    they would hit the wrong artifact)."""
    det = _key(contract=Contract.DETERMINISTIC)
    fast = _key(contract=Contract.FAST)
    assert det.entry_id != fast.entry_id
    assert det.export_symbol != fast.export_symbol


def test_build_key_is_stable_and_paths_are_layered():
    key = _key()
    assert key.entry_id == _key().entry_id
    root = Path("/tmp/opforge-test-cache")
    # target outermost (drop a machine's artifacts in one delete), source innermost
    assert key.directory(root).parts[-3:] == (
        key.target_fingerprint,
        "t",
        key.source_fingerprint,
    )


def test_disk_round_trip_avoids_rebuilding():
    root = Path(tempfile.mkdtemp())
    key = _key()
    cache = KernelCache(root, persist=True)
    builds = []

    def build():
        builds.append(1)
        return FakeArtifact("v1")

    first = cache.get_or_build(key, build, load=lambda k, p: FakeArtifact("from-disk"))
    assert first.tag == "v1" and len(builds) == 1

    cache.clear_memory()  # simulate a "new process"
    second = cache.get_or_build(key, build, load=lambda k, p: FakeArtifact("from-disk"))
    assert second.tag == "from-disk", "the second time should hit from disk"
    assert len(builds) == 1, "a disk hit must not trigger a recompile"


def test_manifest_is_required_and_verified():
    root = Path(tempfile.mkdtemp())
    key = _key(namespace="manifest")
    cache = KernelCache(root, persist=True)
    cache.get_or_build(key, FakeArtifact, load=lambda k, p: FakeArtifact())

    # 1) manifest corrupted
    cache.clear_memory()
    key.manifest_path(root).write_text('{"format_version": 999}')
    for description, expected in (("corrupted manifest", ManifestError),):
        try:
            cache.get_or_build(key, FakeArtifact, load=lambda k, p: FakeArtifact())
        except expected:
            pass
        else:
            raise AssertionError(f"{description} should raise ManifestError")

    # 2) manifest missing
    cache.clear_memory()
    key.manifest_path(root).unlink()
    try:
        cache.get_or_build(key, FakeArtifact, load=lambda k, p: FakeArtifact())
    except ManifestError:
        pass
    else:
        raise AssertionError("a missing manifest should raise ManifestError")

    # 3) digest mismatch
    cache.clear_memory()
    Manifest.create(key, artifact_digest="0" * 64).write(key.manifest_path(root))
    try:
        cache.get_or_build(key, FakeArtifact, load=lambda k, p: FakeArtifact())
    except ManifestError:
        pass
    else:
        raise AssertionError("an artifact digest mismatch should raise ManifestError")


def test_manifest_round_trip_preserves_all_fields():
    key = _key(namespace="roundtrip")
    manifest = Manifest.create(key, artifact_digest="a" * 64)
    parsed = Manifest.loads(manifest.to_json())
    assert parsed == manifest
    assert parsed.entry_id == key.entry_id
    assert parsed.export_symbol == key.export_symbol


def test_seal_blocks_runtime_compilation():
    cache = KernelCache(Path(tempfile.mkdtemp()), persist=False)
    cache.seal()
    try:
        cache.get_or_build(_key(namespace="sealed"), FakeArtifact)
    except CompilationSealed:
        pass
    else:
        raise AssertionError("build still succeeded after sealed compilation")
    assert cache.compiled_since_seal == ("sealed",), cache.compiled_since_seal
    assert cache.sealed


def test_seal_still_serves_already_built_kernels():
    cache = KernelCache(Path(tempfile.mkdtemp()), persist=False)
    key = _key(namespace="warm")
    cache.get_or_build(key, FakeArtifact)  # prewarm
    cache.seal()
    assert cache.get_or_build(key, FakeArtifact) is not None, (
        "a prewarmed op must not be blocked by sealed compilation"
    )
    assert cache.compiled_since_seal == ()


def test_launcher_signature_ignores_scalar_values_but_tracks_shapes():
    launcher = Launcher()
    artifact = FakeArtifact()

    for eps in (1e-6, 1e-5, 3.14):
        launcher(LaunchSpec("op", artifact, (FakeTensor((4, 128)),), {"eps": eps}))
    assert launcher.cached_signatures == 1, "scalar values must not enter the launch signature"

    launcher(LaunchSpec("op", artifact, (FakeTensor((8, 128)),), {"eps": 1e-6}))
    assert launcher.cached_signatures == 2, "tensor shapes should enter the launch signature"


def test_launcher_distinguishes_dtype_and_kernel_id():
    launcher = Launcher()
    artifact = FakeArtifact()
    launcher(LaunchSpec("op", artifact, (FakeTensor((4, 8), "float32"),)))
    launcher(LaunchSpec("op", artifact, (FakeTensor((4, 8), "bfloat16"),)))
    launcher(LaunchSpec("other", artifact, (FakeTensor((4, 8), "float32"),)))
    assert launcher.cached_signatures == 3


def test_launcher_actually_invokes_the_artifact():
    launcher = Launcher()
    artifact = FakeArtifact("hit")
    result = launcher(LaunchSpec("op", artifact, (1,), {"k": 2}))
    assert result[0] == "hit" and artifact.calls == 1
    launcher(LaunchSpec("op", artifact, (1,), {"k": 2}))
    assert artifact.calls == 2, "even a cache hit must actually execute the artifact"


def test_manifest_digest_uses_real_file_hashing():
    """Ensure verification uses a real file digest, not a constant or placeholder."""
    root = Path(tempfile.mkdtemp())
    key = _key(namespace="digest")
    cache = KernelCache(root, persist=True)
    cache.get_or_build(key, FakeArtifact, load=lambda k, p: FakeArtifact())
    assert file_sha256(key.artifact_path(root)) == Manifest.load(
        key.manifest_path(root)
    ).artifact_digest
