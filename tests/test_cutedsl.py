"""CuTeDSL backend and the pluggability of the `Backend` protocol."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from opforge import Contract, get_backend, kernel, register_backend
from opforge.cache.store import KernelCache
from opforge.errors import KernelSpecError
from opforge.kernel import KernelHandle
from opforge.lang import registered_backends, select_backend

# --------------------------------------------------------------------------- #
# Pluggability: without touching the core code at all, registering a new backend makes it
# usable by ops
# --------------------------------------------------------------------------- #


class _EchoArtifact:
    def run(self, args, kwargs):
        return ("echo", args, dict(kwargs))


class _EchoBackend:
    name = "echo-test"

    def supports(self, spec, impl):
        return True

    def build(self, spec, impl, contract):
        return _EchoArtifact()

    def load(self, spec, key, path):
        return None


def _echo_impl(x):
    return ("impl", x)


def _echo_ref(x):
    return ("ref", x)


_echo = kernel(
    name="_echo",
    contract=Contract.FAST,
    reference=_echo_ref,
    backend="echo-test",
)(_echo_impl)


def test_builtin_backends_are_registered():
    assert {"python", "triton", "cutedsl"} <= set(registered_backends())


def test_third_party_backend_can_be_registered_and_selected():
    register_backend(_EchoBackend(), replace=True)
    assert "echo-test" in registered_backends()
    assert get_backend("echo-test").name == "echo-test"
    assert select_backend(_echo.spec).name == "echo-test"


def test_kernel_actually_runs_through_the_third_party_backend():
    register_backend(_EchoBackend(), replace=True)
    handle = KernelHandle(_echo.spec)
    result = handle(x=1)
    assert result == ("echo", (), {"x": 1}), result


def test_duplicate_registration_is_rejected_without_replace():
    register_backend(_EchoBackend(), replace=True)
    try:
        register_backend(_EchoBackend())
    except KernelSpecError:
        return
    raise AssertionError("a duplicate backend name must error rather than silently overwrite")


def test_unknown_backend_name_is_rejected():
    try:
        get_backend("does-not-exist")
    except KernelSpecError:
        return
    raise AssertionError("an unknown backend name must error")


def test_explicit_backend_that_does_not_support_the_kernel_is_rejected():
    class _NeverBackend:
        name = "never-test"

        def supports(self, spec, impl):
            return False

        def build(self, spec, impl, contract):  # pragma: no cover
            raise AssertionError("never called")

        def load(self, spec, key, path):  # pragma: no cover
            return None

    register_backend(_NeverBackend(), replace=True)
    handle = kernel(name="_never", contract=Contract.FAST, reference=_echo_ref, backend="never-test")(
        _echo_impl
    )
    try:
        select_backend(handle.spec)
    except KernelSpecError:
        return
    raise AssertionError(
        "when the explicitly specified backend does not support the op, it must error"
    )


# --------------------------------------------------------------------------- #
# CuTeDSL real op
# --------------------------------------------------------------------------- #


def _requires_cutedsl():
    try:
        from opforge.kernels.elementwise import scale
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest(f"cutedsl kernel unavailable: {exc}") from exc
    if scale is None:  # pragma: no cover
        raise unittest.SkipTest("cutedsl unavailable")
    from opforge.lang.cutedsl_backend import cutedsl_usable

    if not cutedsl_usable():
        raise unittest.SkipTest("cutedsl toolchain unavailable")
    return scale


def test_cutedsl_backend_is_selected_for_the_scale_operator():
    scale = _requires_cutedsl()
    assert select_backend(scale.spec).name == "cutedsl"
    assert scale.spec.backend == "cutedsl"


def test_cutedsl_build_compiles_at_build_time():
    """The AOT half: `cute.compile` happens at **build time**, and the artifact carries
    the compiled result.

    Uses an **isolated cache root**: otherwise an artifact left in the default cache by
    a previous run would be loaded, `prewarm_results` would become `"loaded"` instead of
    `"ok"`, and the test would depend on machine state.
    """
    scale = _requires_cutedsl()
    handle = KernelHandle(scale.spec, cache=KernelCache(root=tempfile.mkdtemp()))
    artifact = handle.build()
    assert type(artifact).__name__ == "CuTeDSLArtifact"
    assert artifact.prewarm_results == ("ok",), artifact.prewarm_results


def test_source_fingerprint_is_stable_across_compilation():
    """The source fingerprint must be **consistent before and after compilation**.

    CuTeDSL's preprocessor rewrites the function object passed to `cute.compile`, and
    `inspect.getsource` then returns different text -- if the source fingerprint were not
    pinned at **declaration time**, the cache key computed by the same op would change
    after compilation: a handle created anew in the same process would recompile, and
    cross-process reuse would fail outright.
    """
    scale = _requires_cutedsl()
    cache_root = tempfile.mkdtemp()
    first = KernelHandle(scale.spec, cache=KernelCache(root=cache_root))
    key_before = first.build_key(scale.spec.contract).entry_id
    first.build()

    second = KernelHandle(scale.spec, cache=KernelCache(root=cache_root))
    assert second.build_key(scale.spec.contract).entry_id == key_before


def test_artifact_is_reused_from_disk_without_recompiling():
    """The equivalent of cross-process reuse: another cache instance (**sealed
    compilation**) can load from disk.

    Sealed compilation is the key -- a disk hit must never fall back to `cute.compile`,
    otherwise it raises `CompilationSealed`; getting `prewarm_results == ("loaded",)`
    proves it was a load rather than a compile.
    """
    scale = _requires_cutedsl()
    cache_root = pathlib.Path(tempfile.mkdtemp())

    producer = KernelCache(root=cache_root)
    built = KernelHandle(scale.spec, cache=producer).build()
    assert built.prewarm_results == ("ok",)
    assert list(cache_root.rglob("*.artifact")), "the artifact should land on disk"
    assert list(cache_root.rglob("*.manifest.json")), "the manifest should land on disk"

    consumer = KernelCache(root=cache_root)
    consumer.seal()
    loaded = KernelHandle(scale.spec, cache=consumer).build()
    assert loaded.prewarm_results == ("loaded",)
    assert consumer.stats == {"hits": 1, "misses": 0, "cached": 1}, consumer.stats


def test_cutedsl_operator_without_examples_is_rejected_at_build():
    """Example inputs are a required declaration -- a missing one must error **at build
    time**, not at runtime."""
    _requires_cutedsl()
    import cutlass.cute as cute
    from opforge.lang.cutedsl_backend import declare_cutedsl

    @cute.jit
    def _host_without_examples(x):
        return None

    declare_cutedsl(_host_without_examples)  # deliberately no examples
    handle = kernel(
        name="_no_examples",
        contract=Contract.FAST,
        reference=_echo_ref,
        backend="cutedsl",
    )(_host_without_examples)

    try:
        handle.build()
    except Exception as exc:
        assert "example" in str(exc).lower(), exc
        return
    raise AssertionError("a CuTeDSL op missing example inputs must error at build time")


def test_cutedsl_launch_when_the_driver_supports_it():
    """Full execution -- requires a driver version matching cuda-python."""
    scale = _requires_cutedsl()
    from opforge.lang.cutedsl_backend import cutedsl_launchable

    if not cutedsl_launchable():
        raise unittest.SkipTest(
            "CuTeDSL cannot launch here (usually cudaErrorInsufficientDriver: "
            "driver older than cuda-python's CUDA version)"
        )

    torch = __import__("torch")
    handle = KernelHandle(scale.spec)
    x = torch.randn(1024, device="cuda")
    y = torch.empty_like(x)
    handle(x, 2.0, y)
    torch.cuda.synchronize()
    assert torch.allclose(y, x * 2.0)
