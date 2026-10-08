"""Backend selection: three-layer resolution + **per-contract-variant backends**.

This is the easiest place to get wrong in the `Backend` protocol: the backend is a
property of the **implementation**, not of the op -- different contract tiers of the
same op can land on different backends (the default tier goes to Triton, the FAST tier
to CuTeDSL), so `select_backend` must decide based on `impl`, not `spec`.
"""

from __future__ import annotations

from opforge import Contract, kernel, register_backend
from opforge.errors import KernelSpecError
from opforge.kernel import KernelHandle
from opforge.lang import select_backend
from opforge.lang.backend import BACKEND_MARKER_ATTR

# --------------------------------------------------------------------------- #
# A minimal usable third-party backend: only accepts implementations "tagged for it"
# --------------------------------------------------------------------------- #


class _TaggedArtifact:
    def __init__(self, tag, impl):
        self.tag = tag
        self._impl = impl

    def run(self, args, kwargs):
        return (self.tag,) + tuple(self._impl(*args, **kwargs))


class _TaggedBackend:
    def __init__(self, name):
        self.name = name

    def supports(self, spec, impl):
        return getattr(impl, BACKEND_MARKER_ATTR, None) == self.name

    def build(self, spec, impl, contract):
        return _TaggedArtifact(self.name, impl)

    def load(self, spec, key, path):
        return None


class _DecliningBackend:
    """Registered but **always declines** -- used to verify that "a soft declaration degrades"."""

    name = "declining-test"

    def supports(self, spec, impl):
        return False

    def build(self, spec, impl, contract):  # pragma: no cover
        raise AssertionError("declining backend must never build")

    def load(self, spec, key, path):  # pragma: no cover
        return None


def _setup():
    register_backend(_TaggedBackend("tagged-test"), replace=True)
    register_backend(_DecliningBackend(), replace=True)


# --------------------------------------------------------------------------- #
# Layer 1: the backend declared by the implementation itself (soft)
# --------------------------------------------------------------------------- #


def _default_impl(x):
    return (x, "default")


def _fast_impl(x):
    return (x, "fast")


setattr(_fast_impl, BACKEND_MARKER_ATTR, "tagged-test")

_variant_kernel = kernel(
    name="_variant_ops",
    contract=Contract.HIGH_PRECISION,
    variants={Contract.FAST: _fast_impl},
    backend="auto",
)(_default_impl)


def test_impl_declared_backend_wins_for_that_impl():
    _setup()
    assert select_backend(_variant_kernel.spec, _fast_impl).name == "tagged-test"


def test_unmarked_impl_falls_through_to_probe_order():
    _setup()
    # no backend declared -> falls through to PROBE_ORDER; no DSL toolchain here, so python.
    assert select_backend(_variant_kernel.spec, _default_impl).name == "python"


# --------------------------------------------------------------------------- #
# Per-contract-variant backends: handle.build() goes through **the implementation for that tier**
# --------------------------------------------------------------------------- #


def test_backend_is_chosen_per_contract_variant():
    _setup()
    handle = KernelHandle(_variant_kernel.spec)
    assert type(handle.build(Contract.HIGH_PRECISION)).__name__ == "PythonArtifact"
    assert type(handle.build(Contract.FAST)).__name__ == "_TaggedArtifact"


def test_calling_a_variant_runs_through_its_own_backend():
    _setup()
    handle = KernelHandle(_variant_kernel.spec)
    assert handle(7, contract=Contract.FAST) == ("tagged-test", 7, "fast")
    assert handle(7, contract=Contract.HIGH_PRECISION) == (7, "default")


# --------------------------------------------------------------------------- #
# Layer 1 is **soft**: degrade when the declared backend is unavailable; but not when the op
# hard-requires it
# --------------------------------------------------------------------------- #


def _declining_impl(x):
    return (x, "declining")


setattr(_declining_impl, BACKEND_MARKER_ATTR, "declining-test")

_soft_kernel = kernel(
    name="_soft_ops", contract=Contract.FAST, backend="auto"
)(_declining_impl)

_hard_kernel = kernel(
    name="_hard_ops", contract=Contract.FAST, backend="declining-test"
)(_declining_impl)


def test_soft_declaration_degrades_when_the_backend_declines():
    _setup()
    assert select_backend(_soft_kernel.spec, _declining_impl).name == "python"


def test_hard_requirement_refuses_to_degrade():
    _setup()
    try:
        select_backend(_hard_kernel.spec, _declining_impl)
    except KernelSpecError:
        return
    raise AssertionError(
        "when the op hard-requires a backend in the spec, an unsatisfied one must error, "
        "not degrade"
    )


# --------------------------------------------------------------------------- #
# A declaration conflict (the implementation says A, the op's spec says B) must error,
# not silently pick one
# --------------------------------------------------------------------------- #


_conflict_kernel = kernel(
    name="_conflict_ops", contract=Contract.FAST, backend="python"
)(_fast_impl)


def test_conflicting_declarations_are_rejected():
    _setup()
    try:
        select_backend(_conflict_kernel.spec, _fast_impl)
    except KernelSpecError:
        return
    raise AssertionError(
        "when the backend declared by the implementation conflicts with the spec, it must error"
    )
