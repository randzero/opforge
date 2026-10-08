# opforge — Architecture and interface draft

> Version: v0.1 (draft)
> **This document answers "how it is built" (the layered contracts and interfaces); [`design.md`](design.md) answers "why" (the trade-offs and motivations).**
> When the two conflict, this document's interface definitions take precedence; for motivation, still see design.md.

---

## 0. Layered contracts at a glance

Each layer exposes **only** the items in the right-hand column below; the "not allowed" column is a hard constraint, checked by both CI and code review.

| Layer | Responsibility | Exposes upward | Not allowed |
|---|---|---|---|
| **L0** language layer | Compile an operator description into a loadable artifact | `Backend` protocol, `@cpp_kernel` escape-hatch registration | Must not be aware of any training/inference framework |
| **L1** kernel layer | Pure-tensor forward / backward / reference | `@kernel` decorator, `KernelHandle` | Must not import any training/inference framework |
| **L2** contract layer | Contract model, variant resolution, preconditions | `Contract`, `Precondition`, `resolve()` | Must not bypass the contract to select an implementation directly |
| **L2** cache layer | Fingerprints, manifest, loading, seal compilation | `KernelCache`, `Manifest` | Must not silently fall back to JIT |
| **L2** launch layer | driver-call caching | `LaunchSpec`, `Launcher` | Must not put runtime dynamic dimensions into the signature |
| **L3** adapter layer | Training/inference integration for each | `to_autograd()`, `to_custom_op()` | **Must not be reverse-depended on by L0–L2** |

---

## 1. L0 · Language layer

### 1.1 Backend protocol

```python
class Backend(Protocol):
    name: str                      # "triton" | "cutedsl" | "cpp" | "python"

    def supports(self, spec: KernelSpec, impl: Callable) -> bool: ...

    def build(self, spec: KernelSpec, impl: Callable, contract: Contract) -> Artifact: ...

    def load(self, spec: KernelSpec, key: BuildKey, path: Path) -> Artifact | None: ...
```

Four backends are built in:

| Backend | Purpose | Participates in `source_fingerprint`? |
|---|---|---|
| `TritonBackend` | Regular block-shaped operators | Yes (triton version + source) |
| `CuTeDSLBackend` | Operators needing layout / TMA / warp-level control | Yes (cutlass-dsl version + source) |
| `CppBackend` | **Escape hatch** (see 1.2) | Yes (source + compiler version) |
| `PythonBackend` | Pure-Python reference backend, no compilation; lets the pipeline run in a CPU-only environment | Yes (source) |

**Backend selection is by `impl`, not by `spec`**—different contract tiers of the same operator may land on different backends
(the default tier on Triton, the FAST tier on CuTeDSL). `declare_triton` / `declare_cutedsl` / `declare_cpp`
tag the implementation with `__opforge_backend__`, and `select_backend(spec, impl)` resolves in three layers:

| Layer | Basis | Strength |
|---|---|---|
| 1 | The backend declared by `impl` itself | **Soft**: if that backend cannot build on this platform, keep probing downward (letting the same operator degrade to an available backend, e.g. CPU-only falling to `python`) |
| 2 | `spec.backend != "auto"` | **Hard**: error if unsatisfied, never silently switch backends (switching backends = switching numerical semantics) |
| 3 | Probe in order per `PROBE_ORDER` | Last resort (the order is fixed to avoid "whoever registers first wins" causing non-determinism) |

### 1.2 Escape-hatch registration (class E)

An escape hatch must be **explicitly registered**; "quietly writing a .cu" is not allowed:

```python
@cpp_kernel(
    reason=EscalationReason.MISSING_PRIMITIVE,   # required
    primitives=["tma_descriptor", "named_barrier"],  # which primitives are missing
)
def attention_fwd(...): ...
```

| `EscalationReason` | Meaning | Allowed? |
|---|---|---|
| `MISSING_PRIMITIVE` | The DSL has no corresponding primitive (class A) | ✅ Allowed |
| `CALL_SHAPE_MISMATCH` | Only the call shape does not match (class B) | ❌ **Rejected**, requiring the L2 bucketing idiom instead |
| `PERF_GAP` | The DSL can express it but misses the performance target | ⚠️ Allowed, but benchmark evidence must be attached |

> `CALL_SHAPE_MISMATCH` is explicitly rejected to prevent class-B cases from quietly sinking down—that would force users to install the nvcc toolchain.
> CI tracks the **escape-hatch ratio** (escape-hatch operators / total operators) and warns when it exceeds a threshold.

---

## 2. L1 · Kernel layer

### 2.1 `KernelSpec` (operator metadata)

```python
@dataclass(frozen=True)
class KernelSpec:
    name: str
    default_impl: Callable                  # the decorated function (= the default tier)
    contract: Contract                      # default contract
    params: tuple[ParamSpec, ...] = ()
    preconditions: tuple[Precondition, ...] = ()
    variants: Mapping[Contract, Callable] = field(default_factory=dict)
    reference: Callable | None = None
    backward: Callable | None | str = None
    supports_dynamic_shape: bool = False
    min_alignment: int | None = None        # element alignment requirement
    backend: str = "auto"
    tags: tuple[str, ...] = ()
    module: str = ""
    impl_sources: Mapping[str, str] = field(default_factory=dict)
    """Each implementation's source text **as declared** -- captured at decoration time for
    the source fingerprint (see §4.4)."""
```

### 2.2 `@kernel` decorator (full signature)

```python
def kernel(
    *,
    name: str,
    contract: Contract,
    preconditions: Sequence[Precondition] = (),
    variants: Mapping[Contract, Callable] | None = None,
    reference: Callable | None = None,
    backward: Callable | None | Literal["auto"] = None,
    supports_dynamic_shape: bool = False,
    min_alignment: int | None = None,
    backend: str = "auto",
    tags: Sequence[str] = (),
) -> Callable[[Callable], KernelHandle]:
    ...
```

| Parameter | Semantics | Default behavior |
|---|---|---|
| `name` | Globally unique operator name | Required |
| `contract` | **Default contract**; the caller may override | Required (no default—forcing the author to take a position) |
| `preconditions` | Entry preconditions, checked by the adapter layer | Empty |
| `variants` | Implementations for other contracts (see §3.2) | Only the decorated function (= the default-contract tier) |
| `reference` | Pure-PyTorch ground truth | Looked up lazily by the `<operator name>_ref` naming convention; `handle.ref()` raises if absent |
| `backward` | Hand-written backward; `"auto"` = use the reference's autograd | `"auto"` |
| `supports_dynamic_shape` | Whether shapes may contain symbolic dimensions | `False` (conservative) |
| `min_alignment` | Alignment requirement; if unsatisfied, a precondition failure is triggered | `None` |
| `backend` | Specifies the backend; when not `"auto"` it is a **hard** requirement (error if unsatisfied) | `"auto"` |
| `tags` | For CI grouping / filtering | Empty |

**Note that `contract` is required**—this is deliberate: no default is provided, forcing the author to think through its numerical semantics the moment the operator is written.

### 2.3 `KernelHandle`

```python
class KernelHandle:
    spec: KernelSpec
    name: str                            # == spec.name
    contract: Contract                   # the default contract
    reference: Callable | None           # pure-PyTorch ground truth (resolved lazily)
    backward: Callable | None | str      # hand-written backward, or "auto"

    def __call__(self, *args, contract: Contract | None = None,
                 exec_context: ExecContext | None = None, **kwargs): ...
    def build(self, contract: Contract | None = None) -> Any: ...   # the AOT entry point
    def ref(self, *args, **kwargs) -> Any: ...                      # call the reference directly
    def build_key(self, contract: Contract) -> BuildKey: ...
```

**Call semantics**:

1. Take `contract` (defaulting to `spec.contract`);
2. Have L2 resolve an available implementation (§3.3);
3. Check `preconditions` (§3.4); if unsatisfied, **return `Unavailable`** (not raise);
4. Execute through L2's `Launcher`.

> Item 3's "return `Unavailable` instead of raising" is deliberate: an operator being unavailable in a given scenario is the **norm** (e.g. during CUDA Graph capture), so the caller needs a fallback path rather than being interrupted by an exception.

---

## 3. L2 · Contract layer

### 3.1 `Contract` enum

```python
class Contract(IntEnum):
    DETERMINISTIC  = 0   # Strictest: bitwise-identical across ranks / runs
    HIGH_PRECISION = 1   # fp32 accumulation; non-deterministic order allowed
    FAST           = 2   # Loosest: split-k non-determinism and low-precision accumulation allowed
```

| Contract | Guarantees | Prohibits | Typical use |
|---|---|---|---|
| `DETERMINISTIC` | Fixed reduction order; consistent across ranks | Atomic accumulation, randomized reduction, non-deterministic split | Training, reproducible experiments |
| `HIGH_PRECISION` | fp32 (or higher) accumulation | Low-precision accumulation | Training, evaluation |
| `FAST` | No additional guarantees | — | Inference |

**Partial order**: `DETERMINISTIC ⊒ HIGH_PRECISION ⊒ FAST`
(further left is **stricter**; a "strict" implementation can satisfy a "loose" request, but not vice versa.)

### 3.2 Variant naming and derivation

| Item | Rule |
|---|---|
| In-file naming | `forward` (default contract), `forward_det` / `forward_hp` / `forward_fast` |
| Suffix | `det` / `hp` / `fast`, mapping one-to-one onto `Contract` |
| Explicit declaration | `variants={Contract.FAST: forward_fast}` (recommended, so the naming convention is not broken) |
| Same-name conflict | Error when both the naming convention and a `variants` declaration exist |

**Prohibited**: expressing the contract with a boolean parameter (`forward(..., deterministic=True)`). A contract can only be a **separate implementation** or a `Contract` parameter.

### 3.3 Variant resolution (`resolve`) — the core rule of this project

```python
def resolve(requested: Contract, impls: Mapping[Contract, Callable]) -> Callable:
    """Return an implementation that satisfies requested; fallback may only go in the [stricter] direction, else error.

    Smaller enum values are stricter: DETERMINISTIC(0) ⊒ HIGH_PRECISION(1) ⊒ FAST(2).
    """
    # Scan from loosest to strictest: prefer the most economical implementation; if none is found, tighten all the way, and if tightening still fails, error.
    for c in (Contract.FAST, Contract.HIGH_PRECISION, Contract.DETERMINISTIC):
        if c.value <= requested.value and c in impls:   # c is at least as strict as requested
            return impls[c]
    raise ContractUnsatisfied(requested, sorted(impls))
```

The three columns below each indicate which implementations **impls contains only** (not "at least contains"):

| Request | impls contains only `{det, fast}` | impls contains only `{fast}` | impls contains only `{det}` |
|---|---|---|---|
| `DETERMINISTIC` | Use `det` | ❌ **Error** | Use `det` |
| `HIGH_PRECISION` | Use `det` (stricter, acceptable) | ❌ **Error** | Use `det` |
| `FAST` | Use `fast` (prefer the most economical) | Use `fast` | Use `det` (fallback to stricter, acceptable) |

**Two invariants** (enforced by CI):

1. **Fallback may only go stricter**—never use `fast` to satisfy a `det` request;
2. **Fallback must be visible**—every "requested differs from actual" must be counted (`opforge_fallback_total{op,requested,actual}`); silence is not allowed.

> Motivation: when determinism is carried by an ad-hoc boolean flag, the most likely failure is "some call site forgot to pass it, so it quietly took a non-deterministic path". Making the fallback direction one-way plus observable leaves such bugs nowhere to hide.

### 3.4 `Precondition` protocol

```python
class Precondition(Protocol):
    name: str
    def check(self, call: CallContext) -> PreconditionResult: ...
```

`CallContext` carries `args` / `kwargs` plus an `ExecContext` provided by the adapter layer:
`device` / `world_size` / `rank` / `is_capturing` (CUDA Graph) / `is_warming_up`.
Tensor checks use **duck typing** (`.shape` / `.dtype` only), so the core layer needs no torch.

**Built-in conditions** (generically named, extensible):

| Condition | What it checks | Consequence if unsatisfied (when unchecked) |
|---|---|---|
| `RequiresAllRanksLockstep()` | Whether all ranks are in lockstep | Multicast/spin barrier **deadlock** (every GPU at 100% SM) |
| `RequiresNoCapture()` | That no CUDA Graph capture is in progress | Undefined behavior during capture / invalid graph |
| `RequiresAligned(multiple_of=n)` | The primary input's last dim is a multiple of `n` | Illegal kernel memory access or performance collapse |
| `RequiresMinLeadingDim(minimum=n)` | Dim 0 of the primary input is at least `n` | Taken a branch unsuited to small leading dims |
| `RequiresDtypeIn(*dtypes)` | The primary input's dtype is in the allowlist | Silently take the wrong branch |

Unsatisfied conditions make the operator return `Unavailable` (§2.3), not raise.

**Constraint**: `check()` must be a **cheap pure function** (must not synchronize CUDA, must not allocate).

---

## 4. L2 · Cache layer

### 4.1 Two-level fingerprints

```python
@dataclass(frozen=True)
class BuildKey:
    source_fingerprint: str      # see below
    target_fingerprint: str
    namespace: str
    contract: Contract           # ★ the contract goes into the key
```

| Fingerprint | **Must contain** | **Must not contain** |
|---|---|---|
| `source_fingerprint` | Operator source content; DSL / dependency versions; the `RECORD` digest of each dependency wheel | Timestamps, hostname, username |
| `target_fingerprint` | Device compute capability; driver version; CUDA version; platform / libc / Python SOABI | **This run's shapes** |

**Anti-pattern (which causes repeated recompilation in production)**: putting `num_tokens`, `seq_len`, `batch_size` into either fingerprint. These must be passed into the kernel as **runtime arguments** and handled with mask/guard inside the kernel.

### 4.2 `Manifest`

```python
@dataclass(frozen=True)
class Manifest:
    format_version: int
    source_fingerprint: str
    target_fingerprint: str
    namespace: str
    contract: str
    entry_id: str
    export_symbol: str          # derived from namespace + entry_id (to avoid symbol collisions)
    artifact_digest: str
```

**Load contract**:

1. No manifest → **raise** (no silent recompilation, no silent use);
2. Any field not matching the current `BuildKey` → **raise**;
3. `artifact_digest` not matching → **raise**.

> All three follow "crash rather than be wrong". Using a mismatched artifact is worse than crashing—it shows up as a numerical error far downstream.

### 4.3 `KernelCache`

```python
class KernelCache:
    def get_or_build(
        self,
        key: BuildKey,
        build: Callable[[], Artifact],
        *,
        load: Callable[[BuildKey, Path], Artifact | None] | None = None,
    ) -> Artifact: ...
    """memory -> disk -> build. `load` is only consulted when provided, and a present-but-
    mismatched manifest raises instead of falling back to `build`."""

    def seal(self) -> None: ...
    """Seal compilation: from then on any missed build raises outright."""

    @property
    def compiled_since_seal(self) -> tuple[str, ...]: ...
    """Operator names that attempted to compile after the seal (for assertions and warnings)."""
```

**Typical usage** (at the end of a serving process's startup):

```python
cache.seal()
# no compilation should occur again for the whole lifetime; a single assert guarantees it
assert not cache.compiled_since_seal, cache.compiled_since_seal
```

### 4.4 Artifact persistence protocol (optional)

If an artifact wants cross-process reuse, it implements `export`; if not (the pure-Python reference backend), it does not—
the cache layer degrades to a purely in-process cache, and the remaining semantics (hit/miss, seal compilation) are exactly the same.

```python
class PersistableArtifact(Protocol):
    def export(self, path: Path, key: BuildKey) -> None: ...
```

The `key` is passed in so that the artifact can write identity information such as the **export symbol** into it: at load time
`BuildKey.export_symbol` is the sole lookup key, so export and load must use the same name
(CuTeDSL's `export_to_c(function_prefix=...)` ↔ `module[key.export_symbol]`).
Immediately after `export`, the cache layer computes `artifact_digest` and writes it into the manifest,
and at load time it validates the manifest before calling `Backend.load`—**if the manifest does not match, error; never recompile silently**.

How each backend persists:

| Backend | Artifact | Cross-process reuse mechanism |
|---|---|---|
| `python` | None | Does not implement `export`; degrades to in-process cache |
| `triton` | None | Reuses Triton's **own** cache directory (`TRITON_CACHE_DIR`); `load` returns `None` |
| `cutedsl` | `<digest>.artifact` (object file) | Official `export_to_c` + `cute.runtime.load_module`, with the manifest under our control |
| `cpp` | None | Reuses `torch.utils.cpp_extension`'s own extension cache |

> The Triton row means "reuse its own cache directory"—which amounts to treating its internal layout (hash-based directory names, file naming)
> as an interface; the CuTeDSL row goes through the official API and is the cleanest one in this project.

**The source fingerprint must be captured "at declaration time"**: CuTeDSL's preprocessor rewrites the function object that has been through `cute.compile`,
so afterwards `inspect.getsource` reads the rewritten text. If read on demand, the same operator would compute
**two different cache keys** before and after compilation—a newly created handle in the same process would recompile, and cross-process reuse would break outright.
Therefore `@kernel` stores each implementation's source text into `KernelSpec.impl_sources` at decoration time
(the reference is pure Python, is never rewritten, and is still read on demand).

---

## 5. L2 · Launch layer

### 5.1 `LaunchSpec`

```python
@dataclass
class LaunchSpec:
    kernel_id: str
    artifact: Any                     # produced by the cache layer; provides run(args, kwargs)
    runtime_args: tuple = ()
    runtime_kwargs: Mapping[str, Any] = field(default_factory=dict)
    compute_signature: Callable[[tuple, Mapping[str, Any]], Hashable] | None = None
```

| Field | In the signature? | Notes |
|---|---|---|
| `kernel_id` | ✅ | Must be stable; unique **per contract tier** (`rms_norm:fast`) |
| `runtime_args` / `runtime_kwargs` dtype / shape / strides | ✅ | Decides which code path runs |
| **Values** of scalar args (`eps`, flags) | ❌ | Runtime data |
| `artifact` | ❌ | Resolved once and cached, not part of the key |
| pointers / stream / workspace | ❌ | Runtime objects |

The default `compute_signature` (`default_signature`, built on `arg_signature`) renders each argument as
`("tensor", dtype, shape)` / `("scalar", type name)` — so two calls differing only in a
scalar's **value** share one signature. Override it when the operator has a smarter
notion of identity.

**Same-origin rule as §4.1**: runtime dynamic dimensions must not enter the signature—otherwise the in-process dict blows up too.

### 5.2 `Launcher`

```python
class Launcher:
    def __call__(self, spec: LaunchSpec) -> None: ...
```

Semantics: cache **"signature → resolved artifact entry point"**, with runtime arguments passed in on every call.

**Two pitfalls we hit** (caught by tests while implementing the skeleton):

1. **Do not close over this call's arguments in the cache.** The signature intentionally excludes scalar values; if the cache closes over "the first call's arguments",
   subsequent calls with the same signature but different scalar values would get **the previous result**. The cache should hold only the "artifact entry point",
   with arguments passed in on every call.
2. **`kernel_id` must be unique per contract tier** (e.g. `rms_norm:fast`). If all tiers share one id,
   the launch cache would hand a `_det` call to the `_fast` entry point.

---

## 6. L3 · Adapter layer interface

### 6.1 Training side

```python
def to_autograd(op: KernelHandle, contract: Contract | None = None) -> type[torch.autograd.Function]: ...
```

- First version: `forward` calls the kernel, `backward` uses `op.ref`'s autograd (`backward="auto"`);
- Later: swap hand-written `backward` implementations in incrementally, with the **interface unchanged**;
- When `preconditions` are unsatisfied, the training side **falls back to `op.ref`** (pure PyTorch), guaranteeing training always runs.

### 6.2 Inference side

```python
def to_custom_op(
    op: KernelHandle,
    *,
    mutates_args: Sequence[str] = (),
    fake: Callable | None = None,
    variant: Contract = Contract.FAST,
) -> None: ...
```

- Register schema + implementation + **fake** (`fake` only infers shapes and takes no part in contract selection).
  **An operator without a fake cannot be traced by `torch.compile`**—during tracing all tensors are meta, and a missing fake
  would run the real implementation and blow up on device/memory. `to_custom_op` therefore provides a **default fake** for operators whose
  "output shape = first tensor argument"; operators with more complex shape rules must pass `fake=` explicitly;
- `mutates_args` must be declared truthfully (see the lesson in design.md: a wrong declaration invalidates the compiled graph);
- CUDA Graph / dynamic-shape adaptation lives **only in this layer** and is not sunk downward.

---

## 7. End-to-end example: `rms_norm`

```python
# ---------- L1: the operator itself ----------
@kernel(
    name="rms_norm",
    contract=Contract.HIGH_PRECISION,
    variants={Contract.DETERMINISTIC: rms_norm_det},
    reference=rms_norm_ref,
    backward="auto",
    supports_dynamic_shape=True,
)
def rms_norm(x, weight, eps, *, residual=None, zero_centered=False):
    """returns (out, residual_out?)"""

def rms_norm_det(x, weight, eps, *, residual=None, zero_centered=False):
    """Bitwise-identical version (fixed reduction order)"""

def rms_norm_ref(x, weight, eps, *, residual=None, zero_centered=False):
    """Pure-PyTorch ground truth"""

# ---------- L3: inference side ----------
to_custom_op(rms_norm, mutates_args=("residual",), fake=rms_norm_fake)

# ---------- Call ----------
out = rms_norm(x, w, eps, contract=Contract.HIGH_PRECISION)
```

**The pipeline that runs**:

```
rms_norm(...)                    L1
  → resolve(contract, impls)     L2 contract layer   ← fallback may only go stricter, and is counted
  → check(preconditions, ctx)    L2 contract layer   ← returns Unavailable if unsatisfied
  → cache.get_or_build(key)      L2 cache layer      ← on a hit, load directly
  → launcher(spec)               L2 launch layer     ← dict lookup + launch
  → Backend.build(...)           L0                  ← only on a miss
```

---

## 8. Architectural invariants (for review and CI checks)

| # | Invariant | How it is checked |
|---|---|---|
| I1 | L1 code must not import any training/inference framework | import whitelist lint |
| I2 | L0–L2 must not reverse-depend on L3 | Dependency graph check |
| I3 | Contract fallback may only go in the stricter direction | Exhaustive `resolve()` unit tests |
| I4 | Every contract fallback is counted | Metric-existence test |
| I5 | Runtime dynamic dimensions do not enter fingerprints / launch signatures | Fingerprint-stability test (same operator, different shapes → same key) |
| I6 | A missing or mismatched manifest must raise | Negative test that deliberately corrupts the manifest |
| I7 | No compilation may occur after seal compilation | `assert not compiled_since_seal` after warmup |
| I8 | An escape hatch must carry a `reason`, and the ratio is monitored | Decorator assertion + CI statistics |
| I9 | Every operator has a `reference` | Collector validation; missing → CI fails |

---

## Appendix A · Naming quick reference

| Concept | Name |
|---|---|
| Contract enum | `Contract.{DETERMINISTIC,HIGH_PRECISION,FAST}` |
| Variant suffix | `_det` / `_hp` / `_fast` |
| Precondition | `Requires*` |
| Unavailable return | `Unavailable` (not an exception) |
| Cache key | `BuildKey` (= two-level fingerprints + namespace + contract) |
| Artifact descriptor | `Manifest` |
| Launch descriptor | `LaunchSpec` |
