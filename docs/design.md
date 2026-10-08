# opforge — System Design

> Version: v0.1 (design draft) · Status: draft
> Positioning: an open-source operator framework that serves **both training and inference**: **write JIT, ship AOT, contracts made explicit**.

---

## 1. Background and Goals

### 1.1 Background

Today's operator libraries are essentially split by "use case":

- **Training side**: dominated by C++/CUDA extensions (optimizers, embedding, cross-entropy). The form is "an accelerator part of the framework" — slow to change, high barrier to entry.
- **Inference side**: dominated by JIT DSLs (Triton / CuTeDSL). Fast iteration, but **runtime compilation** brings cold starts and nondeterminism.

The **math kernels on the two sides overlap heavily** (norm, activation, quantization, GEMM, MoE), yet get written twice because the **numeric contracts, execution forms, and delivery modes** differ.
The result is the same computation implemented twice, tuned twice, and buggy twice.

### 1.2 Goals

- **G1 One kernel, two uses**: the same compute kernel serves both training and inference; the differences are isolated through layering rather than duplicating code.
- **G2 Write JIT, ship AOT**: the source form is a JIT DSL (fast iteration, low barrier, AI-friendly); the delivery form is a **precompiled artifact** (production does no runtime compilation).
- **G3 The numeric contract is a first-class citizen**: an operator explicitly declares its own precision/determinism contract, and callers choose per scenario — instead of threading ad-hoc flags through every call site.
- **G4 The escape hatch is controlled**: things the DSL can't express have a clear lowering path (C++/PTX), but **the path must be narrow** — anything that can be absorbed by an idiom must not introduce C++.
- **G5 Delivery is layered**: the training side and the inference side **need not install the same set of dependencies**.

### 1.3 Non-Goals

- Not building an inference serving framework itself (no scheduling, no paged-attention scheduler, no serving gateway).
- Not building a training framework itself (no parallelism strategy, no ZeRO/pipeline orchestration).
- Not building an autograd engine (use the host framework's).
- Not pursuing "one codebase runs on all hardware" — start with one class of accelerators, leave extension points in the interfaces.

---

## 2. Overall Architecture

**Core principle: share the "compute kernel" and the "contract model", but do not share "framework orchestration" and "deployment form".**

```
┌──────────────────────────────────────────────────────────────────────┐
│ L4  Delivery   source distribution  ┊  precompiled AOT (optional)    │
├──────────────────────────────────────────────────────────────────────┤
│ L3  Adapter   ┌─ training side ──────┐  ┌─ inference side ─────────┐ │
│               │ autograd.Function    │  │ custom op + fake         │ │
│               │ optimizer orch.      │  │ CUDA Graph / dyn. shape  │ │
│               └──────────────────────┘  └──────────────────────────┘ │
│               ✗ not shared                      ✗ not shared         │
├──────────────────────────────────────────────────────────────────────┤
│ L2  Contract   numeric contract + variants + preconditions  ✅ shared│
├──────────────────────────────────────────────────────────────────────┤
│ L1  Kernel     forward / backward / reference (pure tensor) ✅ shared│
├──────────────────────────────────────────────────────────────────────┤
│ L0  Language   JIT DSL + C++/PTX escape hatch (narrow)      ✅ shared│
└──────────────────────────────────────────────────────────────────────┘
```

**Why L3 must be separate**: it depends on **framework runtime state** — the autograd graph, ProcessGroup, CUDA Graph capture state, paged KV. Cramming these into the kernel layer would make the kernel layer depend back on some framework, and thus non-portable.

**Why L1/L2 can be shared**: they only require "tensor in → tensor out".

---

## 3. Core Design

### 3.1 An operator = forward + backward + reference + contract

```python
@kernel(
    name="rms_norm",
    contract=Contract.DETERMINISTIC,     # see 3.2
    preconditions=(),                    # see 3.3
)
def rms_norm(x, weight, eps, *, residual=None, zero_centered=False):
    """returns (out, residual_out?)"""

def rms_norm_ref(x, weight, eps, *, residual=None, zero_centered=False):
    """pure PyTorch implementation — both the precision ground truth and the fallback path"""
```

| Component | Role |
|---|---|
| the decorated function | Production implementation (DSL kernel); the default contract's tier |
| `variants={...}` | The other contract tiers |
| `backward` | For training. Can be a hand-written kernel, **or initially derived automatically from `reference`'s autograd** |
| `reference` | Pure PyTorch ground truth. Three roles: ① CI precision baseline ② fallback for the hand-written kernel ③ first version of backward |
| `contract` | see 3.2 |
| `preconditions` | see 3.3 |

**Design stance: derive the interface backward from training as the "strictest constraint"** — training needs backward, which forces forward to expose intermediate quantities (e.g. RMSNorm's inverse-RMS factor, which its backward pass consumes); the inference side happens to use those same intermediates for fusion. Doing it the other way (inference first, training later) tends to reveal that the forward interface isn't differentiable and has to be rewritten.

### 3.2 The numeric contract model

This is the project's **biggest point of difference** from existing operator libraries. A contract is an **attribute** of an operator, not a call parameter:

| Contract | Meaning | Who needs it |
|---|---|---|
| `DETERMINISTIC` | **Bitwise-identical** across ranks / across runs (nondeterministic reductions disabled, accumulation order fixed) | Training, reproducible experiments |
| `HIGH_PRECISION` | fp32 accumulation, nondeterministic order allowed | Training, evaluation |
| `FAST` | split-k nondeterminism and low-precision accumulation allowed | Inference |

```python
# Callers choose per scenario; when unmet, degrade explicitly and log, rather than silently returning a different result
out = op.rms_norm(x, w, eps, contract=Contract.HIGH_PRECISION)
```

**Hard requirements**:

1. Every operator **must** declare a default contract;
2. Variants are derived from the contract, named uniformly (e.g. `_det` / `_hp` / `_fast`); carrying the contract as a boolean parameter is **not allowed**;
3. Cross-rank operators must declare their **communication preconditions** (see 3.3);
4. A contract change = a cache key change (see 3.4).

> Motivation: when the need for "bitwise-identical" results is carried by ad-hoc flags and environment variables, it **bleeds through the entire call chain** until finally nobody can say which semantics a given kernel actually has. Modeling it as a first-class citizen solves this at the root.

### 3.3 Escape hatch: splitting "what the JIT can't write" into two categories

| Category | Cause | Examples | Handling |
|---|---|---|---|
| **Category A** | The DSL **has no corresponding primitive** | TMA descriptors, warp specialization, named barriers, inline PTX, symmetric-memory multicast, device-side collective communication | **Must** use the C++/PTX escape hatch |
| **Category B** | The computation itself is writable, but the **call form / type system doesn't match** | "one kernel consuming N heterogeneous tensors" (optimizer foreach) | **Prefer to absorb within the DSL**, do not introduce C++ |

**Category B's in-DSL solution (recommended)**: bucket by dtype.

```
Naive:         300 tensors → 300 launches        (launch-bound)
C++ templates: 300 tensors → ~2 launches
DSL bucketing: group by dtype → ~3 launches     ← keep the DSL, eliminate 99% of launch overhead
```

Because the number of dtypes is naturally a small constant (2–3 kinds), the number of launches after bucketing is a single digit. **C++ lowering is permitted only for Category A** — every extra category of C++ adds one more build chain, ABI, and delivery burden.

**Preconditions**: an operator **declares** at its entry what it needs, and the adapter layer gates on it. Typical:

```python
preconditions=[RequiresAllRanksLockstep()]   # e.g. multicast barriers need all ranks in lockstep
```

> Motivation: a precondition that lives only in a comment is a precondition nobody checks — until it turns into an incident, e.g. a multi-GPU deadlock during CUDA Graph capture. It must be programmatically checkable.

### 3.4 Compile cache and fingerprints

The biggest cost of writing JIT is **runtime nondeterminism**. The cache is a core feature, not an optimization.

**Two-level fingerprint** (both must go into the key; missing one silently misuses an artifact):

| Fingerprint | Contents |
|---|---|
| `source_fingerprint` | Operator source content + host/numpy/torch DSL versions + **the `RECORD` digest of each dependency wheel** (to distinguish "same version number but different binary") |
| `target_fingerprint` | Device architecture + driver version + CUDA version + platform/glibc/SOABI |

**Layout**:

```
<cache_dir>/v1/<target_fp>/<namespace>/<source_fp>/<entry_id>.artifact
                                              <entry_id>.manifest.json
```

**Hard requirements**:

1. Every artifact **must carry a manifest** recording the fingerprints, exported symbol names, and SHA256; verify on load, and **error out on a mismatch rather than silently falling back to JIT**;
2. **Runtime dynamic dimensions (token count, sequence length) must not enter the key** — otherwise the cache explodes and recompiles repeatedly in production;
3. Exported symbol names are derived from namespace + key (to avoid symbol collisions when loading multiple artifacts in the same process);
4. Provide a **seal-compilation** switch: after warmup, any new compilation errors out immediately — turning "quietly getting slower" into "failing on the spot".

> Concretely per backend: only the CuTeDSL path gets 1–3 all the way through (the official `export_to_c` emits the object,
> `cute.runtime.load_module` loads it, and the manifest records `export_symbol` + SHA256).
> Triton has no export API, so all it can do is relocate its own cache directory, and therefore its "reuse" carries no manifest verification —
> which is exactly why M5 deferred only the Triton half (see §6).

### 3.5 Launch-overhead management (4 layers, stackable)

| Layer | Means | Applies to |
|---|---|---|
| 1 | **driver-call caching**: cache "signature → a built launch call", skipping the DSL's per-call binding / key computation / table lookup | General |
| 2 | **Operator fusion**: N kernels → 1 | General |
| 3 | **PDL** (programmatic dependent launch): the next kernel starts before the previous one finishes | Forward chains |
| 4 | **CUDA Graph**: a chain of launches recorded once, replayed once | **Inference side** (at L3) |

**Note**: what resides in device memory is kernel **code**, which only saves "compile/load", **not launch**. In high-frequency small-kernel scenarios, the bottleneck is always "number of launches × CPU cost per launch".

### 3.6 Delivery and install layering

```
opforge              # L0-L2: kernels + compile cache + contracts     (both install)
opforge[training]    # L3 training adapter: autograd + optimizer orch. (training installs)
opforge[inference]   # L3 inference adapter: custom op + fake + graph  (inference installs)
opforge-prebuilt     # optional: precompiled AOT artifacts (grouped by target_fp) — ⏸ deferred, see §6
```

**Three delivery forms**:

| Form | Who uses it | Cold start |
|---|---|---|
| Source (JIT, first compile) | Training / development | Slow, acceptable |
| Source + local cache | Training / single-machine inference | Fast on same-machine restart |
| **Source + precompiled package (AOT)** | Production inference | Near zero |

Precompiled packages are built by CI on the target architecture and laid out by `target_fingerprint`; when one can't be installed, fall back automatically to the source path.

---

## 4. Directory Structure (proposed)

```
opforge/
├── docs/
│   └── design.md               # this document
├── src/opforge/
│   ├── lang/                   # L0: DSL adapter layer (triton / cutedsl backends + PTX escape hatch)
│   ├── kernels/                # L1: operator implementations (split into directories by domain)
│   │   ├── norm/
│   │   ├── activation/
│   │   ├── quant/
│   │   ├── gemm/
│   │   ├── moe/
│   │   └── attention/
│   ├── contract/               # L2: contract model + variant derivation + preconditions
│   ├── cache/                  # L2: fingerprints / manifest / loading / seal compilation
│   ├── launch/                 # L2: driver-call caching
│   ├── training/               # L3: autograd adapter + optimizer (extras)
│   └── inference/              # L3: custom op / fake / graph adapter (extras)
├── tests/
│   ├── accuracy/               # comparison against the operator's `reference`
│   └── benchmark/              # performance cases
├── tools/                      # CI entry points: run accuracy, run performance, build precompiled packages
└── pyproject.toml
```

---

## 5. Quality Gates / CI

**Every operator ships its own cases, and CI discovers them just by importing the module.**

```python
@benchmark_case(tag="small", axes={"M": 128, "N": 4096, "K": 4096})
@benchmark_case(tag="large", axes={"M": 4096, "N": 4096, "K": 4096})
@accuracy_case(name="rms_norm", seed=16)
def rms_norm(...): ...
```

The accuracy gate needs no wiring: the reference is already declared on the operator
(`@kernel(reference=...)`, or found by the `<name>_ref` convention), so CI reads it off
`KernelHandle.reference`.

| Gate | Criterion |
|---|---|
| Accuracy | Compare against `reference`; exceeding threshold blocks the merge |
| Performance | Diff against the baseline branch; regression beyond threshold raises a warning |
| Contract | Operators marked `DETERMINISTIC` **must** run the "bitwise-identical across two runs + bitwise-identical across multiple ranks" cases |
| Cache | First process produces artifacts → second process hits without recompiling → a changed target fingerprint does not falsely hit |
| Seal compilation | Force no compilation after warmup; any compilation fails |

---

## 6. Roadmap

| Stage | Content | Deliverable | Status |
|---|---|---|---|
| **M1** | L0/L1/L2 skeleton: DSL adapter layer, contract model, cache and fingerprints, driver-call caching | Able to write an operator and reuse its artifact across two processes | ✅ |
| **M2** | First batch of operators (norm / activation / quant / GEMM) + accuracy and performance CI | A usable kernel set | Partial (norm / elementwise / reduction) |
| **M3** | Both L3 adapters: training-side autograd, inference-side custom op + fake | One model end-to-end on each of training and inference | Partial (`to_autograd` / `to_custom_op` available; hand-written backward and CUDA Graph adaptation not done) |
| **M4** | Escape hatch (Category A C++/PTX) | — | ✅ |
| **M4.5** | Real DSL operators (Triton / CuTeDSL) + **CuTeDSL cross-process artifact reuse** | Official `export_to_c` + `cute.runtime.load_module`, fully self-controlled manifest | ✅ (compile and load verified; launch needs a driver no older than cuda-python's CUDA version) |
| **M5** | Precompiled package (AOT distribution) | `pip install opforge-prebuilt` gives a hot install | ⏸ **explicitly deferred**, see below |

### On the deferral decision for M5 (precompiled package)

**Deferred, not forgotten.** Current state: `tools/build_aot.py` can populate compiled artifacts into two directories
(Triton's `TRITON_CACHE_DIR`, opforge's own `OPFORGE_CACHE_DIR`),
**cross-process no-recompile already has test coverage**; all that's missing is "turning that directory into an installable `.whl`".

**Why it isn't urgent**:
1. **The problem it solves only hurts when there are "many new machines / new containers"** — for a single machine, or an image that `COPY`s the directory once, it's already enough;
2. The measured cost is small: one operator's cold start is **8.8s** (compilation), and the artifact is only **868KB**;
3. Doing it cleanly is awkward: **Triton has no export API**, so all we can do is relocate its own cache directory — which amounts to treating Triton's
   internal layout (hashed directory names, file naming, `.json` format) as an interface, liable to break on any upgrade.

**The CuTeDSL half is already done** — the artifacts of `cute.compile` have an official `export_to_c`,
loading uses `cute.runtime.load_module`, and the manifest is fully self-controlled: the artifact is
`<OPFORGE_CACHE_DIR>/v1/<target_fp>/<ns>/<source_fp>/<entry_id>.artifact`.
So what remains of M5 is really only the Triton half, and that is precisely the "awkward to do cleanly" half:
one option is "relocate the cache + use the triton version / RECORD digest in `source_fingerprint` as a rejection check".

Alternative (the currently recommended usage): point each backend's cache directory at a persistent location, and let the deployment side's image/mount handle it —
this is exactly what vLLM itself does (it proactively redirects `TRITON_CACHE_DIR` to its own compile-cache root, making the whole thing easy to transport).

---

## 7. Risks and Trade-offs

| Risk | Explanation | Countermeasure |
|---|---|---|
| Wrong cache-fingerprint design | Silently misuses artifacts (worse than crashing) | Two-level fingerprint + strict manifest verification + "missing manifest → error out" |
| Over-lowering to C++ | Delivery burden explodes (requires an nvcc toolchain) | Strict Category A/B split; for Category B prefer DSL idioms |
| Nobody uses the contract model | It becomes a dead letter | Contracts go into the cache key + into CI gates; if unused, error out |
| The shared surface between training/inference is overestimated | In reality only the trunk can be shared | Build the trunk first (norm/activation/quant/GEMM/MoE); keep attention/optimizer explicitly separate |
| Runtime compilation is uncontrollable | Production jitter | Seal-compilation switch + precompiled package |

---

## Appendix A · One-line criterion (deciding which layer an operator belongs in)

> **If it needs any of "framework runtime state / compiler graph / framework data-layout conventions / special numeric contract", it must stay at L3 (the adapter layer).**
> Anything that is pure `tensor → tensor` goes down to L1.
