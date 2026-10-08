# Contributing Guide

Start with [`docs/design.md`](docs/design.md) (why the project is designed this way)
and [`docs/architecture.md`](docs/architecture.md) (interfaces and invariants).
**Before touching L0-L2, be sure to read through the architecture invariants I1-I9**
— CI enforces them.

## Development environment

**Python 3.12 is recommended** (matches production). The project declares
`requires-python = ">=3.10"`, and CI covers 3.10 / 3.11 / 3.12.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"          # core + numpy + pytest/ruff
```

### GPU dependencies (torch / triton / cutedsl)

```bash
uv pip install --python .venv/bin/python \
  "torch==2.9.0+cu128" triton nvidia-cutlass-dsl
```

**Two known pitfalls**:

1. **You must pin the CUDA variant explicitly (`+cu128`).** A plain
   `pip install torch` may resolve to a newer
   CUDA build (e.g. CUDA 13), which fails on an older driver with
   `cudaErrorInsufficientDriver`.
   First check which CUDA version your driver supports via `nvidia-smi`, then pick the
   matching `+cuXXX`.
2. **`triton` / `nvidia-cutlass-dsl` have Python ABI constraints.** If the index you
   use only carries wheels for some ABIs (e.g. only `cp310`), resolution fails outright
   on 3.12 — switch to an index that has the matching ABI.

**On a machine with a GPU**, before running Triton/CuTeDSL you also need:

```bash
source tools/gpu_env.sh
```

It sets two environment variables (`LD_PRELOAD` for the real libcuda +
`PYTHONNOUSERSITE`), both of which are container-environment quirks, not a design choice
of this project — see the "GPU environment" section of the README for the reasoning.

## Three commands

```bash
python3 tools/run_tests.py             # tests (pytest style; if pytest is installed you can also just run pytest)
python3 tools/check_invariants.py      # architecture invariants I1-I9
TRITON_CACHE_DIR=/tmp/k python3 tools/build_aot.py   # AOT precompilation
```

`tools/run_tests.py` exists for **environments without pytest** (a zero-dependency core
layer is one of the design goals). Conventions:

* Tests are plain `test_*` functions inside a module;
* A test that needs to be skipped raises `unittest.SkipTest` (both runners honor it);
* Helpers go in `tests/support.py` (files whose name starts with `_` or does not start
  with `test_` are not treated as test cases).

## Packaging (locally, no CI needed)

```bash
# Option one: use build (recommended; produces both sdist and wheel)
python3 -m pip install build
python3 -m build --wheel            # artifacts land in dist/

# Option two: pip only
python3 -m pip wheel . -w dist --no-deps
```

The artifact looks like `dist/opforge-0.0.1.dev0-py3-none-any.whl` — the
**`py3-none-any`** tag means it is a
pure-Python package with no platform tag (the core layer and both DSL backends import
lazily, so they do not bind to torch/triton).

Install into a clean environment to verify:

```bash
python3 -m venv /tmp/v && /tmp/v/bin/pip install dist/opforge-*.whl
/tmp/v/bin/python -c "import opforge; print(opforge.__version__)"

# For the L3 adapter layer, add the extras:
/tmp/v/bin/pip install "opforge[training,inference] @ file://$PWD/dist/opforge-0.0.1.dev0-py3-none-any.whl"
# Only on the target machine: opforge[torch,triton,cutedsl]
```

The version number comes from `version = "0.0.1.dev0"` in `pyproject.toml`. If we later
adopt `setuptools-git-versioning`, it will be derived automatically from git tags.

**AOT compilation artifacts are not built here.** Operator compilation artifacts land in
their respective cache directories
(`TRITON_CACHE_DIR` for Triton, `OPFORGE_CACHE_DIR` for CuTeDSL), and you must run
`tools/build_aot.py` **on the same architecture you deploy to** — see the
"AOT precompilation" section of the README.

## Architecture conventions

| Layer | What you can do | What you **cannot** do |
|---|---|---|
| L0 `lang/` | Backend protocol, escape-hatch registration | Module-level `import triton` / `cutlass` / `torch` |
| L1 `kernel/` | Operator declaration and call pipeline | Import any training/inference framework |
| L2 `contract/` `cache/` `launch/` | Contract, fingerprint, cache, launch | Depend on L3 in reverse |
| L3 `training/` `inference/` | Framework integration | — |

The core layer (L0-L2) has **zero module-level dependencies**: triton / cutlass / torch
may only be imported lazily.
This is not fastidiousness — it guarantees that contract resolution, fingerprints, and
caching can all be fully tested in CPU-only CI.

## Adding an operator

1. Put it under `src/opforge/kernels/<domain>/` (`norm/`, `elementwise/`, …);
2. **Write the reference first** (pure PyTorch ground truth) — it is CI's accuracy
   baseline and also the fallback path on the training side.
   A missing reference is rejected by invariant I9;
3. Declare it with `@kernel(...)`. **`contract` is a required parameter with no
   default** — choose it carefully;
4. If you need other contract tiers, add `variants={Contract.FAST: ...}`;
5. For a GPU path: use `declare_triton(impl, kernel, prewarm=...)` for Triton,
   `declare_cutedsl(impl, examples=...)` for CuTeDSL, and point `backend=` at it.
   **CuTeDSL host functions must carry type annotations** (`x: cute.Tensor`,
   `alpha: cutlass.Float32`)
   and sample inputs must be wrapped into a `cute.Tensor` with `from_dlpack` — the
   cross-process reuse path goes through `export_to_c`,
   which generates a C ABI from the parameter types, and a bare `torch.Tensor` will not
   make it through (see `kernels/elementwise/scale.py` for details);
6. Add tests: at least one accuracy assertion against the reference; if you can, add
   dtype / edge-shape / CUDA Graph coverage;
7. `python3 tools/run_tests.py && python3 tools/check_invariants.py`.

## Adding a backend

Implement the three methods of the `Backend` protocol, then call
`register_backend(...)` — **no changes to the core are needed**:

```python
class MyBackend:
    name = "my-backend"
    def supports(self, spec, impl) -> bool: ...         # impl = the implementation being built, don't just look at spec
    def build(self, spec, impl, contract): ...          # the artifact must provide run(args, kwargs)
    def load(self, spec, key, path): ...                # return None if disk loading is unsupported

register_backend(MyBackend())
```

**`supports` must inspect `impl`, not `spec.default_impl`** — different contract
variants of the same operator
can land on different backends, and what a backend should check is "the marker on this
impl" (e.g. the Triton backend looks at
`__opforge_triton_kernels__`). Likewise, the `__opforge_backend__`
marker that `declare_*` stamps on an implementation is a **soft** declaration: when that
backend is unavailable on the current platform, it degrades to another backend.

**To make an artifact reusable across processes**, give the artifact class an
`export(self, path, key)` — the cache layer will see it and
write it to disk along with a manifest, and on load it validates the manifest
(fingerprint + SHA256) before calling your `load`. If you do not implement it
(as `python` / `triton` / `cpp` do not), it degrades to in-process caching, with all
other semantics identical.
`key.export_symbol` is the sole lookup key on load, so export and load must use the same
name —
the CuTeDSL backend is the best model to follow
(`export_to_c(function_prefix=key.export_symbol)` ↔ `module[key.export_symbol]`).

`tests/test_cutedsl.py` contains a complete "third-party backend" test you can model after.

### Built-in backends

| Backend | When to use it |
|---|---|
| `triton` | Regular block-structured operators (preferred) |
| `cutedsl` | When you need layout / TMA / warp-level control |
| `cpp` | **Escape hatch**: when a DSL lacks a primitive. Must be registered via `@cpp_kernel(reason=...)`, and only `MISSING_PRIMITIVE` / `PERF_GAP` are accepted |
| `python` | Reference implementation / CPU fallback; also the model for a "third-party backend" |

With `backend="auto"`, backend selection resolves in three tiers: the backend declared by
the implementation itself (soft) → `spec.backend` (hard)
→ `PROBE_ORDER` probing (`cutedsl → triton → cpp → python`); an operator that declares no
DSL kernel at all falls back to `python`. Reference examples: `kernels/norm/rms_norm.py`
(Triton),
`kernels/elementwise/scale.py` (CuTeDSL), `kernels/reduction/warp_reduce.py` (cpp escape
hatch).
Note that `PROBE_ORDER` is fixed (the `auto` probe order): a new backend can by default
only be selected **explicitly** —
this avoids the non-deterministic "whoever registers first wins" behavior.

## Two easy-to-trip pitfalls

* **Precompilation must be done on the target architecture**: `build_aot.py` reports
  `skip:` on a machine with no GPU
  and returns non-zero. Cross-architecture compilation artifacts cannot be reused.
* **Never put runtime dynamic dimensions into cache keys or launch signatures**: shapes
  may go into the launch signature, but
  **values such as token count / sequence length must not go into cache keys**, or the
  cache explodes and the online path recompiles over and over.
  See invariant I5.

## Committing

* Small steps, single purpose;
* Before committing, get `tools/run_tests.py` and `tools/check_invariants.py` passing;
* Commit messages should state **why**, not just **what changed**.

## License

By contributing you agree to license your work under Apache-2.0 (see `LICENSE`).
