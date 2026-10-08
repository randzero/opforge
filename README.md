# opforge

> **The operator forge** — write kernels once, ship them to both training and inference.

[English](README.md#opforge) · [中文](README.md#中文)

An open-source **operator/kernel framework** built on one thesis:

- **Write JIT** — kernels are authored in a Python DSL (Triton / CuTeDSL): fast to iterate,
  low barrier to entry, friendly to AI assistance.
- **Ship AOT** — compiled artifacts are written to disk ahead of time and distributed with
  the package: production does no runtime compilation, so cold start is predictable.
- **One kernel, two uses** — the same compute kernel serves both training and inference.
  What genuinely cannot be shared (autograd orchestration, CUDA Graph, paged KV) stays in
  each side's adapter layer.
- **The numeric contract is a first-class citizen** — `DETERMINISTIC` / `HIGH_PRECISION` /
  `FAST` are declared by the operator and chosen by the caller, instead of `bitwise=True`
  being threaded through every call site.

> Status: **skeleton + four backends (python / triton / cutedsl / cpp escape hatch) + AOT
> precompilation** — L0–L3 interfaces, three real DSL operators, 109 tests, 9 architecture
> invariants; green on both CPU-only and GPU configurations.
> CuTeDSL **cross-process artifact reuse is implemented** (official `export_to_c` +
> `cute.runtime.load_module`); the compile half is verified here, but the launch half needs a
> driver no older than cuda-python's CUDA version (see [GPU environment notes](#gpu-environment-notes)).
> The precompiled-wheel package (`opforge-prebuilt`) is **explicitly deferred** — see
> [`docs/design.md`](docs/design.md) §6 for the reasoning and the alternatives.

---

## Why this exists

Training-side and inference-side operator libraries are two separate worlds today: the
training side (Megatron / DeepSpeed / …) and the inference side (vLLM / SGLang /
TensorRT-LLM) each write their own. But **most of the math is identical** — norm,
activation, quantization, GEMM, the MoE backbone. The differences are not in "how to
compute", but in:

| Difference | Training | Inference |
|---|---|---|
| Needs backward | Yes | No |
| Numeric requirement | Bitwise-reproducible / high-precision accumulation | Latency first, can relax |
| Shape distribution | Fixed batch, long sequences | Dynamic batch, decode |
| Execution form | autograd + optimizer | CUDA Graph + paged KV |
| Delivery | Source + fast iteration | **Precompiled artifact + cold start** |

`opforge`'s claim is to handle these differences by **layering**: push the shared parts
down, keep the unshared parts apart. See [`docs/design.md`](docs/design.md).

## Layering

```
L0 language layer   backend protocol (triton / cutedsl / cpp escape hatch / python reference)
L1 kernel layer     operator declaration (@kernel) and the call pipeline   ┐
L2 contract layer   numeric contracts, variant resolution, preconditions   │ shared
L2 cache layer      two-level fingerprints, manifest, sealed compilation   │ by both
L2 launch layer     per-signature caching of the launch closure            ┘
L3 adapter layer    training-side to_autograd / inference-side to_custom_op ← separate
```

**L0–L2 have zero module-level dependencies** — contract resolution, fingerprints, caching
and launching can all be run and tested without any framework or GPU.

---

## Getting started

### 1. Requirements

- **Python ≥ 3.10** (3.12 recommended; CI covers 3.10 / 3.11 / 3.12).
- That is all for the core: `numpy` is the only runtime dependency and is installed for you.
- `torch` is only needed by the L3 adapter layer; `triton` / `nvidia-cutlass-dsl` are only
  needed by the GPU backends.

### 2. Install

The project is **not published on PyPI yet**, so install from a checkout:

```bash
git clone <repo-url> opforge && cd opforge

# Core only: contract layer + cache + launch + the pure-Python reference backend.
pip install -e .

# If you also want the L3 adapter layer (autograd / custom-op integration):
pip install -e ".[training,inference]"

# If you want the GPU backends (compile Triton / CuTeDSL kernels):
pip install -e ".[torch,triton,cutedsl]"

# Or everything:
pip install -e ".[all]"
```

### 3. Write an operator

An operator is a decorated function. You must supply a `contract`, and you should supply a
`reference` (pure-PyTorch ground truth) — the reference is the CI accuracy baseline, the
training-side fallback path, and (with `backward="auto"`) the source of the backward pass.

```python
from opforge import Contract, RequiresDtypeIn, kernel


def rms_norm_fast(x, weight, eps=1e-6, *, zero_centered=False):
    """FAST tier: accumulate in the input dtype — faster, less accurate."""
    ...


@kernel(
    name="rms_norm",
    contract=Contract.HIGH_PRECISION,       # required: the default tier
    variants={Contract.FAST: rms_norm_fast},  # optional: the other tiers
    preconditions=(RequiresDtypeIn("float32", "float16", "bfloat16"),),
    backward="auto",
)
def rms_norm(x, weight, eps=1e-6, *, zero_centered=False):
    """y = x / sqrt(mean(x**2) + eps) * weight"""
    ...


def rms_norm_ref(x, weight, eps=1e-6, *, zero_centered=False):
    """Pure-PyTorch ground truth.

    Found automatically by the `<operator name>_ref` naming convention, so you never have
    to pass it explicitly.
    """
    ...
```

To make it run on a GPU you additionally declare which backend builds the implementation
(`declare_triton` / `declare_cutedsl` / `declare_cpp`) — see
[`CONTRIBUTING.md`](CONTRIBUTING.md) §"Adding an operator" and the three built-in examples
`kernels/norm/rms_norm.py` (Triton), `kernels/elementwise/scale.py` (CuTeDSL) and
`kernels/reduction/warp_reduce.py` (the C++ escape hatch).

### 4. Call it

```python
from opforge import Contract
from opforge.kernels.norm import rms_norm

out = rms_norm(x, w, contract=Contract.HIGH_PRECISION)
```

The call pipeline is: **resolve the contract → check preconditions → cache hit or build →
launch**. Two failure modes are deliberately distinct:

- the contract cannot be satisfied → raises `ContractUnsatisfied` (**hard failure**;
  carrying on would silently change the numeric semantics);
- a precondition is unmet (no CUDA Graph capture, ranks not in lockstep, wrong dtype …) →
  returns `Unavailable` (**normal fallback**, not an exception, so the caller can route
  elsewhere).

### 5. Choosing a numeric contract

Contracts are ordered, and fallback is **only allowed toward a stricter tier**:

| Requested | Implementations `{DET, FAST}` | Implementations `{FAST}` only |
|---|---|---|
| `DETERMINISTIC` | uses `DET` | **raises `ContractUnsatisfied`** |
| `HIGH_PRECISION` | uses `DET` (stricter, acceptable) | **raises `ContractUnsatisfied`** |
| `FAST` | uses `FAST` | uses `FAST` |

Every fallback is counted, so it can be surfaced as a metric rather than staying invisible:

```python
from opforge import fallback_counts

print(fallback_counts())   # {("rms_norm", "DETERMINISTIC", "HIGH_PRECISION"): 3, ...}
```

### 6. Run the tests

```bash
# Tests (bundled stdlib runner; if pytest is installed, plain `pytest` works too)
PYTHONPATH=src python3 tools/run_tests.py

# Architecture invariants I1–I9
PYTHONPATH=src python3 tools/check_invariants.py
```

Both are also what CI runs. The test runner needs no pytest on purpose: a zero-dependency
core layer is one of the design goals, so the core must be testable in a bare environment.

---

## AOT precompilation

The second half of "write JIT, ship AOT". Operators compile the shapes they will need at
**build time** and write them to disk; runtime only loads.

```bash
# Build time (on the same architecture you deploy to) — mount both dirs:
TRITON_CACHE_DIR=/opt/opforge-triton \
OPFORGE_CACHE_DIR=/opt/opforge-kernels \
python3 tools/build_aot.py

# Runtime: mount the same two dirs — installed means warm start.
export TRITON_CACHE_DIR=/opt/opforge-triton
export OPFORGE_CACHE_DIR=/opt/opforge-kernels
```

The two directories have different jobs: **Triton artifacts can only live in Triton's own
cache** (it has no export API), whereas **CuTeDSL artifacts are managed by opforge itself**
(the object file from `export_to_c` plus a manifest, at
`<OPFORGE_CACHE_DIR>/v1/<target_fp>/<ns>/<source_fp>/<entry_id>.artifact`).
Running the build twice therefore shows directly that the second run recompiled nothing:

```
3 kernels, 3 compiled, 0 loaded, ok=True          # cold: everything compiles
  rms_norm         triton   HIGH_PRECISION   ok
  scale            cutedsl  HIGH_PRECISION   ok
  warp_reduce_sum  cpp      DETERMINISTIC    ok

3 kernels, 2 compiled, 1 loaded, ok=True          # run again: CuTeDSL hits the disk
  rms_norm         triton   HIGH_PRECISION   ok
  scale            cutedsl  HIGH_PRECISION   loaded
  warp_reduce_sum  cpp      DETERMINISTIC    ok
```

**An operator must declare which shapes to compile**, otherwise the build step has nothing
to work from: Triton uses `declare_triton(..., prewarm=...)`, CuTeDSL uses
`declare_cutedsl(..., examples=...)` (for CuTeDSL the example inputs are a required
argument of `cute.compile`, not an optional optimization).

At runtime, combine it with `KernelCache.seal()` to assert "nothing compiles for the rest
of the process lifetime" — after sealing, any build that misses raises `CompilationSealed`,
so "the second run passed" is itself proof that nothing was recompiled.

## Packaging

```bash
python3 -m build --wheel          # or: python3 -m pip wheel . -w dist --no-deps
# artifact: dist/opforge-0.0.1.dev0-py3-none-any.whl
```

## GPU environment notes

On a machine with a GPU, `source tools/gpu_env.sh` before running the Triton path. It does
two things, both **container-environment quirks rather than design choices of this
project**:

1. **`LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.1`** — in the container
   `/usr/lib/x86_64-linux-gnu/libcuda.so` points at a zero-byte stub and the real driver
   lives in `/usr/local/nvidia/lib64/`. torch loads it with ctypes' default `RTLD_LOCAL`,
   so the symbols never enter the global scope, and Triton's driver extension then fails
   with `undefined symbol: cuModuleGetFunction` — which also breaks `torch.compile`'s
   inductor backend.
2. **`PYTHONNOUSERSITE=1`** — skips a broken torch under `~/.local` that would otherwise
   shadow the working one.

**CuTeDSL has one more constraint**: compilation only needs the toolchain
(`CUTE_DSL_ARCH=sm_90a` must be set explicitly), but **execution** requires the driver
to be no older than cuda-python's CUDA version. When it is older, launching raises
`cudaErrorInsufficientDriver` — so on such a host CuTeDSL's **AOT compile + cross-process
load half still works while the launch half cannot run**; the corresponding test skips
itself.

Note that verifying cross-process reuse does not depend on being able to launch: the
producing process writes the artifact with `export_to_c`, and the consuming process loads
it into a **sealed** cache (`KernelCache.seal()`) — getting the artifact back proves there
was no fallback to `cute.compile`, which would have raised `CompilationSealed`.

## Repository layout

```
docs/
  design.md          why: trade-offs and motivations
  architecture.md    how: layered interfaces and invariants
src/opforge/
  contract/          numeric contracts, variant resolution, preconditions
  kernel/            @kernel decorator, KernelSpec, call pipeline
  cache/             two-level fingerprints, manifest, sealed compilation
  launch/            per-signature caching of the launch closure
  lang/              backend protocol (python / triton / cutedsl / cpp), escape-hatch registration
  kernels/           operator implementations (norm/, elementwise/, reduction/)
  training/          L3 training-side adapter
  inference/         L3 inference-side adapter
  aot.py             AOT precompile entry point
tests/               accuracy, contract, cache, backend, AOT, adapter tests
tools/               test runner, invariant checker, AOT build, GPU env script
.github/workflows/   CI (CPU) and the GPU job
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/design.md`](docs/design.md) | **Why**: layered architecture, the numeric contract model, escape-hatch policy, compile cache and fingerprints, delivery forms, CI quality gates, roadmap |
| [`docs/architecture.md`](docs/architecture.md) | **How**: the L0–L4 interfaces — the full `@kernel` signature, the `Contract` enum and variant derivation, the `Precondition` protocol, the cache/launch layers, architecture invariants I1–I9 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Layering conventions, how to add an operator / a backend, and the two easy-to-trip pitfalls |

## License

Apache-2.0 (see [`LICENSE`](LICENSE)).

---
---

# 中文

> **算子锻造坊** —— 一次写内核，同时发到训练与推理两侧。

[English](README.md#opforge) · [中文](README.md#中文)

一个开源**算子/内核框架**，主张只有一条：

- **JIT 写** —— 内核用 Python DSL（Triton / CuTeDSL）编写：迭代快、门槛低、易被 AI 辅助。
- **AOT 发** —— 编译产物提前落盘并随包分发：生产环境不做运行期编译，冷启动可预测。
- **一套内核，两用** —— 同一份计算内核同时服务训练与推理；真正不共用的部分
  （autograd 编排、CUDA Graph、paged KV）留在各自的适配层。
- **数值契约是一等公民** —— `DETERMINISTIC` / `HIGH_PRECISION` / `FAST` 由算子声明、
  调用方按场景选择，而不是把 `bitwise=True` 一路传遍所有调用点。

> 状态：**骨架 + 四个后端（python / triton / cutedsl / cpp 逃生口）+ AOT 预编译** ——
> L0–L3 接口、三个真 DSL 算子、109 项测试、9 条架构不变量；CPU-only 与 GPU 两种配置都跑绿。
> CuTeDSL 的**跨进程产物复用已实现**（官方 `export_to_c` + `cute.runtime.load_module`）：
> 编译那半已验证，launch 那半要求驱动不旧于 cuda-python 的 CUDA 版本才能跑（见 [GPU 环境](#gpu-环境)）。
> 预编译包（`opforge-prebuilt`）**明确推迟** —— 理由与替代方案见
> [`docs/design.md`](docs/design.md) §6。

---

## 为什么会有这个项目

训练侧与推理侧的算子库今天基本是两套：训练侧（Megatron / DeepSpeed 等）和推理侧
（vLLM / SGLang / TensorRT-LLM）各写各的。但**大部分算子的数学是一样的** —— norm、
激活、量化、GEMM、MoE 主干。差别不在"怎么算"，而在：

| 差异 | 训练 | 推理 |
|---|---|---|
| 需要 backward | 是 | 否 |
| 数值要求 | 逐位可复现 / 高精度累加 | 延迟优先，可放开 |
| shape 分布 | 固定 batch、长序列 | 动态 batch、decode |
| 执行形态 | autograd + 优化器 | CUDA Graph + paged KV |
| 交付 | 源码 + 快速迭代 | **预编译产物 + 冷启动** |

`opforge` 的主张是把这些差异**分层处理**：共用的下沉、不共用的分开。
详见 [`docs/design.md`](docs/design.md)。

## 分层

```
L0 语言层   后端协议（triton / cutedsl / cpp 逃生口 / python 参考后端）
L1 内核层   算子声明（@kernel）与调用流水线        ┐
L2 契约层   数值契约、变体解析、前置条件           │ 两侧共用
L2 缓存层   两级指纹、manifest、封编译             │
L2 启动层   按签名缓存启动闭包                     ┘
L3 适配层   训练侧 to_autograd / 推理侧 to_custom_op  ← 两侧分开
```

**L0–L2 的模块级依赖为零** —— 契约解析、指纹、缓存、启动都能脱离任何框架与 GPU 运行与测试。

---

## 快速开始

### 1. 环境要求

- **Python ≥ 3.10**（推荐 3.12；CI 覆盖 3.10 / 3.11 / 3.12）。
- 核心层只需要这些：`numpy` 是唯一的运行期依赖，会随安装自动装上。
- `torch` 只有 L3 适配层需要；`triton` / `nvidia-cutlass-dsl` 只有 GPU 后端需要。

### 2. 安装

本项目**尚未发布到 PyPI**，请从源码检出安装：

```bash
git clone <repo-url> opforge && cd opforge

# 只装核心：契约层 + 缓存 + 启动 + 纯 Python 参考后端
pip install -e .

# 需要 L3 适配层（autograd / custom op 集成）时：
pip install -e ".[training,inference]"

# 需要 GPU 后端（编译 Triton / CuTeDSL 内核）时：
pip install -e ".[torch,triton,cutedsl]"

# 或者全装：
pip install -e ".[all]"
```

### 3. 写一个算子

一个算子就是一个被装饰的函数。`contract` 是**必填**的；同时应该给出 `reference`
（纯 PyTorch 真值）—— 它既是 CI 的精度基准，也是训练侧的回退路径，配合
`backward="auto"` 还是反向实现的来源。

```python
from opforge import Contract, RequiresDtypeIn, kernel


def rms_norm_fast(x, weight, eps=1e-6, *, zero_centered=False):
    """FAST 档：在输入 dtype 上累加 —— 更快、更不精确。"""
    ...


@kernel(
    name="rms_norm",
    contract=Contract.HIGH_PRECISION,        # 必填：默认档位
    variants={Contract.FAST: rms_norm_fast}, # 可选：其它档位
    preconditions=(RequiresDtypeIn("float32", "float16", "bfloat16"),),
    backward="auto",
)
def rms_norm(x, weight, eps=1e-6, *, zero_centered=False):
    """y = x / sqrt(mean(x**2) + eps) * weight"""
    ...


def rms_norm_ref(x, weight, eps=1e-6, *, zero_centered=False):
    """纯 PyTorch 真值。

    按 `<算子名>_ref` 的命名约定自动找到，不需要显式传入。
    """
    ...
```

要让它跑在 GPU 上，还要声明由哪个后端构建这个实现
（`declare_triton` / `declare_cutedsl` / `declare_cpp`）—— 见
[`CONTRIBUTING.md`](CONTRIBUTING.md) 的「加一个算子」一节，以及三个内置范例：
`kernels/norm/rms_norm.py`（Triton）、`kernels/elementwise/scale.py`（CuTeDSL）、
`kernels/reduction/warp_reduce.py`（C++ 逃生口）。

### 4. 调用

```python
from opforge import Contract
from opforge.kernels.norm import rms_norm

out = rms_norm(x, w, contract=Contract.HIGH_PRECISION)
```

调用流水线是：**解析契约 → 检查前置条件 → 缓存命中或构建 → 启动**。
两种失败刻意区分开：

- 契约无法满足 → 抛 `ContractUnsatisfied`（**硬失败**：继续执行等于悄悄改变数值语义）；
- 前置条件不满足（在 CUDA Graph 捕获期、多 rank 未步调一致、dtype 不符……）→
  返回 `Unavailable`（**正常回退**，不是异常，好让调用方换路走）。

### 5. 选择数值契约

契约是有序的，回退**只允许朝更严的方向**：

| 请求 | 有 `{DET, FAST}` | 只有 `{FAST}` |
|---|---|---|
| `DETERMINISTIC` | 用 `DET` | **抛 `ContractUnsatisfied`** |
| `HIGH_PRECISION` | 用 `DET`（更严，可接受） | **抛 `ContractUnsatisfied`** |
| `FAST` | 用 `FAST` | 用 `FAST` |

每次回退都会被计数，因此可以当成指标暴露出来，而不是让它无声无息：

```python
from opforge import fallback_counts

print(fallback_counts())   # {("rms_norm", "DETERMINISTIC", "HIGH_PRECISION"): 3, ...}
```

### 6. 跑测试

```bash
# 测试（自带 stdlib 运行器；装了 pytest 也可以直接 pytest）
PYTHONPATH=src python3 tools/run_tests.py

# 架构不变量 I1–I9
PYTHONPATH=src python3 tools/check_invariants.py
```

这两条也是 CI 跑的。测试运行器刻意不依赖 pytest：核心层零依赖是本项目的设计目标之一，
所以核心必须能在裸环境里被测。

---

## AOT 预编译

"JIT 写、AOT 发"的后半句。算子在 **build 期**把要用的形状编好并落盘，运行期只加载：

```bash
# 构建期（在与部署相同的架构上）—— 两个目录都值得挂：
TRITON_CACHE_DIR=/opt/opforge-triton \
OPFORGE_CACHE_DIR=/opt/opforge-kernels \
python3 tools/build_aot.py

# 运行期把同一个目录挂进去 —— 装机即热启动
export TRITON_CACHE_DIR=/opt/opforge-triton
export OPFORGE_CACHE_DIR=/opt/opforge-kernels
```

两个目录分工不同：**Triton 的产物只能住在它自己的缓存里**（它没有 export API），
**CuTeDSL 的产物由 opforge 自己管**（`export_to_c` 出的 object + manifest，路径
`<OPFORGE_CACHE_DIR>/v1/<target_fp>/<ns>/<source_fp>/<entry_id>.artifact`）。
于是预编译跑两遍时，报告能直接体现"第二遍没有再编译"：

```
3 kernels, 3 compiled, 0 loaded, ok=True          # 冷启动：全编
  rms_norm         triton   HIGH_PRECISION   ok
  scale            cutedsl  HIGH_PRECISION   ok
  warp_reduce_sum  cpp      DETERMINISTIC    ok

3 kernels, 2 compiled, 1 loaded, ok=True          # 再跑一遍：CuTeDSL 命中磁盘
  rms_norm         triton   HIGH_PRECISION   ok
  scale            cutedsl  HIGH_PRECISION   loaded
  warp_reduce_sum  cpp      DETERMINISTIC    ok
```

**算子必须声明要编哪些形状**，否则构建期无从下手：Triton 用
`declare_triton(..., prewarm=...)`，CuTeDSL 用 `declare_cutedsl(..., examples=...)`
（后者的样例输入是 `cute.compile` 的必需参数，不是可选优化）。

运行期想断言"全程不再编译"，配合 `KernelCache.seal()` 即可 —— 封编译之后任何未命中的
构建都会抛 `CompilationSealed`，所以"第二遍能跑通"本身就证明了没重编译。

## 打包

```bash
python3 -m build --wheel          # 或 python3 -m pip wheel . -w dist --no-deps
# 产物：dist/opforge-0.0.1.dev0-py3-none-any.whl
```

## GPU 环境

在有 GPU 的机器上跑 Triton 路径前，先 `source tools/gpu_env.sh`。它做两件事，都是
**容器环境的坑，不是本项目的设计**：

1. **`LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.1`** —— 容器里
   `/usr/lib/x86_64-linux-gnu/libcuda.so` 指向一个 0 字节桩，真驱动在
   `/usr/local/nvidia/lib64/`。torch 用 ctypes 默认的 `RTLD_LOCAL` 加载它，符号不进
   全局作用域，于是 Triton 的 driver 扩展加载时报 `undefined symbol:
   cuModuleGetFunction`，**连 `torch.compile` 的 inductor 后端也一起用不了**。
2. **`PYTHONNOUSERSITE=1`** —— 避开 `~/.local` 下一份损坏的 torch，它会遮蔽正常的版本。

**CuTeDSL 另有一条限制**：编译只要求工具链（`CUTE_DSL_ARCH=sm_90a` 需显式指定），
但**执行**要求驱动版本不旧于 cuda-python 的 CUDA 版本；驱动更旧时 launch 会报
`cudaErrorInsufficientDriver` —— 所以在那样的机器上，CuTeDSL 的
**AOT 编译 + 跨进程加载那半照常可用，只有 launch 跑不了**，对应测试自动 skip。

跨进程复用的验证方式与"能不能 launch"无关：构建进程用 `export_to_c` 落盘，消费进程用
**已封编译**（`KernelCache.seal()`）的缓存去加载 —— 能拿回产物就证明没有回退到
`cute.compile`（否则会抛 `CompilationSealed`）。

## 目录

```
docs/
  design.md          为什么：取舍与动机
  architecture.md    怎么做：分层接口与不变量
src/opforge/
  contract/          数值契约、变体解析、前置条件
  kernel/            @kernel 装饰器、KernelSpec、调用流水线
  cache/             两级指纹、manifest、封编译
  launch/            按签名缓存启动闭包
  lang/              后端协议（python / triton / cutedsl / cpp）、逃生口登记
  kernels/           算子实现（norm/、elementwise/、reduction/）
  training/          L3 训练侧适配
  inference/         L3 推理侧适配
  aot.py             AOT 预编译入口
tests/               精度、契约、缓存、后端、AOT、适配层
tools/               测试运行器、不变量检查、AOT 构建、GPU 环境脚本
.github/workflows/   CI（CPU）与 GPU 作业
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/design.md`](docs/design.md) | **为什么**：分层架构、数值契约模型、逃生口策略、编译缓存与指纹、交付形态、CI 质量门、路线图 |
| [`docs/architecture.md`](docs/architecture.md) | **怎么做**：L0–L4 各层接口 —— `@kernel` 完整签名、`Contract` 枚举与变体派生、`Precondition` 协议、缓存/启动层、架构不变量 I1–I9 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 分层约定、如何加算子 / 加后端、两个容易踩的坑 |

## 许可证

Apache-2.0（见 [`LICENSE`](LICENSE)）。
