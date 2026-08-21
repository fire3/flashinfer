# FlashInfer SM89 开发 / 构建脚本

为 DeepSeek V4 稀疏 MLA 的 SM89 (Ada) 移植准备的开发与构建脚本集，组织方式
参考 vLLM 仓库的 `scripts/build/`。本 fork 的 SM89 构建全部以
`FLASHINFER_CUDA_ARCH_LIST=8.9` 为目标。

## 工作流

### 一次性：安装 conda 环境依赖

```bash
# 使用当前已激活的 conda 环境
bash scripts/build/sm89_install_deps.sh
# 或指定环境名（不存在则创建，python 3.12）
ENV_NAME=flashinfer-sm89 bash scripts/build/sm89_install_deps.sh
```

该脚本安装：build/ninja/setuptools/packaging/apache-tvm-ffi（build backend
的 `--no-build-isolation` 依赖）、cu130 索引的 torch、`requirements.txt`
其余运行时依赖（torch 行已剥离，避免被解析成 CPU wheel）。moe_ep
（NIXL/NCCL-EP）默认关闭（`BUILD_NIXL_EP=0`），SM89 稀疏 MLA 不需要。

### 每次切换分支 / 拉取上游后：准备三方依赖

```bash
bash scripts/build/sm89_prepare_deps.sh          # cutlass / spdlog / cccl
bash scripts/build/sm89_prepare_deps.sh --all    # 另含 3rdparty/nixl（EP）
bash scripts/build/sm89_prepare_deps.sh --check  # 只校验不修改
```

FlashInfer 的三方依赖版本以 git submodule 提交锁定在 index 中（参见
`git submodule status`），等价于 vLLM 的 `prepare_deps.sh` 按 CMake pin
同步 `.deps`。editable 安装下 `flashinfer/data/cccl` 由 pyproject 的
package-dir 映射到 `3rdparty/cccl`，submodule 就绪后 JIT include 路径即
可用。

### 开发模式：editable 安装

```bash
bash scripts/build/sm89_build_wheel.sh --editable
```

等价于 `pip install -e . --no-build-isolation`（Python 改动即时生效；
C++/CUDA 改动需重新执行）。运行时 JIT 还需要：

- `nvcc` 在 `PATH`（`sm89_env.sh` 会自动从 pip `nvidia-cuda-nvcc` 布局
  发现）；
- 按需 `LD_LIBRARY_PATH` 指向 conda 环境 lib（避免 ICU 等第三方库版本
  冲突）；
- `FLASHINFER_DISABLE_VERSION_CHECK=1`。

### 构建分发 wheel

```bash
bash scripts/build/sm89_build_wheel.sh                    # AOT wheel
bash scripts/build/sm89_build_jit_cache_wheel.sh          # JIT-cache 平台 wheel
```

产物带 `+sm89` local version 标记并经过校验：

- AOT：`dist/flashinfer_python-0.6.16.post3+sm89-*.whl`
- JIT-cache：`dist/flashinfer_jit_cache-0.6.16.post3+sm89-*.whl`（cp39-abi3，
  与 flashinfer-python 配套安装后运行时免 nvcc/ninja）

### 验证

```bash
bash scripts/build/sm89_verify.sh                          # pytest + import 冒烟
bash scripts/build/sm89_verify.sh --wheel dist/xxx.whl     # 另校验 wheel 内模块
```

`test_sparse_mla_sm89_gate.py` 无 GPU 即可运行；`test_sparse_mla_sm89_decode.py`
需要 SM89 GPU。

## 脚本与环境变量

| 脚本 | 作用 |
| --- | --- |
| `sm89_env.sh` | source 用：arch/local version/CCCL 开关/moe_ep 开关/nvcc 发现 |
| `sm89_install_deps.sh` | conda 环境依赖安装（torch cu130、build backend、requirements） |
| `sm89_prepare_deps.sh` | 按当前 checkout 锁定版本初始化 git submodules |
| `sm89_build_wheel.sh` | AOT wheel（`--editable` 进入开发安装） |
| `sm89_build_jit_cache_wheel.sh` | JIT-cache 平台 wheel |
| `sm89_verify.sh` | pytest + import/wheel 校验 |

常用环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `FLASHINFER_CUDA_ARCH_LIST` | `8.9` | nvcc 目标架构（SM89 构建固定） |
| `FLASHINFER_LOCAL_VERSION` / `SM89_WHEEL_MARKER` | `sm89` | wheel local version 标记 |
| `FLASHINFER_EXTRA_CUDAFLAGS` | `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1` | CCCL 版本校验绕行 |
| `BUILD_NIXL_EP` / `BUILD_NCCL_EP` | `0` | moe_ep 后端开关（EP 需要时置 1） |
| `OUTPUT_DIR` | `dist` | wheel 输出目录 |
| `MAX_JOBS` / `FLASHINFER_NVCC_THREADS` | 未设置 | 并行度（透传给 build backend） |
| `ENV_NAME` / `PYTHON_VERSION` | 当前环境 / 3.12 | install_deps 的环境名与 Python 版本 |
| `TORCH_VERSION` / `TORCH_INDEX_URL` | 最新 / cu130 索引 | torch 固定与来源 |
| `PYPI_INDEX_URL` | 未设置 | PyPI 镜像（如清华源） |
