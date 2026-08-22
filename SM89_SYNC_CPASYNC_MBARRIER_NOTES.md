# SM89 同步修复日记：cp.async 模拟 bulk 的 mbarrier 可见性

日期：2026-08-22
分支：`v0.6.17-sm89`
影响范围：sparse MLA decode（DSv4 / DSv3.2）与 prefill 的 SM89 路径
状态：代码已改，待远程重编译 + 服务测试后提交

## 1. 背景

SM89（Ada/L40S）移植复用了 SM120 的 sparse MLA 框架。SM120 侧 IO warp 用
`cp.async.bulk`（TMA）+ `mbarrier.arrive.expect_tx` 让硬件把拷贝完成计入
mbarrier 的 tx-count；SM89 没有 TMA，`cp_async_bulk_g2s()` 退化为逐 16B 的
`cp.async`，`expect_tx` 变 no-op。原来的 SM89 完成路径是：

```cpp
cp_async_wait_all();               // 只保证“发出线程自己”能看到拷贝
bar_sync_t<4, IO_THREADS>();       // 只包含 IO warp，不含 math warp
if (lane == 0) mbarrier_arrive();  // 普通 arrive
```

math 侧在 `mbarrier_wait_parity()` 之后补一个 `bar_sync_t<3, MATH_THREADS>`
（同样只含 math warp）。这套“两边各自 bar、跨侧靠 mbarrier”的写法在真机上能跑，
但按 PTX 手册字面，**cp.async 的拷贝数据对 math 侧没有形式上的可见性保证**——
这正是本次修复的对象。

## 2. PTX ISA 9.1 手册要求（逐条）

### 2.1 mbarrier.arrive 缺省是 release

§9.7.13.15.13（mbarrier.arrive）：

> If the .sem qualifier is absent, .release is assumed by default.

结论：现有 `mbarrier_arrive()` 的裸 `mbarrier.arrive.shared::cta.b64` 已经带
release 语义，generic-proxy 的写（scale 的 `st.shared` + `__threadfence_block`）
在 acquire 方可见。这一条**不需要改代码**，但注释要写清楚，避免后人误加 `.relaxed`
或误删 barrier。

### 2.2 test_wait / try_wait.parity 缺省是 acquire

§9.7.13.15.16（mbarrier.test_wait/try_wait）：

> When ... with .acquire qualifier returns True, they form the acquire pattern...
> If the .sem qualifier is absent, .acquire is assumed by default.

且 acquire 返回 True 时保证（该节“ordering of memory operations”清单）：

1. 完成阶段内、release arrive 之前的**非异步**内存访问可见；
2. **cp.async** 操作：完成阶段内、`cp.async.mbarrier.arrive` 之前的拷贝可见；
3. **cp.async.bulk**：使用同一 mbarrier、release arrive 之前的 bulk 可见。

SM89 的 `mbarrier_wait_parity()` 用 `mbarrier.test_wait.parity`（要求 sm_80+；
`try_wait` 要求 sm_90+，SM89 不能用），同样缺省 acquire。

**关键点：裸 cp.async 不在第 1 条里（“except async operations”），只在第 2 条里**
（前提是发出线程执行了 `cp.async.mbarrier.arrive`）。旧的
`cp_async_wait_all() + 普通 release arrive` 不属于任何一条 → 数据可见性没有形式
保证，只能靠硬件实际行为。

### 2.3 fence.proxy.async 在 SM89 不可用

§fence（9.7.9.1）：`fence.proxy.async` 要求 **sm_90 or higher**。SM89 上不能
用“async proxy ↔ generic proxy”代理栅栏来补 cp.async 可见性，唯一的形式化手段
就是 2.2 的第 2 条：`cp.async.mbarrier.arrive`。

### 2.4 cp.async.mbarrier.arrive 的语义与计数

§9.7.9.25.3（cp.async.mbarrier.arrive），要求 **sm_80 or higher**（SM89 可用）：

> Makes the mbarrier object track all prior cp.async operations initiated by the
> executing thread... When .noinc modifier is not specified, the pending count of
> the mbarrier object is incremented by 1 prior to the asynchronous arrive-on
> operation. This results in a zero-net change for the pending count...

推论：

- **非 noinc 形式**对 pending count 净零变化——它只负责“把拷贝和 mbarrier 绑
  定”，不是完成那次 arrive。phase 仍由某个线程的一次普通 `mbarrier_arrive()`
  完成，`mbarrier.init` 的计数（=1）**不用改**，正是手册 Example 1 的形态。
- **noinc 形式**要求 init 计数预先计入每次 arrive-on（Example 2：init = 完成
  arrive 数 + 所有 noinc arrive-on 数）。旧代码 init=1 却尝试
  `cp.async.mbarrier.arrive.noinc`，pending 会被减成负数 → 未定义行为/指令陷阱，
  这就是旧注释里 “noinc traps in this context” 的真正原因。
- 每个 IO lane 都要执行一次 `cp.async.mbarrier.arrive`：第 2 条只覆盖“执行该指令
  的线程”自己的 cp.async。只让 lane 0 执行覆盖不到其他 lane 的拷贝。

### 2.5 named barrier 只对参与线程保证可见性

§9.7.13.15.11（bar.sync / barrier.cta.sync）：

> ... prior memory accesses requested by this thread are performed relative to
> **all threads participating in the barrier**.

推论：`bar_sync_t<4, IO_THREADS>`（只含 IO warp）和 `bar_sync_t<3,
MATH_THREADS>`（只含 math warp）都**不是跨 IO→math 的 fence**。它们能做的：

- IO 侧 bar：让各 lane 的 arrive-on 增量、scale 写彼此有序，再统一被 lane 0 的
  release arrive 覆盖（传递性）；
- math 侧 bar：让各 math warp 在读到同一块 smem（reduce 缓冲、sm_p_full）前进度
  一致。

所以旧代码把 math 侧 bar 称作 “CTA-wide acq-rel” 是误导——跨侧数据可见性只能由
mbarrier 的 release/acquire 对提供。

## 3. 修改内容

### 3.1 `arch/cp_async.cuh`

- 新增 `cp_async_mbarrier_arrive(mbar)`：裸 `cp.async.mbarrier.arrive.shared::cta.b64`，
  非 noinc。保留原 `..._noinc` helper（当前无调用），注释说明为什么不能用 noinc。
- `cp_async_bulk_g2s()` / `cp_async_bulk_g2s_l2hint()` 的 SM89 分支补注释：
  调用方必须在 gather 后执行 `cp_async_mbarrier_arrive()` + 一次
  `mbarrier_arrive()`。

### 3.2 `arch/barrier.cuh`

- `mbarrier_arrive()` / `mbarrier_wait_parity()` / `mbarrier_arrive_expect_tx()`
  补语义注释（2.1/2.2/2.3 的结论），代码不变。

### 3.3 `common/kv_cache_io.cuh`（prefill 的 gather）

SM89 完成路径由：

```cpp
cp_async_wait_all();
bar_sync_t<4, IO_THREADS>();
if (io_tid == 0) mbarrier_arrive(mbar);
```

改为：

```cpp
cp_async_mbarrier_arrive(mbar);   // 每个 IO lane，绑定自己的 cp.async
bar_sync_t<4, IO_THREADS>();      // 各 lane arrive 增量/scale 写有序
if (io_tid == 0) mbarrier_arrive(mbar);  // 唯一完成性 arrive（init=1 不变）
```

### 3.4 `decode_dsv4_kernel.cuh` / `decode_dsv3_2_kernel.cuh`

- `issue_gather` 的 SM89 完成路径同样替换（去掉 `cp_async_wait_all()`）。
- 头部与 math 侧 wait 后的注释改为准确描述：test_wait 缺省 acquire 负责跨侧数据
  可见性；`bar_sync_t<3, MATH_THREADS>` 只是 math warp 间的进度同步。

### 3.5 `prefill_kernel.cuh`

- 两处 SM89 “CTA-wide acq-rel” 注释改为“math-wide sync”，数据排序由 mbarrier
  release/acquire 提供。代码路径不变（走 3.3 的 `io_bulk_gather_tile`）。

## 4. 为什么去掉 cp_async.wait_all 是安全的

phase 完成条件 = pending count 归零。每个 IO lane 的 `cp.async.mbarrier.arrive`
（非 noinc）先 +1 再在**拷贝完成时** -1；lane 0 的普通 arrive 只把 init 的 1 减掉。
因此只要还有任何 lane 的拷贝未完成，pending 就 > 0，phase 不会翻转。math 侧
`test_wait` 返回 True 时，所有拷贝必然已完成且可见（2.2 第 2 条）。

去掉 wait_all 后 IO warp 不再等自己的拷贝，下一 chunk 的地址计算/发拷贝与当前拷贝
重叠更好；mbarrier 完成了原本 wait_all 的“等拷贝”职责。

## 5. 风险边界与后续验证

- 显式 `.release`/`.acquire` 限定符（`mbarrier.arrive.release.cta`、
  `test_wait.parity.acquire.cta`）在 sm_89 上的可用性手册未逐条列出（只明确
  `.relaxed` 要求 sm_90+），本次**不添加**，依赖手册明确给出的缺省语义，行为等价。
- 验证建议（远程）：
  1. 跑 `tests/attention/test_sparse_mla_sm89_decode.py`（数值 vs 参考实现）；
  2. 长上下文 + 多请求压测，重点观察 fp8 主路径的 bitwise 一致性（可临时开
     `FLASHINFER_DEBUG=1` 或对照 prefill/decode 输出的 LSE）；
  3. 服务重启后跑一轮真实请求（vLLM serve 脚本），确认无
     `illegal memory access` / 数值漂移。
