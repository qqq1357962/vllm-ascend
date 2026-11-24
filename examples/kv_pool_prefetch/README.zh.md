# 主模型 KV 预取到 GM

合入本分支后，vLLM-Ascend worker 在 warmup 和图捕获前分配 GM 池。上一层 attention
输出投影结束后，独立 NPU stream 按下一层已有的 Pivot top-k 从分页缓存搬运选中的历史
主模型 KV。目标层写入当前 KV 后，主 stream 等待历史搬运、补齐选中的近期位置，再由
Sparse Flash Attention（SFA）按页批量读取。普通 SFA 调用沿用原路径。

## 构建和运行

使用与该 vLLM-Ascend 分支匹配的 vLLM、Torch、TorchNPU 和 CANN 环境。在仓库根目录
按标准源码安装方式重编译 vLLM-Ascend 及其自定义算子。下面以 Ascend 910B2 为例。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SOC_VERSION=ascend910b2 COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation
```

用本地 GLM-5.2 W4A8C8 权重与匹配的 DSpark draft 权重启动。单机 TP8、EP，
B1、DSpark MTP3（Q4），BF16 latent KV 512 + RoPE 64，page 128，top-2048。
预取默认关闭。导出 `VLLM_ASCEND_ENABLE_SFA_KV_PREFETCH=1` 启用。

```bash
export VLLM_ASCEND_ENABLE_SFA_KV_PREFETCH=1
export TARGET_MODEL=/path/to/GLM-5.2-W4A8C8
export DRAFT_MODEL=/path/to/DSpark
vllm serve "$TARGET_MODEL" \
  --trust-remote-code --quantization ascend --tensor-parallel-size 8 \
  --enable-expert-parallel --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 8 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --max-model-len 393216 --gpu-memory-utilization 0.94 \
  --block-size 128 --kv-cache-dtype bfloat16 --enable-prefix-caching --async-scheduling \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}' \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":3}" \
  --additional-config '{"enable_pivot_lightning_indexer":true,"enable_dsa_cp":false,"multistream_overlap_shared_expert":true,"draft_window_size":2048}'
```

DCP1 使用 `--decode-context-parallel-size 1 --max-model-len 66560 --gpu-memory-utilization 0.97`。
保持 PCP1、关闭 DSA-CP、KVPP 和 KV transfer。基线将开关设为 0，其他参数保持一致。开启时服务
启动日志应显示 `Enabled SFA KV prefetch for BF16 sparse MLA decode`。算子缺失或模型没有
符合 BF16 分页布局和 top-k 复用条件的层时，启动会报出对应原因。

实现位置

- [vLLM-Ascend 调度](../../vllm_ascend/attention/sfa_kv_prefetch.py)
- [Ascend C gather](../../csrc/kernels/sfa_kv_pool_gather.cpp)
- [Ascend 910B SFA 批量读取](../../csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_vector_mla.h)

## 历史实测 ITL

以下数据来自同一历史 vLLM-Ascend 运行时的原型对照。当前分支的集成路径需要重新测量。
测试为 Ascend 910B2、输入 65,536 token、输出 512 token、并发 1，每个变体预热 1 次，
计时 8 次。temperature 0，seed 1024，`ignore_eos=True`。ITL 是相邻流式 MTP 解码
chunk 的中位时间差，一个 chunk 可以包含多个接受的 token。

| 配置 | 原生分页 SFA（ms） | GM 预取 + 紧凑 SFA（ms） | 降低（ms） | 降幅 |
| --- | ---: | ---: | ---: | ---: |
| DCP8 | 56.6709 | 55.2933 | 1.3776 | 2.43% |
| DCP1 | 45.6089 | 45.5494 | 0.0595 | 0.13% |
