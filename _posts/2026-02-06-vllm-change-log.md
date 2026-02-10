---
layout: post
title: "Tracing vLLM's 2× Performance Gains"
authors: "Sanchit Ahuja and Harshit Garg"
date: 2026-02-06
---

The current demand for LLM inference has far outpaced the supply of hardware to serve it. GPU manufacturing timelines are measured in quarters whereas model deployment timelines are measured in days. The default industry response had been buy more GPUs, wait for the next chip which is expensive and slow. But there is a less examined question: how much performance is left on the table in the software stack running on the GPUs we already have?

In this post, we present evidence that the answer is "roughly 2×." By benchmarking every minor release of [vLLM](https://github.com/vllm-project/vllm), a widely adopted open-source LLM inference engine, from v0.1.0 through v0.11.0 on the same model, same dataset, same single A100 GPU, we find that throughput nearly doubled purely through software evolution. We trace these gains to specific engineering decisions documented in vLLM's changelogs: memory management via PagedAttention, CUDA graph capture, `torch.compile` integration, and scheduler redesigns. We also document a notable 9% performance *regression* between v0.4.0 and v0.6.0, and use the changelogs to explain why it happened and how it was resolved.

## Investigating Software-Level Gains

The first version of vLLM [(Kwon et al., 2023)](https://doi.org/10.1145/3600006.3613165), an LLM inference engine, was released in June 2023 with the promise of being a unified solution for efficient model deployment. In the two years since, vLLM has evolved rapidly, not just in its support for a wide range of model architectures, but also through significant kernel-level improvements, optimization techniques, and broader system enhancements.

Inspired by previous work on tracing the evolution of complex software systems, such as the study of Linux evolution across generations [(Ren et al., 2019)](https://doi.org/10.1145/3341301.3359640), we set out to examine how vLLM itself has changed and matured during this period.

### Experimental Setup

The vLLM framework has not yet issued a major release, so we focused our study on benchmarking its minor versions from 0.1.0 through 0.11.0. Because each version maintains backward compatibility, we selected a model supported since v0.1.0: `stabilityai/stablelm-tuned-alpha-7b`. For evaluation, we used the ShareGPT dataset[^1]. All experiments were run on a single A100 40GB GPU with CUDA 12.8. We tracked five metrics in total: throughput (requests per second), the total time taken by the engine to complete inference, the average end-to-end latency per request, and the average latency per generated token and per output token.

[^1]:[ShareGPT Vicuna Unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

### Results

Even with the same GPU, model, and dataset held constant across all versions, we observe substantial improvements in every major vLLM metric we tracked. Average latency per token dropped from 0.29s in v0.2.0 to 0.13s, and average latency per output token fell from 1.56s to 0.78s—nearly a 2× speedup. Throughput showed a similar leap, rising from 6.82 req/s in v0.2.0 to 13.58 req/s as shown in the figures below. These gains highlight an important point: despite no increase in GPU memory, vLLM achieved almost 2× better inference performance purely through software evolution.

<div class="figure-stack">
  <figure>
    <img class="zoomable" src="/assets/blog/2025-vllm/latency_per_token.png" alt="Latency per token across vLLM versions">
    <figcaption>(a) Latency per token</figcaption>
  </figure>
  <figure>
    <img class="zoomable" src="/assets/blog/2025-vllm/latency.png" alt="Overall latency across vLLM versions">
    <figcaption>(b) Overall latency</figcaption>
  </figure>
  <figure>
    <img class="zoomable" src="/assets/blog/2025-vllm/throughput.png" alt="Throughput across vLLM versions">
    <figcaption>(c) Throughput performance</figcaption>
  </figure>
</div>
<p class="figure-caption"><b>Figure 1:</b> Performance metrics across vLLM versions 0.2.0 to 0.11.0. We skipped 0.1.0 because the changes from the first version to the second were too drastic to make for clean analysis.</p>

## What is vLLM doing to achieve this?

To investigate, we manually conducted a changelog study across various minor version releases of vLLM. Our analysis revealed that vLLM's 2× performance improvement stems mostly from algorithmic changes in memory management, scheduling, and kernel design. However, the path to realizing these gains was non-linear: performance actually regressed from v0.4.0 through v0.6.x before recovering in v0.7.0. This was an interesting observation for us.

### Why Performance Regressed

Between v0.4.0 and v0.6.0, throughput dropped from 10.71 req/s to 9.78 req/s, a roughly 9% regression despite continuous development effort. A close reading of the changelogs for v0.5.0 and v0.6.0 points to several contributing factors.

**CUDA graph memory pressure**: CUDA graphs were introduced in v0.5.0, and a dedicated PR ([#5074](https://github.com/vllm-project/vllm/pull/5074)) added output buffers for graph capture to reduce their memory footprint—an acknowledgment that the footprint was already a concern. On our 40 GB A100, the pre-allocated graph buffers compete directly with KV cache memory. Fewer available KV cache blocks means a smaller effective batch size, which lowers throughput. Because CUDA graphs require fixed tensor shapes, the engine must also pad variable-length batches, adding further overhead for workloads with heterogeneous sequence lengths like ShareGPT.

**Abstraction tax from new feature scaffolding**: v0.5.0 laid the groundwork for speculative decoding ([#5400](https://github.com/vllm-project/vllm/pull/5400), [#5252](https://github.com/vllm-project/vllm/pull/5252)), automatic prefix caching ([#5324](https://github.com/vllm-project/vllm/pull/5324)), multi-modal vision support ([#4197](https://github.com/vllm-project/vllm/pull/4197), [#5237](https://github.com/vllm-project/vllm/pull/5237)), and a new `CustomOp` interface for hardware portability ([#5255](https://github.com/vllm-project/vllm/pull/5255)). A new `BlockManagerV2` ([#3834](https://github.com/vllm-project/vllm/pull/3834)) was also introduced to support CPU/GPU swapping for speculative decoding. Even when these features are not explicitly enabled, they add indirection in the scheduler and model-runner hot paths, extra dispatch layers, conditional branches, and abstraction boundaries that the runtime must traverse on every forward pass.

**Partially landed `torch.compile` integration**: By v0.6.0, the codebase had begun integrating `torch.compile`, but the integration was incomplete. A PR in that release explicitly targeted "Dynamo guard evaluation overhead" ([#7898](https://github.com/vllm-project/vllm/pull/7898)), confirming that the compilation infrastructure was imposing cost without yet delivering its intended kernel-fusion benefits. Users were, in effect, paying the price of the new compiler pipeline with none of the payoff.

**Workload mismatch**: It is also worth noting that many of the optimizations in v0.5.0–v0.6.0 such as multi-step scheduling, asynchronous output processing, chunked prefill with prefix caching—target high-concurrency, large-scale serving scenarios. Our benchmark of ~1,000 ShareGPT prompts on a single GPU may not be a good proxy where these optimizations break even, let alone provide gains.

**Fair warning**, these are all hypothesis which we arrived on after manually digging into the changelogs. We did not validate these changes by running targeted experiments testing each of these new features.

### Stability Returns to vLLM

By v0.7.0, the infrastructure investments made in v0.5.0–v0.6.0 began to pay off. Throughput jumped from 9.78 req/s to 12.30 req/s, a 26% improvement over v0.6.0 and surpassing the previous high of 10.71 req/s set by v0.4.0.

**`torch.compile` fully landed.** The partial integration that had imposed Dynamo guard overhead in v0.6.0 was completed in v0.7.0, enabling end-to-end kernel fusion across the forward pass. Where v0.6.0 paid the cost of the compiler pipeline without the benefit, v0.7.0 delivered the intended payoff: fused GPU kernels with reduced launch overhead and optimized memory access patterns. The net effect more than compensated for the abstraction layers introduced in the preceding releases.

**Feature scaffolding matured into optimized code paths.** The speculative decoding, prefix caching, and multi-modal abstractions introduced in v0.5.0 had been rough-edged—functional but not yet tuned for the hot path. Over the v0.6.x cycle, these subsystems were hardened: the asynchronous output processor ([#7049](https://github.com/vllm-project/vllm/pull/7049)) that had shipped in v0.6.0, and FlashInfer was adopted for FP8 KV cache operations ([#7798](https://github.com/vllm-project/vllm/pull/7798)). By v0.7.0, these optimizations had stabilized enough to benefit even our single-GPU, text-only benchmark.

**CUDA graph overhead was amortized.** The memory pressure from graph capture buffers, which had constrained effective batch size in v0.5.0, was mitigated by continued work on memory efficiency—including extended CUDA graph sizing for newer GPUs ([#7894](https://github.com/vllm-project/vllm/pull/7894)) and better buffer management. The fixed cost of graph capture was now spread across a more efficient execution pipeline.

From v0.7.0 onward, performance plateaued in the 13.3–13.8 req/s range through v0.11.0. This stability suggests that the major architectural bets which were PagedAttention, CUDA graphs, `torch.compile`, and the refactored scheduler had reached a mature equilibrium, with subsequent releases delivering incremental refinements now.

## Limitations

### Measurement Scope

Our analysis uses a 7B parameter model from 2023 on a single A100 GPU, which may not represent modern production deployments. Larger models (70B+) might already incorporate optimizations that reduce the relative impact of framework improvements. Multi-GPU setups introduce distributed communication overheads that could dominate performance. Different hardware architectures (H100, TPUs) have distinct optimization profiles that may not benefit equally from these software improvements.

### Optimization Specificity

Our changelog analysis reveals concerning patterns about generalizability. Many optimizations are hardware-specific (CUDA graphs for H200, x86-specific paths), limiting portability. Others are model-specific (MoE kernels, encoder-decoder improvements), raising questions about whether these gains transfer to different architectures. This specificity suggests that achieving consistent improvements across diverse deployments requires significant engineering effort.

## Conclusion

Over ten minor releases, vLLM nearly doubled its throughput on the same GPU, model, and dataset—from 6.82 req/s in v0.2.0 to 13.58 req/s in v0.11.0—without a single byte of additional GPU memory. The gains came from a concrete set of engineering choices: PagedAttention's memory management, CUDA graph capture for reduced kernel launch overhead, `torch.compile` for kernel fusion, and scheduler refinements for better batching. None of these required new hardware.

The path was not monotonic. Between v0.4.0 and v0.6.0, performance regressed by roughly 9% as the project absorbed the cost of building infrastructure for speculative decoding, multi-modal support, and cross-hardware portability. Our changelog analysis suggests this was not accidental but structural: the abstraction layers that slowed v0.5.0–v0.6.0 were prerequisites for the `torch.compile` integration and scheduler overhaul that delivered a 26% throughput jump in v0.7.0. The regression was the cost of laying foundations; the recovery was the return on that investment.

The plateau from v0.7.0 through v0.11.0 (13.3–13.8 req/s) is equally informative. It suggests that the current architectural paradigm, continuous batching with paged KV cache and compiled kernels—may be approaching its ceiling on this hardware and model class. The next big change will likely require either new algorithmic ideas (disaggregated prefill/decode, workload-aware speculative decoding) or hardware capabilities that open different optimization surfaces entirely.

Our study has clear limits. A single 7B model on a single A100 cannot represent the diversity of production deployments. Whether this 2× trajectory holds for 70B+ models under tensor parallelism, for mixture-of-experts architectures, or on H100/H200 hardware with different memory budgets remains open. The regression hypotheses we derived from changelogs are plausible but not validated. We have not yet run the controlled experiments (e.g., `--enforce-eager` on v0.5.3, `nsys` profiling of scheduler overhead) needed to definitively prove that these were the probable reasons causing the regression.

## Future Work

### Validating the Regression Hypotheses

The most immediate next step is to confirm or refute the causal claims we made about the v0.4.0–v0.6.0 regression. Specifically:

- **CUDA graph memory pressure** can be isolated by re-running v0.5.3 and v0.6.0 with `--enforce-eager` (which disables CUDA graphs entirely). If throughput recovers to ~10.7 req/s, graph buffer allocation is the dominant factor. Monitoring `gpu_cache_usage_perc` across versions would further quantify how many KV cache blocks were lost to graph capture buffers.
- **Abstraction overhead** from the `CustomOp` interface, `BlockManagerV2`, and multi-modal code paths could be measured by profiling the scheduler and model-runner hot loops with `nsys` on v0.4.0 vs. v0.6.0 and comparing kernel launch counts and CPU-side scheduling time.
- **Partial `torch.compile` cost** can be tested by disabling compilation on v0.6.0 and comparing against the default configuration.

Each of these is a single controlled experiment against a known baseline—straightforward to run and sufficient to convert our changelog-derived hypotheses into measured attributions.

### Broader Model and Hardware Coverage

Our study uses a single 7B model on a single A100. Two natural extensions would strengthen the conclusions:

- **Larger and newer architectures.** Models like Llama 3.1 70B and Mixtral 8x7B exercise code paths (tensor parallelism, MoE routing kernels) that `stablelm-7b` does not touch. Running the same version sweep on these models would reveal whether the regression and recovery pattern we observed is universal or specific to small, dense models.
- **Multi-GPU scaling.** vLLM introduced significant changes to its distributed executor across these versions (multiprocessing default in v0.5.0, pipeline parallel support for Intel GPUs in v0.6.0). Benchmarking on 2- and 4-GPU setups would test whether the single-GPU throughput trajectory holds or diverges under communication overhead.
- **Newer hardware.** The v0.6.0 changelog explicitly mentions extended CUDA graph sizing for H200 ([#7894](https://github.com/vllm-project/vllm/pull/7894)). Repeating our experiments on H100 or H200 would test whether the CUDA graph memory pressure we hypothesized is specific to the 40GB memory budget of our A100.

### Cross-Framework Comparison

vLLM is not the only inference engine that has evolved rapidly over this period. SGLang [(Zheng et al., 2025)](https://dl.acm.org/doi/10.5555/3737916.3739916), for instance, has made different optimization choices around RadixAttention for automatic KV cache reuse and compressed finite state machines for structured generation. A controlled comparison as in with same model, same dataset, same hardware, same version timespan, would help distinguish which of vLLM's gains come from broadly applicable algorithmic ideas (e.g., PagedAttention, continuous batching) versus implementation-specific engineering (e.g., FlashInfer integration, custom CUTLASS kernels). This would also surface cases where one framework's optimization regressed performance in a way that another framework avoided entirely.


---

### References

1. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://doi.org/10.1145/3600006.3613165). *SOSP '23*.
2. Ren, X., Rodrigues, K., Chen, L., Vega, C., Stumm, M., & Yuan, D. (2019). [An analysis of performance evolution of Linux's core operations](https://doi.org/10.1145/3341301.3359640). *SOSP '19*.
3. Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kober, C., Leng, C., Han, S., Barrett, B., Gonzalez, J., & Stoica, I. (2025). [SGLang: Efficient Execution of Structured Language Model Programs](https://dl.acm.org/doi/10.5555/3737916.3739916). *NeurIPS '24*.