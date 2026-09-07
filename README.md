# CS683: Advanced Computer Architecture — Programming Assignment 1

### Authors & Submission Information
- **Tejavath Sri Chandravadan** (Roll No: `24B0931`)
- **Yashwanth Murasani** (Roll No: `24B1039`)
- **Julapelly Hanikar Rao** (Roll No: `24B1079`)
- **Shreshta Bathini** (Roll No: `24B1085`)

**Moodle Submission Archive**: `24B0931_24B1039_24B1079_24B1085_pa1.tar.gz` *(Sorted roll number ordering per specification)*  

## Submission Package Directory Layout

The structure of the final submission archive adheres strictly to the layout specified in the assignment guidelines:

```
pa1-dhurandhar-microarchitecture/
├── task1/
│   ├── src/
│   │   ├── conv_reorder.cpp           # Stage 1: Loop reordering (hoisting kernel taps outward)
│   │   ├── conv_unroll.cpp            # Stage 2: Loop unrolling (8 independent scalar accumulators)
│   │   ├── conv_tile.cpp              # Stage 3: Cache tiling / blocking
│   │   ├── conv_simd.cpp              # Stage 4: 256-bit AVX2 SIMD vectorization
│   │   └── conv_optimized.cpp         # Stage 5: Final combined 2D register-unrolled kernel
│   └── task1_report.pdf               # Comprehensive technical report for Task 1
├── task2/
│   ├── src/
│   │   ├── matmul_prefetch.cpp        # Task 2A: Software prefetching & LFB contention tuning
│   │   ├── matmul_simd.cpp            # Task 2B: 256-bit AVX2 vectorization & memory wall analysis
│   │   └── matmul_optimized.cpp       # Task 2C: Synergistic L2 blocking + 4x2 register tile + prefetch
│   └── task2_report.pdf               # Comprehensive IEEE-style technical report for Task 2
├── plots/
│   ├── task1_speedup_vs_size.png      # Task 1 speedup vs matrix size across all 5 stages
│   ├── task1_speedup_vs_kernel.png    # Task 1 speedup vs kernel size (K=3, 5, 7) across all stages
│   ├── task2_speedup_vs_size.png      # Task 2 speedup vs matrix size (Prefetch vs SIMD vs Optimized)
│   ├── task2_prefetch_degree.png      # Task 2 prefetch lookahead degree / distance D sweep
│   └── task2_prefetch_level.png       # Task 2 prefetch cache fill level & locality hint analysis
└── README.md                          # Comprehensive project documentation (this file)
```
<!-- 
## Task 1: 2D Spatial Convolution

### 1. Mathematical Formulation
Single-channel spatial convolution of an input image with a $K \times K$ kernel under zero-padded "same" boundary conditions:
$$\text{out}[oy, ox] = \sum_{ky=0}^{K-1} \sum_{kx=0}^{K-1} \text{in}[oy+ky, ox+kx] \cdot \text{ker}[ky, kx]$$
Where:
- Input image: $H \times W$ padded by $p = \lfloor K/2 \rfloor$ on all four borders.
- Output image: $H \times W$ stored contiguously in row-major order.
- Inner loop bounds checks are completely eliminated by computing on pre-padded input buffers.

---

### 2. Progressive Optimization Stages

#### Stage 1: Loop Reordering (`conv_reorder.cpp`)
- **Baseline Bottleneck**: The reference naive loop order ($oy \to ox \to ky \to kx$) walks kernel taps in the innermost loop. This incurs $K \times K$ repeated scalar loads of kernel weights per pixel, prevents contiguous memory streaming, and degrades hardware prefetch efficiency.
- **Transformation**: Hoisted kernel loops outward ($oy \to ky \to kx \to ox$).
- **Microarchitectural Benefit**: The kernel weight `ker[ky, kx]` becomes a loop-invariant scalar for the entire output row, loaded once into a register. The innermost loop along $ox$ becomes a contiguous, unit-stride SAXPY stream:
  $$\text{out}[oy, ox] \mathrel{+}= \text{ker\_val} \cdot \text{in}[oy+ky, ox+kx]$$
- **Speedup**: Achieves **$1.32\times\text{--}1.54\times$ speedup** across matrix dimensions.

#### Stage 2: Loop Unrolling (`conv_unroll.cpp`)
- **Baseline Bottleneck**: In scalar SAXPY loops, single-accumulator dependencies (`acc += ...`) create a serialized latency chain of 4–5 clock cycles per floating-point addition, stalling execution ports 0 and 1.
- **Transformation**: Unrolled the inner loop by an unroll factor of 8 (matching the 256-bit AVX2 vector width) using 8 independent scalar accumulators (`acc0` through `acc7`).
- **Microarchitectural Benefit**: Exposes Instruction-Level Parallelism (ILP), amortizes loop branch overhead by $8\times$, and allows the CPU out-of-order execution window to schedule independent arithmetic operations simultaneously.
- **Speedup**: Delivers **$3.21\times\text{--}3.52\times$ speedup**.

#### Stage 3: Cache Tiling (`conv_tile.cpp`)
- **Profiling & Exploration**: Evaluated 2D cache block tiles across $T \in \{16, 32, 64, 128, 256, 1024\}$.
- **Hardware Counter Insight**: On $2048 \times 2048$, tiling reduced L1-D Misses Per Kilo-Instruction (MPKI) from $0.24$ down to $0.11$. However, standalone runtime speedup was marginal ($1.03\times\text{--}1.05\times$).
- **Explanation**: In 2D convolution, memory accesses within a row are strictly contiguous and linear. Intel's hardware stream prefetcher already detects unit strides with $>95\%$ accuracy. Adding explicit 2D loop tiling introduced outer loop branching overhead without improving cache reuse sufficiently to offset it.

#### Stage 4: SIMD Vectorization (`conv_simd.cpp`)
- **Vector Extensions**: Replaced scalar loops with 256-bit AVX2 intrinsics (`_mm256_fmadd_ps`).
- **Instruction Stream Compaction**: Dynamic retired instructions dropped by **82.74%** (from $4.55 \times 10^9$ down to $7.85 \times 10^8$ on $1024 \times 1024$).
- **Speedup**: 128-bit SSE achieved $3.69\times\text{--}3.96\times$, while 256-bit AVX2 achieved **$7.06\times\text{--}7.58\times$ speedup** ($7.42\times$ average, a 642% gain over scalar baseline).

#### Stage 5: Combined Optimized Kernel (`conv_optimized.cpp`)
- **Architectural Co-Design**: Combines 256-bit AVX2 with aggressive 2D register unrolling, computing $2 \text{ output rows} \times 32 \text{ output columns}$ (4 vector registers wide) concurrently.
- **Register Budgeting**: Uses 8 vector accumulators (`v00` through `v13`) and 1 broadcast weight vector, fitting within the 16 architectural YMM registers without stack spilling.
- **FMA Memory Folding**: Directly folds memory loads into `_mm256_fmadd_ps`, leveraging x86 memory-operand addressing to maximize pipeline throughput.
- **Speedup**: Delivers **$11.07\times\text{--}16.30\times$ speedup** across all matrix sizes.

---

### 3. Task 1 Quantitative Performance Summary

| Image Dimension | Naive Time | Reorder | Unroll | Tiling | SIMD (AVX2) | Combined Optimized | Peak Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128 $\times$ 128** | 0.046 ms | 1.32$\times$ | 3.32$\times$ | 1.00$\times$ | 7.26$\times$ | **0.003 ms** | **16.30$\times$** |
| **256 $\times$ 256** | 0.161 ms | 1.50$\times$ | 3.21$\times$ | 0.81$\times$ | 7.74$\times$ | **0.010 ms** | **15.33$\times$** |
| **512 $\times$ 512** | 0.721 ms | 1.32$\times$ | 3.29$\times$ | 0.82$\times$ | 7.98$\times$ | **0.059 ms** | **12.25$\times$** |
| **1024 $\times$ 1024** | 2.564 ms | 1.54$\times$ | 3.36$\times$ | 0.87$\times$ | 7.81$\times$ | **0.193 ms** | **13.26$\times$** |
| **2048 $\times$ 2048** | 11.285 ms | 1.46$\times$ | 3.52$\times$ | 0.91$\times$ | 7.04$\times$ | **1.019 ms** | **11.07$\times$** |

*Primary Graded Benchmark ($2048 \times 2048, K=3$ under grading harness): **8.64$\times$ speedup (39.81 GFLOP/s, 1.896 ms)** — **100 / 100 points**.*

---

## Task 2: Dense Matrix Multiplication for Inference (SGEMM)

### 1. Mathematical Formulation
Contiguous NT-layout General Matrix Multiply ($C = A \cdot B^T$), directly mirroring the execution pattern of `llama.cpp`'s `ggml_mul_mat`:
$$C[i][j] = \sum_{p=0}^{K-1} A[i][p] \cdot B[j][p]$$
Where:
- $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{N \times K}$ (both row-major, $B$ stored transposed), $C \in \mathbb{R}^{M \times N}$.
- Because $B$ is transposed, both operands are accessed with **unit stride along the inner contraction loop $p$**, forming continuous dot products.
- Computational Workload: $2 \cdot M \cdot N \cdot K$ floating-point operations. Primary graded benchmark: $M = N = K = 1024$.

---

### 2. Progressive Optimization Stages

#### Task 2A: Software Prefetching (`matmul_prefetch.cpp`)
- **Exploration Space**: Swept prefetch lookahead distance $D \in [1, 256]$ and cache locality hints (`_MM_HINT_T0`, `_MM_HINT_T1`, `_MM_HINT_T2`, `_MM_HINT_NTA`).
- **Line Fill Buffer (LFB) Contention Discovery**: Issuing `_mm_prefetch` on every loop iteration ($p++$) generates 16 redundant prefetch instructions per 64-byte cache line. Flooding the physical core's 12–16 LFBs causes structural pipeline stalls, causing performance to drop by nearly $50\%$ ($0.51\times$).
- **Stride Gating Solution**: Constrained prefetch issuance using bitmask stride gating:
  ```cpp
  if ((p & 7) == 0) {
      _mm_prefetch(reinterpret_cast<const char*>(a + p + DIST), HINT);
      _mm_prefetch(reinterpret_cast<const char*>(b + p + DIST), HINT);
  }
  ```
- **Key Insight**: In unvectorized scalar execution, prefetching adds instruction overhead without hiding compute latency ($\approx 1.00\times$). Optimal parameters: **$D = 16\text{--}32$ float elements** with **`_MM_HINT_T2`** (L3 prefetch avoids polluting the active L1-D working set).

#### Task 2B: SIMD Vectorization & The Memory Bandwidth Wall (`matmul_simd.cpp`)
- **Vector Width Scaling**:
  - **64-bit SIMD**: Flat $1.63\times\text{--}1.68\times$ speedup; instructions reduced by **$-46\%$**.
  - **128-bit SSE**: Stable $\approx 2.7\times$ speedup; instructions reduced by **$-72\%$**.
  - **256-bit AVX2**: Slashes dynamic instructions by **$-86.3\%$** ($1.08 \times 10^{10} \to 1.52 \times 10^9$).
- **The Memory Bandwidth Wall**: On small working sets ($256 \times 256$ and $512 \times 512$), 256-bit AVX2 delivers high speedups of **$6.56\times\text{--}7.49\times$** ($33.97\text{ GFLOP/s}$). However, at $2048 \times 2048$ (working set: $48\text{ MiB}$, vastly exceeding the $12\text{ MiB}$ LLC), speedup plummets to **$3.03\times$**. Because SIMD consumes operands $8\times$ faster, the execution pipeline stalls waiting for DRAM lines to arrive, shifting the bottleneck from compute-bound to memory-bandwidth bound.

#### Task 2C: Synergistic Co-Design (`matmul_optimized.cpp`)
To break through the memory wall and surpass the 25.0$\times$ speedup threshold for maximum points (70/70 speedup points), our final kernel integrates four complementary microarchitectural optimizations:

1. **2D L2 Cache Blocking ($B_M = 64, B_N = 128$)**: Bounds active sub-blocks of $A$ ($64 \times 1024 \times 4\text{ B} = 256\text{ KiB}$) and $B$ ($128 \times 1024 \times 4\text{ B} = 512\text{ KiB}$) to $768\text{ KiB}$, fitting entirely within the $1.25\text{ MiB}$ private L2 cache. This reduces L2 misses by **$-61.9\%$** and LLC misses by **$-62.5\%$**.
2. **$4 \times 2$ Vector Register Tiling**: Unrolls 4 rows of $A$ and 2 rows of $B$ across 8 concurrent YMM accumulator registers (`c00` through `c31`). In each inner iteration, loading 4 vectors of $A$ and 2 vectors of $B$ performs $4 \times 2 = 8$ FMAs, achieving an arithmetic intensity of $8 / 6 \approx 1.33$ FMAs per vector load. This cuts L1-D cache misses by **$-76.2\%$**.
3. **Targeted Software Prefetching ($D = 16\text{--}32$, `_MM_HINT_T0` / `T2`)**: Pre-warms upcoming cache lines of $B$ into the cache hierarchy ahead of computation, providing an empirical **$+4.14\times$ speedup boost** (pushing performance from $23.43\times$ up to $27.57\times$).
4. **In-Register Vector Reduction (`opt_add`)**: Folds 256-bit accumulator vectors into scalar sums using 128-bit SSE shuffles and horizontal adds, completely avoiding serialized memory write-backs.

---

### 3. Task 2 Quantitative Performance Summary

#### Graded Milestone Summary ($M = N = K = 1024$)
| Implementation Stage | Execution Time | Throughput | Speedup vs Naive | Autograder Points |
| :--- | :---: | :---: | :---: | :---: |
| `matmul_naive` (Scalar Baseline) | 945.66 ms | 2.27 GFLOP/s | 1.00$\times$ | Baseline Reference |
| `matmul_prefetch` (Isolated Task 2A) | 968.79 ms | 2.22 GFLOP/s | 0.98$\times$ | 10 / 10 Points |
| `matmul_simd` (Isolated Task 2B) | 163.41 ms | 13.14 GFLOP/s | 5.79$\times$ | 10 / 10 Points |
| **`matmul_optimized` (Combined Task 2C)** | **36.74 ms** | **58.46 GFLOP/s** | **25.74$\times$** | **70 / 70 Points** |
| **Total Task 2 Autograder Score** | — | — | — | **100 / 100 Points** |

#### Microarchitectural Hardware Performance Counters ($1024 \times 1024$)
| Hardware Event | Naive Baseline | Standalone SIMD | Combined Optimized Kernel | Microarchitectural Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Dynamic Instructions** | $1.08 \times 10^{10}$ | $1.52 \times 10^9$ | **$1.48 \times 10^9$** | **$-86.3\%$ instructions** |
| **L1 Data Cache Misses** | $1.25 \times 10^8$ | $8.95 \times 10^7$ | **$2.98 \times 10^7$** | **$-76.2\%$ L1-D misses** |
| **L2 Cache Misses** | $1.10 \times 10^9$ | $8.12 \times 10^8$ | **$4.20 \times 10^8$** | **$-61.9\%$ L2 misses** |
| **LLC (L3) Cache Misses** | $1.61 \times 10^7$ | $1.42 \times 10^7$ | **$6.04 \times 10^6$** | **$-62.5\%$ LLC misses** |

#### Scalability Across Problem Dimensions
| Matrix Dimension | Naive Time | Standalone Prefetch | Standalone SIMD | Final Optimized Kernel | Sustained GFLOP/s | Speedup vs Naive |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256 $\times$ 256** | 6.64 ms | 0.90$\times$ | 6.56$\times$ | **0.39 ms** | 85.6 GFLOP/s | **16.92$\times$** |
| **512 $\times$ 512** | 59.20 ms | 0.93$\times$ | 7.49$\times$ | **2.59 ms** | 103.8 GFLOP/s | **22.88$\times$** |
| **1024 $\times$ 1024** | 511.77 ms / 945.66 ms* | 0.99$\times$ | 5.09$\times$ | **19.72 ms / 36.74 ms*** | 58.46 GFLOP/s | **25.74$\times\text{--}25.95\times$** |
| **2048 $\times$ 2048** | 4389.50 ms | 1.01$\times$ | 3.03$\times$ | **171.80 ms** | 100.0 GFLOP/s | **25.55$\times$** |

*\*Note: Secondary timing denotes automated multi-pass grading harness results.*

--- -->
<!-- 
## Empirical Plots Guide (`plots/`)

All required plots are high-resolution (200+ DPI) figures generated directly from empirical hardware benchmark data:

### 1. `plots/task1_speedup_vs_size.png`
- **Focus**: Evaluates Task 1 speedup scaling across matrix dimensions ($128 \times 128$ to $2048 \times 2048$) for all five stages.
- **Key Insight**: Demonstrates that while standalone optimizations (reorder, unroll, tile, SIMD) provide incremental benefits, their co-designed combination (`conv_optimized.cpp`) achieves up to $16.30\times$ speedup, sustaining $>11\times$ even at $2048 \times 2048$.

### 2. `plots/task1_speedup_vs_kernel.png`
- **Focus**: Evaluates Task 1 performance across kernel filter dimensions ($K \in \{3, 5, 7\}$).
- **Key Insight**: Highlights how loop unrolling and SIMD scaling sustain high arithmetic throughput as arithmetic intensity scales with larger $K \times K$ filter stencils.

### 3. `plots/task2_speedup_vs_size.png`
- **Focus**: Compares Naive Baseline (1.00x), Software Prefetching, SIMD (AVX2), and the Final Optimized Code across matrix dimensions $256 \times 256$ to $2048 \times 2048$.
- **Key Insight**: Clearly illustrates the **Memory Bandwidth Wall** in unblocked SIMD (falling from $7.49\times$ down to $3.03\times$) and proves how the Final Optimized Code overcomes it through L2 cache blocking, maintaining a steady **$25.55\times\text{--}25.74\times$ speedup**.

### 4. `plots/task2_prefetch_degree.png`
- **Focus**: Dual-panel analysis of prefetch lookahead distance / degree $D \in [1, 256]$:
  - **Panel (a)**: Standalone scalar prefetching sweep across `T0`, `T1`, and `T2`, identifying $D = 16\text{--}32$ as the optimal range.
  - **Panel (b)**: Final Optimized Kernel prefetch distance sweep, showing that without prefetch ($D=0$) performance is $23.43\times$, while lookahead prefetching at $D=16\text{--}32$ boosts performance by **$+4.14\times$ up to $27.57\times$**.

### 5. `plots/task2_prefetch_level.png`
- **Focus**: Dual-panel analysis across cache fill levels (`T0`, `T1`, `T2`, `NTA`):
  - **Panel (a)**: Standalone prefetching average and best speedups across cache tiers.
  - **Panel (b)**: Final Optimized Kernel locality response, proving that non-temporal prefetching (`_MM_HINT_NTA`) causes a severe **$-5.20\times$ speedup penalty** (dropping to $18.67\times$) because reused register-tiled elements bypass the cache hierarchy. Confirms `_MM_HINT_T0` and `T2` as optimal. -->


## Build, Verification, and Reproduction Guide

### Standalone Compilation
Both tasks can be compiled independently using standard `make`:

```bash
# Compile Task 1
cd task1
make
./bin/conv                       # Runs official evaluation benchmark (2048x2048, K=3)

# Test individual Task 1 kernels
./bin/conv reorder 512 512 3
./bin/conv unroll  512 512 3
./bin/conv tile    1024 1024 3
./bin/conv simd    1024 1024 3
./bin/conv all     2048 2048 3

# Compile Task 2
cd ../task2
make
./bin/matmul                     # Runs official evaluation benchmark (1024x1024x1024)

# Test individual Task 2 kernels
./bin/matmul prefetch 1024 1024 1024
./bin/matmul simd     1024 1024 1024
./bin/matmul all      2048 2048 2048
```
<!-- 
### Hardware Performance Counter Profiling
To verify hardware counters using Linux `perf`:
```bash
# Profile Task 1 cache and instruction metrics
perf stat -e instructions,cycles,L1-dcache-load-misses,L1-dcache-loads ./task1/bin/conv

# Profile Task 2 L1-D, L2, and LLC miss metrics
perf stat -e instructions,cycles,L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses ./task2/bin/matmul
``` -->

