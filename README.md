# PlayStation 5 Distributed LLM Inference Cluster

A high-performance, cost-effective distributed Large Language Model (LLM) inference cluster built using **two Sony PlayStation 5 (PS5)** consoles running Linux. By combining the PS5's high-bandwidth unified GDDR6 memory architecture over a 1 Gbps Layer 2 hardware-offloaded network using `llama.cpp` Vulkan RPC, this setup achieves high token generation speeds (up to **41 t/s**) for dense and Mixture-of-Experts (MoE) models up to 35B parameters.

## 📌 Performance Highlights

All benchmarks were conducted using `Q4_K_M` quantized GGUF weights across two PS5 nodes via `llama.cpp` (build `b11813`) Vulkan RPC pipeline parallelism.

| Model | Architecture | Active Parameters | Total VRAM | Prompt Eval (Prefill) | Generation Throughput | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Qwen3.6-35B-A3B** | MoE | \~3.0B | \~17.0 GB | \~220.0 t/s | **41.00 t/s** | 
| **Qwen2.5-14B** | Dense | \~14.7B | \~9.0 GB | \~180.0 t/s | **35.01 t/s** | 
| **Qwen3.8-27B** | Dense | \~27.0B | \~17.0 GB | \~145.0 t/s | **17.27 t/s** | 

## 🛠 Hardware & Network Infrastructure

* **Nodes:** 2x Sony PlayStation 5 Consoles (Oberon APU / RADV GFX1013)
* **Memory:** 16 GB GDDR6 Unified Memory per node (\~14.1 GB usable VRAM per console after system reservation)
* **Combined VRAM Pool:** **\~28.2 GB** across the cluster
* **Network Switch:** MikroTik RB5009 (L2 Hardware Offloading active, standard MTU 1500)
* **Interconnect:** 1 Gbps Ethernet via L2 bridge switching (<20μs forwarding latency)

## 🐧 OS & System Configuration

Both consoles are booted into Ubuntu 26.04 LTS using the [soniciso1/ps5-linux-loader](https://github.com/soniciso1/ps5-linux-loader).

### 1. Minimal OS Memory Footprint

To maximize available GDDR6 RAM for LLM weights and KV caching, disable the graphical desktop environment (GUI) to run in headless mode:

```bash
sudo systemctl set-default multi-user.target
```

### 2. User Permissions & Graphics Access

Grant your user account the appropriate permissions for Vulkan direct rendering:

```bash
sudo usermod -aG render,video $USER
```

### 3. Kernel Boot Parameters (`/boot/efi/cmdline.txt`)

Modify the boot line to unlock dynamic TTM memory page allocation and eliminate CPU kernel mitigation overhead:

```text
root=LABEL=ubuntu2604 rw rootwait console=ttyTitania0 console=tty0 video=DP-1:1920x1080@60 mitigations=off idle=halt preempt=full ttm.pages_limit=4194304
```

> ⚠️ **CRITICAL PARAMETER:** `ttm.pages_limit=4194304` is mandatory. Without it, the kernel caps Vulkan allocation to \~3–4 GB. Setting this unlocks up to **14.1 GB usable unified memory** per console.

## 🚀 Building `llama.cpp` (Version `b11813`)

Compile matching binaries on both PS5 consoles using the Vulkan and GGML RPC backends. Ensure both nodes are compiled from the exact same release tag (`b11813`) to prevent socket wire format version mismatches.

### 1. Install Dependencies

```bash
sudo apt update && sudo apt install -y build-essential cmake git libvulkan-dev vulkan-tools glslang-tools
```

### 2. Build Binaries

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout b11813
cmake -B build \
  -DGGML_VULKAN=ON \
  -DGGML_RPC=ON \
  -DGGML_NATIVE=OFF
cmake --build build --config Release -j$(nproc)
```

## ⚙️ Cluster Deployment & Execution

Assuming generic local network assignments:
* **Host Node (PS5 #1):** `192.168.1.101`
* **Worker Node (PS5 #2):** `192.168.1.102`

### Step 1: Start Worker Node (PS5 #2)

On the secondary PS5 node (`192.168.1.102`), launch the `ggml-rpc-server`:

```bash
./ggml-rpc-server -H 0.0.0.0 -p 50052
```

### Step 2: Start Host Server (PS5 #1)

On the primary PS5 node (`192.168.1.101`), launch `llama-server`, offloading layers to the worker node over TCP RPC:

```bash
./llama-server \
  -m ~/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf \
  --rpc 192.168.1.102:50052 \
  -ngl all \
  -c 64000 \
  -np 2 \
  -ctk q8_0 -ctv q8_0 \
  --temp 0.7 \
  --top-p 0.80 \
  --top-k 20 \
  -fa \
  -t 4 -tb 8 \
  -b 2048 -ub 512 \
  --port 8080 \
  --host 0.0.0.0
```

## 📐 KV Cache & Context Sizing Math

On a 27B dense model (`Q4_K_M`), weights take up **\~8.5 GB** per node. With the 14.1 GB usable unified memory cap per PS5:

$$
\text{Usable KV Headroom} = 14.1\text{ GB (Max Usable)} - 8.5\text{ GB (Model Weights)} = \mathbf{5.6\text{ GB per node}}
$$

* **8-bit KV Quantization (`-ctk q8_0 -ctv q8_0`):** Reduces KV cache footprint to **\~64 KB per token** per node.
* **Context Capacity:** A total context size of `-c 64000` split across `-np 2` parallel slots yields **32,000 context tokens per slot** safely inside the 14.1 GB physical ceiling.

## 🔍 Troubleshooting & Observations

* **Network MTU:** Do not set MTU 9000 (Jumbo Frames). The PS5 network driver imposes a hard `maxmtu` limit of 1500. Standard MTU 1500 switched over MikroTik L2 HW-Offloading yields sub-millisecond layer transfer times without driver errors.
* **GPU Utilization Tools:** Tools like `radeontop` or `rocm-smi` may report low compute activity (15%–35%) during generation. This is normal; token generation is strictly memory-bandwidth bound ($\approx 448\text{ GB/s}$ bus saturation) rather than CU compute bound.
* **Unused Tensor Warnings:** Warnings regarding unused block tensors (e.g., `blk.64.attn_norm.weight`) during model initialization are harmless artifacts from pruned/modified GGUF variants and can be ignored.

## 🤝 Acknowledgments

* [soniciso1/ps5-linux-loader](https://github.com/soniciso1/ps5-linux-loader) for enabling Linux execution on PS5 hardware.
* The [llama.cpp](https://github.com/ggml-org/llama.cpp) team for building Vulkan and GGML RPC backends.
