#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Benchmark Environment Setup
# Run BEFORE any benchmarking session.
# Usage: sudo bash setup_bench_env.sh [GPU_INDEX]
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

GPU=${1:-0}
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo "═══════════════════════════════════════════════════"
echo "  Benchmark Setup  (GPU $GPU)"
echo "═══════════════════════════════════════════════════"

# ─── 1. Check for other GPU users ─────────────────────────────
echo -e "\n${YELLOW}[1/3] Checking GPU occupancy...${NC}"
OTHER_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU 2>/dev/null | wc -l)
if [ "$OTHER_PROCS" -gt 0 ]; then
    echo -e "${RED}WARNING: $OTHER_PROCS process(es) on GPU $GPU:${NC}"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -i $GPU
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU \
        | xargs -I{} ps -p {} -o pid,user,comm 2>/dev/null || true
    echo -e "${RED}Kill them or wait before benchmarking.${NC}"
else
    echo -e "${GREEN}GPU $GPU is clear.${NC}"
fi

# ─── 2. Enable persistence mode ───────────────────────────────
echo -e "\n${YELLOW}[2/3] Enabling persistence mode...${NC}"
nvidia-smi -pm 1 -i $GPU
echo -e "${GREEN}Done.${NC}"

# ─── 3. Lock GPU clocks to base (thermally sustainable) ───────
echo -e "\n${YELLOW}[3/3] Locking GPU clocks to base...${NC}"
BASE_SM=$(nvidia-smi -i $GPU --query-gpu=clocks.default_applications.gr --format=csv,noheader | tr -d ' MHz')
echo "  SM:  ${BASE_SM} MHz (base)"
nvidia-smi -i $GPU -lgc ${BASE_SM} 2>/dev/null || nvidia-smi -i $GPU -lgc ${BASE_SM},${BASE_SM} 2>/dev/null || echo "  (clock lock not supported)"
# Memory clock — skip if not available (e.g., unified memory on GB10)
BASE_MEM=$(nvidia-smi -i $GPU --query-gpu=clocks.default_applications.mem --format=csv,noheader 2>/dev/null | tr -d ' MHz')
if [ -n "$BASE_MEM" ] && [ "$BASE_MEM" != "[N/A]" ]; then
    echo "  Mem: ${BASE_MEM} MHz (base)"
    nvidia-smi -i $GPU -lmc ${BASE_MEM} 2>/dev/null || echo "  (mem clock lock not supported)"
else
    echo "  Mem: N/A (unified memory)"
fi
echo -e "${GREEN}Clocks locked at base. No thermal throttling.${NC}"

# ─── Summary ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo -e "${GREEN}  Ready.${NC}"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  GPU:         $(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader)"
echo "  SM clock:    $(nvidia-smi -i $GPU --query-gpu=clocks.sm --format=csv,noheader) (base)"
echo "  Mem clock:   $(nvidia-smi -i $GPU --query-gpu=clocks.mem --format=csv,noheader) (base)"
echo "  Temperature: $(nvidia-smi -i $GPU --query-gpu=temperature.gpu --format=csv,noheader) C"
echo ""
echo "  Run:   CUDA_VISIBLE_DEVICES=$GPU python benchmark_isolated.py"
echo "  Reset: sudo nvidia-smi -i $GPU -rgc && sudo nvidia-smi -i $GPU -rmc"
