A custom Deep Learning neural architecture implemented from scratch in pure C.

## Overview
- Low-level implementation without external deep learning libraries.
- Supervised learning with periodic activation functions and phase-shift dynamics.
- Optimized memory layout and cache-friendly operations.

## Build and Run
```bash
gcc -O3 main/*.c -o sinenet -lm
./sinenet