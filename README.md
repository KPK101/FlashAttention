# FlashAttention
Implementation of FlashAttention in pycuda

## Status
- Simple attention mechanism implementation in python using numpy
---
### To-Do (CPU): Target (11/05)
- Include multi-headed attention
- More modular and checks
- Make a PyTorch attention module

---

### To-Do (GPU): Target (11/05)
- Implement naive attention computation
- Add tiling to blocks for compute
- Fused kernels (matmul, softmax, linear layer)
---
