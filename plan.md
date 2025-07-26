# Optimization Plan for cpu_llm Training

## 7. Asynchronous Data Loading
- Overlap data loading and preprocessing with training using threads or async tasks.
- Measure data loading and training overlap efficiency.

## 8. Reduce Synchronization
- Minimize locking and synchronization in parallel code, especially in gradient reduction.
- Measure parallel efficiency.

## 9. Use BLAS/LAPACK
- For large matrix operations, use BLAS libraries via crates like `blas`, `openblas-src`, or `ndarray-linalg`.
- Measure matrix operation speed before and after.

## 10. Model Architecture Tweaks
- Reduce model size, use more efficient activation functions, or prune rarely used tokens.
- Measure training speed and model accuracy.

---

**For each optimization:**
- Benchmark training speed and memory usage before and after.
- Validate correctness and stability.
- Document results in this file.
