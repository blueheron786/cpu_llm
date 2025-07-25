# Optimization Plan for cpu_llm Training

## 5. Custom Matrix Libraries
- Replace nested `Vec<Vec<f32>>` with flat arrays (`Vec<f32>`) for better cache locality and SIMD compatibility.
- Use libraries like `ndarray` or `nalgebra` for efficient matrix math.
- Measure training speed and memory usage.

## 6. Profile and Tune Hotspots
- Use `cargo-flamegraph` or `perf` to profile code and focus optimization on bottlenecks.
- Document findings and improvements.

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
