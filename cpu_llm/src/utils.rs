// src/utils.rs
use rand::Rng;

// Constants for loop unrolling and cache line size
const CHUNK_SIZE: usize = 8; // Process 8 elements at a time for better cache utilization
const CACHE_LINE_SIZE: usize = 64; // Typical cache line size in bytes

// Helper function to compute dot product with auto-vectorization
#[inline(always)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
    let mut sum = 0.0f32;
    
    // Process in chunks for better cache utilization
    for i in (0..a.len()).step_by(CHUNK_SIZE) {
        let end = (i + CHUNK_SIZE).min(a.len());
        let mut chunk_sum = 0.0;
        
        // This loop will be auto-vectorized by the compiler
        for j in i..end {
            chunk_sum += a[j] * b[j];
        }
        
        sum += chunk_sum;
    }
    
    sum
}

/// Fully connected layer: output = weights * input + bias
/// Optimized for auto-vectorization on stable Rust
pub fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    weights.iter()
        .take(bias.len())
        .enumerate()
        .map(|(i, row)| dot_product(input, row) + bias[i])
        .collect()
}

/// ReLU activation function optimized for auto-vectorization
pub fn relu(mut vec: Vec<f32>) -> Vec<f32> {
    // This loop will be auto-vectorized by the compiler
    for x in &mut vec {
        *x = x.max(0.0);
    }
    vec
}

/// Softmax output optimized for auto-vectorization
pub fn softmax(mut vec: Vec<f32>) -> Vec<f32> {
    // Find max value
    let max = vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate exps and sum
    let mut sum = 0.0;
    
    // First pass: calculate exps and sum
    for x in &mut vec {
        let exp = (*x - max).exp();
        *x = exp;
        sum += exp;
    }
    
    // Second pass: normalize
    let inv_sum = 1.0 / sum;
    for x in &mut vec {
        *x *= inv_sum;
    }
    
    vec
}

/// Sample index from a probability distribution
pub fn sample(probs: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut acc = 0.0;
    
    // This part is not easily parallelized due to dependency on previous iterations
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i;
        }
    }
    
    // If not found due to rounding, return last valid index
    if !probs.is_empty() {
        probs.len() - 1
    } else {
        0
    }
}
