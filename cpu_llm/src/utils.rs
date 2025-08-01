// src/utils.rs
use rand::Rng;
use rayon::prelude::*;

/// Fully connected layer: output = weights * input + bias (parallelized)
pub fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; weights.len()];
    
    // Use parallel processing for larger matrices
    if weights.len() > 32 {
        out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
            if i < bias.len() && i < weights.len() {
                let row = &weights[i];
                *out_i = bias[i] + input.par_iter().zip(row.par_iter())
                    .map(|(&x, &w)| x * w)
                    .sum::<f32>();
            }
        });
    } else {
        // Sequential for small matrices to avoid overhead
        for (i, row) in weights.iter().enumerate() {
            if i >= bias.len() { break; }
            let mut sum = 0.0;
            for (w, x) in row.iter().zip(input.iter()) {
                sum += w * x;
            }
            out[i] = sum + bias[i];
        }
    }
    
    out
}

/// ReLU activation function (parallelized)
pub fn relu(vec: Vec<f32>) -> Vec<f32> {
    if vec.len() > 64 {
        vec.into_par_iter().map(|x| x.max(0.0)).collect()
    } else {
        vec.into_iter().map(|x| x.max(0.0)).collect()
    }
}

/// Softmax output (parallelized)
pub fn softmax(vec: Vec<f32>) -> Vec<f32> {
    if vec.len() > 64 {
        let max = vec.par_iter().copied().reduce(|| f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = vec.par_iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.par_iter().sum();
        exps.into_par_iter().map(|x| x / sum).collect()
    } else {
        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = vec.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.into_iter().map(|x| x / sum).collect()
    }
}

/// Sample index from a probability distribution
pub fn sample(probs: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut acc = 0.0;
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
