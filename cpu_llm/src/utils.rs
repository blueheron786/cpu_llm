// src/utils.rs
use rand::Rng;
use rayon::prelude::*;

/// Fully connected layer: output = weights * input + bias
pub fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    weights
        .par_iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter().zip(input.iter()).map(|(w, x)| w * x).sum::<f32>() + bias[i]
        })
        .collect()
}

/// ReLU activation function
pub fn relu(vec: Vec<f32>) -> Vec<f32> {
    vec.into_iter().map(|x| x.max(0.0)).collect()
}

/// Softmax output
pub fn softmax(vec: Vec<f32>) -> Vec<f32> {
    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = vec.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum).collect()
}

/// Sample index from a probability distribution
pub fn sample(probs: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let mut r: f32 = rng.gen();
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}
