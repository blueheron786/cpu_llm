// src/utils.rs
use rand::Rng;

/// Fully connected layer: output = weights * input + bias
pub fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(weights.len());
    for (i, row) in weights.iter().enumerate() {
        if i >= bias.len() { break; }
        let mut sum = 0.0;
        for (w, x) in row.iter().zip(input.iter()) {
            sum += w * x;
        }
        out.push(sum + bias[i]);
    }
    out
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
