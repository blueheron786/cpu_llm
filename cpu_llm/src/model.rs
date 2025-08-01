use rand::Rng;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::utils::sample;
use rayon::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct TinyRnnModel {
    // Vocabulary
    pub vocab: Vec<char>,                   // List of chars in vocab
    pub stoi: HashMap<char, usize>,         // char -> index
    pub itos: Vec<char>,                    // index -> char (same as vocab)
    pub vocab_size: usize,

    // Model hyperparameters
    pub context_size: usize,
    pub hidden_size: usize,

    // Model parameters
    pub embedding: Vec<Vec<f32>>,          // vocab_size x hidden_size
    pub ff1_weights: Vec<Vec<f32>>,        // hidden_size x hidden_size
    pub ff1_bias: Vec<f32>,                 // hidden_size
    pub ff2_weights: Vec<Vec<f32>>,        // hidden_size x vocab_size
    pub ff2_bias: Vec<f32>,                 // vocab_size
}

impl TinyRnnModel {
    pub fn new(vocab: Vec<char>) -> Self {
        let vocab_size = vocab.len();
        let context_size = 16;
        let hidden_size = 128; // Increase hidden size for more capacity

        // Build stoi and itos
        let mut stoi = HashMap::new();
        for (i, &ch) in vocab.iter().enumerate() {
            stoi.insert(ch, i);
        }
        let itos = vocab.clone();

        // Initialize weights with small random values
        fn rand_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
            let mut rng = rand::thread_rng();
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-0.1..0.1)).collect())
                .collect()
        }

        let embedding = rand_matrix(vocab_size, hidden_size);
        let ff1_weights = rand_matrix(hidden_size, hidden_size);
        let ff1_bias = vec![0.0; hidden_size];
        let ff2_weights = rand_matrix(hidden_size, vocab_size);
        let ff2_bias = vec![0.0; vocab_size];

        TinyRnnModel {
            vocab,
            stoi,
            itos,
            vocab_size,
            context_size,
            hidden_size,
            embedding,
            ff1_weights,
            ff1_bias,
            ff2_weights,
            ff2_bias,
        }
    }
}

pub fn run_inference(prompt: &str, num_tokens: usize, model: &TinyRnnModel) -> String {
    let _stoi = &model.stoi;
    let itos = &model.itos;

    // Convert prompt chars to token IDs
    let mut context: Vec<usize> = prompt.chars()
        .filter_map(|c| model.vocab.iter().position(|&vc| vc == c))
        .collect();

    let mut output = prompt.to_string();

    // Pre-allocate buffers to avoid repeated allocations
    let mut avg_embed = vec![0.0; model.hidden_size];
    let mut h_buffer = vec![0.0; model.hidden_size];
    let mut logits_buffer = vec![0.0; model.vocab_size];
    let mut probs_buffer = vec![0.0; model.vocab_size];

    for _ in 0..num_tokens {
        // Take last CONTEXT_SIZE tokens (or fewer)
        let recent: Vec<usize> = context
            .iter()
            .rev()
            .take(model.context_size)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        // Parallel embedding averaging
        avg_embed.par_iter_mut().enumerate().for_each(|(i, embed_i)| {
            *embed_i = recent.iter()
                .filter_map(|&id| model.embedding.get(id))
                .map(|embed| embed[i])
                .sum::<f32>() / recent.len().max(1) as f32;
        });

        // Parallel forward pass: ReLU(Linear) → Linear → Softmax
        parallel_linear(&avg_embed, &model.ff1_weights, &model.ff1_bias, &mut h_buffer);
        h_buffer.par_iter_mut().for_each(|x| *x = x.max(0.0)); // Parallel ReLU
        
        parallel_linear(&h_buffer, &model.ff2_weights, &model.ff2_bias, &mut logits_buffer);
        parallel_softmax(&logits_buffer, &mut probs_buffer);

        let next_id = sample(&probs_buffer);
        let next_char = itos.get(next_id).copied().unwrap_or('?');

        output.push(next_char);
        context.push(next_id);
    }

    output
}

/// Batch inference for maximum parallelization - processes multiple prompts simultaneously
pub fn run_batch_inference(prompts: &[&str], num_tokens: usize, model: &TinyRnnModel) -> Vec<String> {
    prompts.par_iter()
        .map(|&prompt| run_inference(prompt, num_tokens, model))
        .collect()
}

/// Parallel linear transformation: output = weights * input + bias
fn parallel_linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32], output: &mut [f32]) {
    // Initialize with bias in parallel
    output.par_iter_mut().zip(bias.par_iter()).for_each(|(o, &b)| *o = b);
    
    // Parallel matrix multiplication
    output.par_iter_mut().enumerate().for_each(|(j, out_j)| {
        *out_j += input.par_iter().enumerate()
            .map(|(i, &x)| x * weights.get(i).and_then(|row| row.get(j)).unwrap_or(&0.0))
            .sum::<f32>();
    });
}

/// Parallel softmax operation
fn parallel_softmax(logits: &[f32], output: &mut [f32]) {
    let max_logit = logits.par_iter().copied().reduce(|| f32::NEG_INFINITY, f32::max);
    
    // Parallel exp computation
    output.par_iter_mut().zip(logits.par_iter()).for_each(|(out, &logit)| {
        *out = (logit - max_logit).exp();
    });
    
    let sum: f32 = output.par_iter().sum();
    let sum = sum.max(1e-8);
    
    // Parallel normalization
    output.par_iter_mut().for_each(|x| *x /= sum);
}
