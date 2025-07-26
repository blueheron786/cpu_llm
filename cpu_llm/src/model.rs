use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use rayon::prelude::*;

// Utility functions
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum_exp).collect()
}

fn sample(probs: &[f32]) -> usize {
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

#[derive(Serialize, Deserialize)]
pub struct TinyRnnModel {
    // Vocabulary
    pub vocab: Vec<String>,
    pub stoi: HashMap<String, usize>,
    pub itos: Vec<String>,
    pub context_size: usize,
    pub hidden_size: usize,
    pub embedding: Vec<Vec<f32>>,
    pub ff1_weights: Vec<Vec<f32>>,
    pub ff1_bias: Vec<f32>,
    pub ff2_weights: Vec<Vec<f32>>,
    pub ff2_bias: Vec<f32>,
    // Buffers for training
    temp_buffers: [Vec<f32>; 4],
}

impl TinyRnnModel {
    pub fn new(
        vocab: Vec<String>,
        context_size: usize,
        hidden_size: usize,
    ) -> Self {
        let stoi: HashMap<_, _> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let itos = vocab.clone();

        // Initialize with small random values using He initialization
        let bound = (6.0 / (vocab.len() + hidden_size) as f32).sqrt();
        let dist = Uniform::new(-bound, bound);

        // Initialize embedding layer with parallel collection
        let embedding: Vec<Vec<f32>> = (0..vocab.len())
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                (0..hidden_size).map(|_| dist.sample(&mut rng)).collect()
            })
            .collect();

        // Initialize first feed-forward layer
        let ff1_weights: Vec<Vec<f32>> = (0..hidden_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                (0..hidden_size).map(|_| dist.sample(&mut rng)).collect()
            })
            .collect();
        
        let ff1_bias = vec![0.0; hidden_size];

        // Initialize second feed-forward layer
        let ff2_weights: Vec<Vec<f32>> = (0..vocab.len())
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                (0..hidden_size).map(|_| dist.sample(&mut rng)).collect()
            })
            .collect();
            
        let ff2_bias = vec![0.0; vocab.len()];
        
        // Initialize reusable buffers for training
        let temp_buffers = [
            vec![0.0; hidden_size],    // h
vec![0.0; vocab.len()],     // logits
            vec![0.0; hidden_size],    // grad_h
            vec![0.0; hidden_size],    // grad_ff1
        ];

        Self {
            vocab,
            stoi,
            itos,
            context_size,
            hidden_size,
            embedding,
            ff1_weights,
            ff1_bias,
            ff2_weights,
            ff2_bias,
            temp_buffers,
        }
    }

    #[inline(always)]
    pub fn forward_buffered(&self, context: &[usize], h_out: &mut [f32], logits_out: &mut [f32]) {
        // Average the embeddings for the context
        h_out.par_iter_mut().enumerate().for_each(|(i, h)| {
            let mut sum = 0.0f32;
            for &id in context {
                sum += self.embedding[id][i];
            }
            *h = sum / context.len() as f32;
        });

        // First feed-forward layer with ReLU
        self.linear_inplace(
            h_out,
            &self.ff1_weights,
            &self.ff1_bias,
            logits_out,
        );

        // ReLU activation
        logits_out.par_iter_mut().for_each(|x| *x = x.max(0.0));

        // Second feed-forward layer
        self.linear_inplace(
            logits_out,
            &self.ff2_weights,
            &self.ff2_bias,
            h_out,
        );
        
        // Copy to logits_out for the final output
        logits_out.copy_from_slice(h_out);
    }
    
    fn linear_inplace(&self, input: &[f32], weights: &[Vec<f32>], bias: &[f32], output: &mut [f32]) {
        // Debug information
        println!("linear_inplace: input.len() = {}, weights.len() = {}, bias.len() = {}, output.len() = {}", 
                 input.len(), weights.len(), bias.len(), output.len());
        
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            if i >= weights.len() {
                println!("Index out of bounds: i = {}, weights.len() = {}", i, weights.len());
                return;
            }
            let row = &weights[i];
            let mut sum = 0.0f32;
            for (j, &w) in row.iter().enumerate() {
                if j >= input.len() {
                    println!("Index out of bounds in row access: j = {}, input.len() = {}", j, input.len());
                    break;
                }
                sum += input[j] * w;
            }
            if i < bias.len() {
                *out = sum + bias[i];
            } else {
                println!("Index out of bounds for bias: i = {}, bias.len() = {}", i, bias.len());
                *out = sum;
            }
        });
    }
}

pub fn run_inference(model: &TinyRnnModel, prompt: &str, num_tokens: usize) -> String {
    let mut output = String::new();
    let mut context: Vec<usize> = prompt
        .chars()
        .filter_map(|c| model.vocab.iter().position(|v| v == &c.to_string()))
        .collect();

    let mut h = vec![0.0; model.hidden_size];
    let mut logits = vec![0.0; model.vocab.len()];

    for _ in 0..num_tokens {
        // Take last CONTEXT_SIZE tokens (or fewer)
        let start = if context.len() > model.context_size {
            context.len() - model.context_size
        } else {
            0
        };
        let context_window = &context[start..];

        model.forward_buffered(context_window, &mut h, &mut logits);

        // Sample next token
        let probs = softmax(&logits);
        let next_token = sample(&probs);
        
        // Convert token to string and add to output
        if let Some(c) = model.vocab.get(next_token) {
            output.push_str(c);
            context.push(next_token);
        }
    }

    output
}
