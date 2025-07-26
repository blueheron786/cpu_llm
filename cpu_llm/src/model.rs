use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use rayon::prelude::*;

// Helper trait for splitting words at punctuation
trait WordSplit {
    fn split_at_word_end(&self) -> (&str, &str);
}

impl WordSplit for &str {
    fn split_at_word_end(&self) -> (&str, &str) {
        let mut split_idx = self.len();
        for (i, c) in self.char_indices().rev() {
            if is_punct(c) {
                split_idx = i;
            } else {
                break;
            }
        }
        self.split_at(split_idx)
    }
}

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
            vec![0.0; vocab.len()],    // logits
            vec![0.0; hidden_size],    // grad_h
            vec![0.0; hidden_size],    // grad_ff1
        ];

        TinyRnnModel {
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
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            let row = &weights[i];
            let mut sum = 0.0f32;
            for (j, &w) in row.iter().enumerate() {
                sum += input[j] * w;
            }
            *out = sum + bias[i];
        });
    }
}

pub fn run_inference(model: &TinyRnnModel, prompt: &str, num_tokens: usize) -> String {
    let mut output_tokens = Vec::new();
    let mut context: Vec<usize> = prompt
        .split_whitespace()
        .flat_map(|word| {
            // Decompose into vocab tokens
            let trimmed = word.trim_end_matches(|c: char| is_punct(c));
            let (core, punct) = trimmed.split_at_word_end();
            let mut ids = Vec::new();

            if let Some(i) = model.vocab.iter().position(|v| v == core) {
                ids.push(i);
            } else {
                ids.extend(core.chars().filter_map(|c| model.vocab.iter().position(|v| v == &c.to_string())));
            }

            for ch in punct.chars() {
                if let Some(i) = model.vocab.iter().position(|v| v == &ch.to_string()) {
                    ids.push(i);
                }
            }

            ids
        })
        .collect();

    let mut h = vec![0.0; model.hidden_size];
    let mut logits = vec![0.0; model.vocab.len()];

    for _ in 0..num_tokens {
        let start = if context.len() > model.context_size {
            context.len() - model.context_size
        } else {
            0
        };
        let context_window = &context[start..];

        model.forward_buffered(context_window, &mut h, &mut logits);

        let probs = softmax(&logits);
        let next_token = sample(&probs);

        if let Some(token_str) = model.vocab.get(next_token) {
            output_tokens.push(token_str.clone());
            context.push(next_token);
        }
    }

    detokenize(&output_tokens)
}

/// Converts a sequence of tokens (words and chars) to a final string.
pub fn detokenize(tokens: &[String]) -> String {
    let mut out = String::new();
    let mut last_was_word = false;

    for tok in tokens {
        if tok.len() == 1 && is_punct(tok.chars().next().unwrap()) {
            // Punctuation directly after previous word/char
            out.push_str(tok);
            last_was_word = false;
        } else {
            if last_was_word {
                out.push(' ');
            }
            out.push_str(tok);
            last_was_word = true;
        }
    }

    out
}

/// Returns true if the character is a punctuation mark
pub fn is_punct(c: char) -> bool {
    ",.!?;:'\"()[]{}".contains(c)
}
