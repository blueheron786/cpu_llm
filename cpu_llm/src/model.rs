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
    // Output projection
    pub output_weights: Vec<Vec<f32>>,
    pub output_bias: Vec<f32>,
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
        let mut ff1_weights = vec![vec![0.0; hidden_size]; hidden_size];
        let mut ff1_bias = vec![0.0; hidden_size];
        let mut ff2_weights = vec![vec![0.0; hidden_size]; hidden_size];
        let mut ff2_bias = vec![0.0; hidden_size];
        
        // Initialize output projection
        let vocab_size = vocab.len();
        let mut output_weights = vec![vec![0.0; vocab_size]; hidden_size];
        let mut output_bias = vec![0.0; vocab_size];
        
        // Initialize with small random values
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-0.1, 0.1);
        
        // Initialize feed-forward weights
        for i in 0..hidden_size {
            for j in 0..hidden_size {
                ff1_weights[i][j] = uniform.sample(&mut rng);
                ff2_weights[i][j] = uniform.sample(&mut rng);
            }
            ff1_bias[i] = 0.0;  // Initialize biases to zero
            ff2_bias[i] = 0.0;
            
            // Initialize output projection weights
            for j in 0..vocab_size {
                output_weights[i][j] = uniform.sample(&mut rng);
            }
        }
        
        // Initialize output bias to zero
        for i in 0..vocab_size {
            output_bias[i] = 0.0;
        }

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
            output_weights,
            output_bias,
            temp_buffers,
        }
    }

    #[inline(always)]
    pub fn forward_buffered(&self, context: &[usize], h_out: &mut [f32], logits_out: &mut [f32]) {
        // Validate buffer sizes
        assert_eq!(h_out.len(), self.hidden_size, "h_out buffer has incorrect size");
        assert_eq!(logits_out.len(), self.vocab.len(), "logits_out buffer must match vocabulary size");
        
        // Average the embeddings for the context
        h_out.par_iter_mut().enumerate().for_each(|(i, h)| {
            let mut sum = 0.0f32;
            for &id in context {
                sum += self.embedding[id][i];
            }
            *h = sum / context.len() as f32;
        });

        // Temporary buffer for intermediate results
        let mut temp = vec![0.0f32; self.hidden_size];

        // First feed-forward layer with ReLU
        {
            let flat_weights: Vec<f32> = self.ff1_weights.iter().flat_map(|v| v.iter().copied()).collect();
            self.linear_inplace(
                h_out,
                &flat_weights,
                &self.ff1_bias,
                &mut temp,
            );
        }

        // ReLU activation
        temp.par_iter_mut().for_each(|x| *x = x.max(0.0));

        // Second feed-forward layer
        {
            let flat_weights: Vec<f32> = self.ff2_weights.iter().flat_map(|v| v.iter().copied()).collect();
            self.linear_inplace(
                &temp,
                &flat_weights,
                &self.ff2_bias,
                h_out,
            );
        }
        
        // Apply output projection (hidden_size -> vocab_size)
        logits_out.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut sum = 0.0;
            for j in 0..self.hidden_size {
                sum += h_out[j] * self.output_weights[j][i];
            }
            *out = sum + self.output_bias[i];
        });
    }
    
    fn linear_inplace(&self, input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32]) {
        // Debug information
        // println!("linear_inplace: input.len() = {}, weights.len() = {}, bias.len() = {}, output.len() = {}", 
        //          input.len(), weights.len(), bias.len(), output.len());
        
        // Ensure output length matches bias length
        assert_eq!(output.len(), bias.len(), "Output length must match bias length");
        
        // Calculate output dimensions
        let input_size = input.len();
        let output_size = output.len();
        
        // Check weight matrix dimensions (should be input_size × output_size)
        assert_eq!(
            weights.len(), 
            input_size * output_size, 
            "Weights length must be input_size * output_size"
        );
        
        // Parallel matrix-vector multiplication
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            // Calculate the dot product of input and weight column
            let mut sum = 0.0f32;
            for j in 0..input_size {
                let weight_idx = i * input_size + j;
                sum += input[j] * weights[weight_idx];
            }
            
            // Add bias and store result
            if i < bias.len() {
                *out = sum + bias[i];
            } else {
                println!("Warning: Bias index {} out of bounds (bias.len() = {}). Using sum without bias.", 
                         i, bias.len());
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
