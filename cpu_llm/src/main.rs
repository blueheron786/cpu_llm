use std::collections::HashMap;
use rand::Rng;
use rayon::prelude::*;

// ==== Hyperparameters ====
const CONTEXT_SIZE: usize = 16;
const HIDDEN_SIZE: usize = 32;
const VOCAB: &str = "abcdefghijklmnopqrstuvwxyz .,!?";

// ==== Tokenizer ====
fn build_vocab() -> (HashMap<char, usize>, Vec<char>) {
    let mut stoi = HashMap::new();
    let mut itos = Vec::new();
    for (i, ch) in VOCAB.chars().enumerate() {
        stoi.insert(ch, i);
        itos.push(ch);
    }
    (stoi, itos)
}

// ==== Random matrix initialization ====
fn rand_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-0.1..0.1)).collect())
        .collect()
}

// ==== Multi-threaded linear layer ====
fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    weights
        .par_iter()
        .enumerate()
        .map(|(i, row)| {
            // safe zip, length must match input len
            row.iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>()
                + bias.get(i).copied().unwrap_or(0.0)
        })
        .collect()
}

// ==== ReLU activation ====
fn relu(vec: Vec<f32>) -> Vec<f32> {
    vec.into_iter().map(|x| x.max(0.0)).collect()
}

// ==== Softmax activation ====
fn softmax(vec: Vec<f32>) -> Vec<f32> {
    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = vec.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum).collect()
}

// ==== Sample from probabilities ====
fn sample(probs: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let r = rng.r#gen::<f32>();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

// ==== Run inference ====
fn run_inference(prompt: &str, num_tokens: usize) -> String {
    let (stoi, itos) = build_vocab();

    let embedding = rand_matrix(VOCAB.len(), HIDDEN_SIZE);
    let ff1_w = rand_matrix(HIDDEN_SIZE, HIDDEN_SIZE);
    let ff1_b = vec![0.0; HIDDEN_SIZE];
    let ff2_w = rand_matrix(HIDDEN_SIZE, VOCAB.len());
    let ff2_b = vec![0.0; VOCAB.len()];

    // Convert prompt chars to indices, ignoring unknown chars
    let mut context: Vec<usize> = prompt
        .chars()
        .filter_map(|c| stoi.get(&c).copied())
        .collect();

    let mut output = prompt.to_string();

    for _ in 0..num_tokens {
        // Take last CONTEXT_SIZE tokens (or fewer)
        let recent: Vec<usize> = context
            .iter()
            .rev()
            .take(CONTEXT_SIZE)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        // Average embeddings of recent tokens
        let mut avg_embed = vec![0.0; HIDDEN_SIZE];
        for &id in &recent {
            if let Some(embedding_vec) = embedding.get(id) {
                for i in 0..HIDDEN_SIZE {
                    avg_embed[i] += embedding_vec[i];
                }
            }
        }
        let n = recent.len().max(1) as f32; // prevent div by zero
        for i in 0..HIDDEN_SIZE {
            avg_embed[i] /= n;
        }

        // Forward pass
        let h = relu(linear(&avg_embed, &ff1_w, &ff1_b));
        let logits = linear(&h, &ff2_w, &ff2_b);
        let probs = softmax(logits);

        // Sample next token id
        let next_id = sample(&probs);
        let next_char = itos
            .get(next_id)
            .copied()
            .unwrap_or('?'); // fallback if out of range (should not happen)

        output.push(next_char);
        context.push(next_id);
    }

    output
}

// ==== Main entry point ====
fn main() {
    let prompt = "It was a dark and stormy night. ";
    let generated = run_inference(prompt, 100);
    println!("Generated: {}", generated);
}
