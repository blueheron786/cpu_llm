use rand::Rng;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::utils::{relu, softmax, linear, sample};

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
        let hidden_size = 32;

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
    let stoi = &model.stoi;
    let itos = &model.itos;

    // Convert prompt chars to token IDs
    let mut context: Vec<usize> = prompt.chars()
        .filter_map(|c| model.vocab.iter().position(|&vc| vc == c))
        .collect();

    let mut output = prompt.to_string();

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

        // Average embeddings
        let mut avg_embed = vec![0.0; model.hidden_size];
        for &id in &recent {
            if let Some(embed) = model.embedding.get(id) {
                for i in 0..model.hidden_size {
                    avg_embed[i] += embed[i];
                }
            }
        }

        let n = recent.len().max(1) as f32;
        for i in 0..model.hidden_size {
            avg_embed[i] /= n;
        }

        // Forward pass: ReLU(Linear) → Linear → Softmax
        let h = relu(linear(&avg_embed, &model.ff1_weights, &model.ff1_bias));
        let logits = linear(&h, &model.ff2_weights, &model.ff2_bias);
        let probs = softmax(logits);

        let next_id = sample(&probs);
        let next_char = itos.get(next_id).copied().unwrap_or('?');

        output.push(next_char);
        context.push(next_id);
    }

    output
}
