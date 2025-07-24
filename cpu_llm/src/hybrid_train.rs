use glob::glob;
use rayon::prelude::*;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::Instant;

// Hyperparameters
const CONTEXT_SIZE: usize = 8;
const HIDDEN_SIZE: usize = 64;
const EPOCHS: usize = 5;
const BATCH_SIZE: usize = 256;
const LR: f32 = 0.05;

// Hybrid tokenizer tokens: either full word or char fallback
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Token {
    Word(String),
    Char(char),
}

impl Token {
    fn as_str(&self) -> String {
        match self {
            Token::Word(w) => w.clone(),
            Token::Char(c) => c.to_string(),
        }
    }
}

// Clean text by normalizing spaces and lowercasing
fn clean_text(text: &str) -> String {
    let s = text.replace('\n', " ").replace('\r', " ");
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    s.to_lowercase()
}

// Build word vocab for words with frequency >= min_freq
fn build_word_vocab(text: &str, min_freq: usize) -> HashSet<String> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        *freq.entry(word.to_string()).or_insert(0) += 1;
    }
    freq.into_iter()
        .filter(|(_, c)| *c >= min_freq)
        .map(|(w, _)| w)
        .collect()
}

// Tokenize text hybrid style: words if known, else chars
fn hybrid_tokenize(text: &str, word_vocab: &HashSet<String>) -> Vec<Token> {
    let mut tokens = Vec::new();
    for word in text.split_whitespace() {
        if word_vocab.contains(word) {
            tokens.push(Token::Word(word.to_string()));
        } else {
            for ch in word.chars() {
                tokens.push(Token::Char(ch));
            }
        }
    }
    tokens
}

struct TinyRnnModel {
    vocab: Vec<String>,
    stoi: HashMap<String, usize>,
    itos: Vec<String>,

    context_size: usize,
    hidden_size: usize,

    embedding: Vec<Vec<f32>>,      // vocab_size x hidden_size
    ff1_weights: Vec<Vec<f32>>,    // hidden_size x hidden_size
    ff1_bias: Vec<f32>,            // hidden_size
    ff2_weights: Vec<Vec<f32>>,    // hidden_size x vocab_size
    ff2_bias: Vec<f32>,            // vocab_size
}

impl TinyRnnModel {
    fn new(vocab: Vec<String>, context_size: usize, hidden_size: usize) -> Self {
        let vocab_size = vocab.len();
        let mut stoi = HashMap::new();
        for (i, w) in vocab.iter().enumerate() {
            stoi.insert(w.clone(), i);
        }
        let itos = vocab.clone();

        fn rand_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
            let mut rng = rand::thread_rng();
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-0.1..0.1)).collect())
                .collect()
        }

        TinyRnnModel {
            vocab,
            stoi,
            itos,
            context_size,
            hidden_size,
            embedding: rand_matrix(vocab_size, hidden_size),
            ff1_weights: rand_matrix(hidden_size, hidden_size),
            ff1_bias: vec![0.0; hidden_size],
            ff2_weights: rand_matrix(hidden_size, vocab_size),
            ff2_bias: vec![0.0; vocab_size],
        }
    }

    // Forward pass for one context (token IDs)
    fn forward(&self, context: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let mut avg_embed = vec![0.0; self.hidden_size];
        for &id in context {
            for i in 0..self.hidden_size {
                avg_embed[i] += self.embedding[id][i];
            }
        }
        for i in 0..self.hidden_size {
            avg_embed[i] /= context.len() as f32;
        }

        let h = relu(&linear(&avg_embed, &self.ff1_weights, &self.ff1_bias));
        let logits = linear(&h, &self.ff2_weights, &self.ff2_bias);
        (h, logits)
    }

    // Update weights with gradients and learning rate
    fn update_weights(
        &mut self,
        grad_w: &[Vec<f32>],
        grad_b: &[f32],
        lr: f32,
        batch_size: usize,
    ) {
        for j in 0..self.hidden_size {
            for k in 0..self.vocab.len() {
                self.ff2_weights[j][k] -= lr * grad_w[j][k] / batch_size as f32;
            }
        }
        for k in 0..self.vocab.len() {
            self.ff2_bias[k] -= lr * grad_b[k] / batch_size as f32;
        }
    }
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; bias.len()];
    for (i, row) in weights.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            output[j] += input[i] * w;
        }
    }
    for (o, b) in output.iter_mut().zip(bias.iter()) {
        *o += *b;
    }
    output
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (*l - max_logit).exp()).sum();
    logits.iter().map(|l| (*l - max_logit).exp() / exp_sum.max(1e-8)).collect()
}

fn main() {
    println!("📁 Loading and concatenating files from data/**/* ...");
    let mut combined_text = String::new();
    let mut file_count = 0;

    for entry in glob("data/**/*").expect("Failed to read glob pattern") {
        if let Ok(path) = entry {
            if path.is_file() {
                println!("📄 Loading {}", path.display());
                if let Ok(content) = fs::read_to_string(&path) {
                    combined_text.push_str(&content);
                    combined_text.push(' '); // separate files with space
                    file_count += 1;
                }
            }
        }
    }

    if file_count == 0 {
        eprintln!("❌ No files found");
        std::process::exit(1);
    }

    println!(
        "📚 Loaded {} files, total {} chars",
        file_count,
        combined_text.len()
    );

    println!("Cleaning text...");
    let clean = clean_text(&combined_text);

    println!("Building word vocab...");
    let word_vocab = build_word_vocab(&clean, 3);

    println!("Tokenizing hybrid...");
    let tokens = hybrid_tokenize(&clean, &word_vocab);

    println!("Building token vocab...");
    let mut vocab_set = HashSet::new();
    for t in &tokens {
        vocab_set.insert(t.as_str());
    }
    let mut vocab: Vec<String> = vocab_set.into_iter().collect();
    vocab.sort();

    println!("Vocab size: {}", vocab.len());

    let mut model = TinyRnnModel::new(vocab.clone(), CONTEXT_SIZE, HIDDEN_SIZE);

    // Convert tokens to IDs
    let token_ids: Vec<usize> = tokens
        .iter()
        .filter_map(|t| model.stoi.get(&t.as_str()).copied())
        .collect();

    println!("Training for {} epochs...", EPOCHS);

    for epoch in 0..EPOCHS {
        println!("🚀 Epoch {}/{}", epoch + 1, EPOCHS);
        let start = Instant::now();
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for idx in (CONTEXT_SIZE..token_ids.len() - 1).step_by(BATCH_SIZE) {
            let end = (idx + BATCH_SIZE).min(token_ids.len() - 1);

            // Parallel batch processing
            let (grad_w_sum, grad_b_sum, loss_sum) = (idx..end)
                .into_par_iter()
                .map(|i| {
                    let mut grad_w = vec![vec![0.0; model.vocab.len()]; model.hidden_size];
                    let mut grad_b = vec![0.0; model.vocab.len()];
                    let mut loss = 0.0_f32;

                    let context = &token_ids[i - CONTEXT_SIZE..i];
                    let target = token_ids[i];

                    let (h, logits) = model.forward(context);
                    let probs = softmax(&logits);

                    let safe_prob = probs[target].max(1e-8);
                    loss += -safe_prob.ln();

                    let mut dlogits = probs.clone();
                    dlogits[target] -= 1.0;

                    for j in 0..model.hidden_size {
                        for k in 0..model.vocab.len() {
                            grad_w[j][k] += dlogits[k] * h[j];
                        }
                    }
                    for k in 0..model.vocab.len() {
                        grad_b[k] += dlogits[k];
                    }

                    (grad_w, grad_b, loss)
                })
                .reduce(
                    || (
                        vec![vec![0.0; model.vocab.len()]; model.hidden_size],
                        vec![0.0; model.vocab.len()],
                        0.0_f32,
                    ),
                    |(mut acc_w, mut acc_b, acc_loss), (grad_w, grad_b, loss)| {
                        for j in 0..model.hidden_size {
                            for k in 0..model.vocab.len() {
                                acc_w[j][k] += grad_w[j][k];
                            }
                        }
                        for k in 0..model.vocab.len() {
                            acc_b[k] += grad_b[k];
                        }
                        (acc_w, acc_b, acc_loss + loss)
                    },
                );

            total_loss += loss_sum;
            batch_count += 1;

            if batch_count % 10 == 0 {
                println!(
                    "   🌀 Batch {} loss {:.4}",
                    batch_count,
                    loss_sum / (end - idx) as f32
                );
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "✅ Epoch complete. Avg loss {:.4} | Time {:.2}s",
            total_loss / (batch_count as f32 * BATCH_SIZE as f32),
            elapsed
        );
    }

    println!("Training complete!");
}
