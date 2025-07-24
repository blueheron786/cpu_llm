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
const BATCH_SIZE: usize = 4096;  // Increased batch size for faster training
const LR: f32 = 0.01;  // Slightly reduced learning rate for larger batches

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

// Clean text by normalizing spaces and lowercasing. Efficiently.
fn clean_text(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut prev_was_space = false;

    for c in text.chars() {
        let c = match c {
            '\n' | '\r' => ' ',
            _ => c.to_ascii_lowercase(),
        };

        if c.is_whitespace() {
            if !prev_was_space {
                cleaned.push(' ');
                prev_was_space = true;
            }
        } else {
            cleaned.push(c);
            prev_was_space = false;
        }
    }

    // Trim trailing space if any
    if cleaned.ends_with(' ') {
        cleaned.pop();
    }

    cleaned
}


// Build word vocab for words with frequency >= min_freq
fn build_word_vocab(text: &str, min_freq: usize) -> HashSet<String> {
    // Split text into chunks for parallel processing
    let chunks: Vec<&str> = text.split_whitespace().collect();
    
    // Process chunks in parallel to count frequencies
    let freq = chunks.par_iter()
        .fold(
            || HashMap::new(),
            |mut acc, word| {
                *acc.entry((*word).to_string()).or_insert(0) += 1;
                acc
            }
        )
        .reduce(
            || HashMap::new(),
            |mut map1, map2| {
                for (word, count) in map2 {
                    *map1.entry(word).or_insert(0) += count;
                }
                map1
            }
        );

    // Filter words meeting minimum frequency
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
    fn new(vocab_vec: Vec<String>, context_size: usize, hidden_size: usize) -> Self {
        let vocab_size = vocab_vec.len();
        let mut stoi = HashMap::with_capacity(vocab_size);
        let itos = vocab_vec.clone();
        for (i, w) in vocab_vec.iter().enumerate() {
            stoi.insert(w.clone(), i);
        }

        fn rand_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
            let mut rng = rand::thread_rng();
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-0.1..0.1)).collect())
                .collect()
        }

        TinyRnnModel {
            vocab: vocab_vec,
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

    // Forward pass for one context (token IDs) with buffer reuse
    fn forward_buffered(&self, context: &[usize], h_out: &mut [f32], logits_out: &mut [f32]) {
        // Reuse h_out as embedding buffer
        h_out.fill(0.0);
        for &id in context {
            for i in 0..self.hidden_size {
                h_out[i] += self.embedding[id][i];
            }
        }
        let context_size = context.len() as f32;
        for i in 0..self.hidden_size {
            h_out[i] /= context_size;
        }

        // In-place ReLU after linear
        linear_inplace(h_out, &self.ff1_weights, &self.ff1_bias, logits_out);
        for x in h_out.iter_mut() {
            *x = x.max(0.0);
        }

        // Final linear layer
        linear_inplace(h_out, &self.ff2_weights, &self.ff2_bias, logits_out);
    }

    // Keep the original forward for compatibility
    fn forward(&self, context: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0; self.hidden_size];
        let mut logits = vec![0.0; self.vocab.len()];
        self.forward_buffered(context, &mut h, &mut logits);
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

// Optimized in-place linear operation
fn linear_inplace(input: &[f32], weights: &[Vec<f32>], bias: &[f32], output: &mut [f32]) {
    output.copy_from_slice(bias);
    for (i, &x) in input.iter().enumerate() {
        let row = &weights[i];
        for (out, &w) in output.iter_mut().zip(row.iter()) {
            *out += x * w;
        }
    }
}

// Original linear for compatibility
fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; bias.len()];
    linear_inplace(input, weights, bias, &mut output);
    output
}

// In-place softmax operation
fn softmax_inplace(logits: &mut [f32], output: &mut [f32]) {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    
    // Compute exp in-place
    for (i, &l) in logits.iter().enumerate() {
        let exp_val = (l - max_logit).exp();
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize in-place
    let sum = sum.max(1e-8);
    output.iter_mut().for_each(|x| *x /= sum);
}

// Original softmax for compatibility
fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; logits.len()];
    let mut logits = logits.to_vec();
    softmax_inplace(&mut logits, &mut output);
    output
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
    let word_vocab = build_word_vocab(&clean, 15);

    println!("First pass: Building vocab...");
    let mut token_freqs: HashMap<String, usize> = HashMap::new();
    
    // First pass: Count token frequencies
    for word in clean.split_whitespace() {
        if word_vocab.contains(word) {
            *token_freqs.entry(word.to_string()).or_insert(0) += 1;
        } else {
            for ch in word.chars() {
                *token_freqs.entry(ch.to_string()).or_insert(0) += 1;
            }
        }
    }

    // Build vocab from frequent tokens
    let min_freq = 2; // Only keep tokens that appear at least twice
    let mut vocab: Vec<String> = token_freqs
        .into_iter()
        .filter(|(_, freq)| *freq >= min_freq)
        .map(|(token, _)| token)
        .collect();
    vocab.sort();

    println!("Vocab size: {} tokens", vocab.len());

    let model = TinyRnnModel::new(vocab.clone(), CONTEXT_SIZE, HIDDEN_SIZE);

    // Second pass: Convert tokens directly to IDs without storing intermediate tokens
    println!("Second pass: Converting text to token IDs...");
    let mut token_ids = Vec::new();
    token_ids.reserve(clean.split_whitespace().count() * 2); // Rough estimate

    for word in clean.split_whitespace() {
        if word_vocab.contains(word) {
            if let Some(&id) = model.stoi.get(word) {
                token_ids.push(id);
            }
        } else {
            for ch in word.chars() {
                if let Some(&id) = model.stoi.get(&ch.to_string()) {
                    token_ids.push(id);
                }
            }
        }
    }

    println!("Training for {} epochs...", EPOCHS);

    for epoch in 0..EPOCHS {
        println!("🚀 Epoch {}/{}", epoch + 1, EPOCHS);
        let start = Instant::now();
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let total_batches = ((token_ids.len() - 1 - CONTEXT_SIZE) as f64 / BATCH_SIZE as f64).ceil() as usize;

        // Pre-allocate thread-local buffers to avoid repeated allocations
        let chunk_size = BATCH_SIZE / rayon::current_num_threads().max(1);
        
        for idx in (CONTEXT_SIZE..token_ids.len() - 1).step_by(BATCH_SIZE) {
            let batch_start = Instant::now();
            let end = (idx + BATCH_SIZE).min(token_ids.len() - 1);

            // Process multiple samples per thread using chunked iteration
            let (grad_w_sum, grad_b_sum, loss_sum) = (idx..end).into_par_iter()
                .chunks(chunk_size)
                .map(|chunk| {
                    // Thread-local buffers allocated once per chunk
                    let mut grad_w = vec![vec![0.0; model.vocab.len()]; model.hidden_size];
                    let mut grad_b = vec![0.0; model.vocab.len()];
                    let mut loss = 0.0_f32;
                    let mut h_buffer = vec![0.0; model.hidden_size];
                    let mut logits_buffer = vec![0.0; model.vocab.len()];
                    let mut probs_buffer = vec![0.0; model.vocab.len()];

                    // Process multiple samples in this thread
                    for i in chunk {
                        let context = &token_ids[i - CONTEXT_SIZE..i];
                        let target = token_ids[i];

                        // Reuse buffers for forward pass
                        model.forward_buffered(context, &mut h_buffer, &mut logits_buffer);
                        softmax_inplace(&mut logits_buffer, &mut probs_buffer);

                        let safe_prob = probs_buffer[target].max(1e-8);
                        loss += -safe_prob.ln();

                        // Compute gradients in-place
                        probs_buffer[target] -= 1.0;
                        
                        // Optimized gradient accumulation
                        for j in 0..model.hidden_size {
                            let h_j = h_buffer[j];
                            let grad_w_j = &mut grad_w[j];
                            for (k, &dl_k) in probs_buffer.iter().enumerate() {
                                grad_w_j[k] += dl_k * h_j;
                            }
                        }
                        grad_b.iter_mut().zip(probs_buffer.iter())
                            .for_each(|(gb, &dl)| *gb += dl);
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
                        // Optimized gradient reduction
                        acc_w.iter_mut().zip(grad_w.iter())
                            .for_each(|(acc_row, grad_row)| {
                                acc_row.iter_mut().zip(grad_row.iter())
                                    .for_each(|(acc, &grad)| *acc += grad);
                            });
                        acc_b.iter_mut().zip(grad_b.iter())
                            .for_each(|(acc, &grad)| *acc += grad);
                        (acc_w, acc_b, acc_loss + loss)
                    },
                );

            model.update_weights(&grad_w_sum, &grad_b_sum, LR, end - idx);

            total_loss += loss_sum;
            batch_count += 1;

            // Print progress less frequently
            if batch_count % 10 == 0 || batch_count == 1 {
                let batch_duration = batch_start.elapsed().as_secs_f64();
                let eta = batch_duration * (total_batches - batch_count) as f64;
                
                println!(
                    "Batch {}/{} ({:.1}%), loss {:.4}, ETA: {:.1}m",
                    batch_count,
                    total_batches,
                    (batch_count as f32 / total_batches as f32) * 100.0,
                    loss_sum / (end - idx) as f32,
                    eta / 60.0
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
