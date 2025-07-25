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
const BATCH_SIZE: usize = 16384;  // Further increased batch size for faster training
const LR: f32 = 0.003;  // Further reduced learning rate for stability

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

    // Forward pass for one context (token IDs) with buffer reuse and parallel processing
    fn forward_buffered(&self, context: &[usize], h_out: &mut [f32], logits_out: &mut [f32]) {
        // Parallel embedding lookup and averaging
        h_out.par_iter_mut().enumerate().for_each(|(i, h)| {
            *h = context.iter()
                .map(|&id| self.embedding[id][i])
                .sum::<f32>() / context.len() as f32;
        });

        // Parallel linear + ReLU
        linear_inplace_parallel(h_out, &self.ff1_weights, &self.ff1_bias, logits_out);
        h_out.par_iter_mut().for_each(|x| *x = x.max(0.0));

        // Parallel final linear layer
        linear_inplace_parallel(h_out, &self.ff2_weights, &self.ff2_bias, logits_out);
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
        let scale = lr / batch_size as f32;
        
        // Parallel weight updates
        self.ff2_weights.par_iter_mut()
            .zip(grad_w.par_iter())
            .for_each(|(weights_row, grad_row)| {
                weights_row.par_iter_mut()
                    .zip(grad_row.par_iter())
                    .for_each(|(w, &g)| {
                        *w -= scale * g;
                    });
            });

        // Parallel bias updates
        self.ff2_bias.par_iter_mut()
            .zip(grad_b.par_iter())
            .for_each(|(b, &g)| {
                *b -= scale * g;
            });
    }
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

// Parallel in-place linear operation
fn linear_inplace_parallel(input: &[f32], weights: &[Vec<f32>], bias: &[f32], output: &mut [f32]) {
    // Initialize output with bias
    output.iter_mut().zip(bias.iter()).for_each(|(o, &b)| *o = b);
    
    // Split the work into chunks for each thread
    let chunk_size = std::cmp::max(1, output.len() / rayon::current_num_threads());
    
    output.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let out_start = chunk_idx * chunk_size;
            
            // Process each input value
            for (i, &x) in input.iter().enumerate() {
                if x == 0.0 { continue; }  // Skip zero inputs
                
                // Process output block for this chunk
                for j in 0..out_chunk.len() {
                    if out_start + j < weights[i].len() {
                        out_chunk[j] += x * weights[i][out_start + j];
                    }
                }
            }
    });
}

// Original in-place linear operation for smaller matrices
fn linear_inplace(input: &[f32], weights: &[Vec<f32>], bias: &[f32], output: &mut [f32]) {
    if output.len() >= 1024 { // Use parallel version for larger matrices
        linear_inplace_parallel(input, weights, bias, output);
        return;
    }
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

// Save model to a file
fn save_model(path: &str, model: &TinyRnnModel) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    
    // Write vocab
    for token in &model.vocab {
        writeln!(file, "{}", token)?;
    }
    
    // Write embeddings
    for row in &model.embedding {
        for &val in row {
            write!(file, "{} ", val)?;
        }
        writeln!(file)?;
    }
    
    // Write ff1 weights
    for row in &model.ff1_weights {
        for &val in row {
            write!(file, "{} ", val)?;
        }
        writeln!(file)?;
    }
    
    // Write ff1 bias
    for &val in &model.ff1_bias {
        write!(file, "{} ", val)?;
    }
    writeln!(file)?;
    
    // Write ff2 weights
    for row in &model.ff2_weights {
        for &val in row {
            write!(file, "{} ", val)?;
        }
        writeln!(file)?;
    }
    
    // Write ff2 bias
    for &val in &model.ff2_bias {
        write!(file, "{} ", val)?;
    }
    writeln!(file)?;
    
    Ok(())
}

fn main() {
    let output_path = "model.txt";
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

    // Process text in chunks to avoid storing the entire cleaned text
    const CHUNK_SIZE: usize = 1_000_000; // Process 1MB at a time
    
    println!("First pass: Counting frequencies in parallel chunks...");
    
    // Create chunk ranges for parallel processing, ensuring we split at char boundaries
    let mut chunk_ranges = Vec::new();
    let mut start = 0;
    while start < combined_text.len() {
        let mut end = (start + CHUNK_SIZE).min(combined_text.len());
        // Adjust end to land on a char boundary
        while !combined_text.is_char_boundary(end) && end > start {
            end -= 1;
        }
        chunk_ranges.push(start..end);
        start = end;
    }
    
    // Process chunks in parallel to count word frequencies
    let word_freqs = chunk_ranges.par_iter()
        .fold(
            || HashMap::new(),
            |mut freq_map, range| {
                let chunk = &combined_text[range.clone()];
                for word in clean_text(chunk).split_whitespace() {
                    *freq_map.entry(word.to_string()).or_insert(0) += 1;
                }
                freq_map
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

    // Build word vocabulary from frequent words
    let word_vocab: HashSet<String> = word_freqs.into_par_iter()
        .filter(|(_, freq)| *freq >= 100)  // Further increased frequency threshold
        .map(|(word, _)| word)
        .collect();
    
    println!("Word vocab size: {} words", word_vocab.len());

    // Second pass: count token frequencies in parallel
    println!("Second pass: Building token frequencies in parallel...");
    let token_freqs = chunk_ranges.par_iter()
        .fold(
            || HashMap::new(),
            |mut freq_map, range| {
                let chunk = &combined_text[range.clone()];
                for word in clean_text(chunk).split_whitespace() {
                    if word_vocab.contains(word) {
                        *freq_map.entry(word.to_string()).or_insert(0) += 1;
                    } else {
                        for ch in word.chars() {
                            *freq_map.entry(ch.to_string()).or_insert(0) += 1;
                        }
                    }
                }
                freq_map
            }
        )
        .reduce(
            || HashMap::new(),
            |mut map1, map2| {
                for (token, count) in map2 {
                    *map1.entry(token).or_insert(0) += count;
                }
                map1
            }
        );

    // Build final vocabulary
    let min_token_freq = 5; // Increased threshold to reduce vocab size
    let mut vocab: Vec<String> = token_freqs.iter()
        .filter(|(_, &freq)| freq >= min_token_freq)
        .map(|(token, _)| token.clone())
        .collect();
    vocab.sort();
    drop(token_freqs); // Free memory

    println!("Final vocab size: {} tokens", vocab.len());
    let mut model = TinyRnnModel::new(vocab, CONTEXT_SIZE, HIDDEN_SIZE);

    // Third pass: Convert text to IDs using parallel processing
    println!("Third pass: Converting to token IDs in parallel...");
    
    // Process chunks in parallel and collect token IDs
    let token_ids: Vec<usize> = chunk_ranges.par_iter()
        .flat_map(|range| {
            let chunk = &combined_text[range.clone()];
            let mut chunk_ids = Vec::new();
            
            for word in clean_text(chunk).split_whitespace() {
                if word_vocab.contains(word) {
                    if let Some(&id) = model.stoi.get(word) {
                        chunk_ids.push(id);
                    }
                } else {
                    for ch in word.chars() {
                        if let Some(&id) = model.stoi.get(&ch.to_string()) {
                            chunk_ids.push(id);
                        }
                    }
                }
            }
            chunk_ids
        })
        .collect();
    
    drop(combined_text); // Free the original text
    drop(word_vocab); // Free word vocabulary

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

    match save_model(output_path, &model) {
        Ok(_) => println!("✅ Model saved to {}", output_path),
        Err(e) => eprintln!("❌ Failed to save model: {}", e),
    }
}
