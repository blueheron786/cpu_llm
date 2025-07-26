//! Hybrid tokenizer and training for the CPU LLM

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use glob::glob;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::runtime::Runtime;
use cpu_llm::async_io::process_files;
use cpu_llm::model::TinyRnnModel;

// Hyperparameters
const CONTEXT_SIZE: usize = 6; // Moderate context for better quality
const HIDDEN_SIZE: usize = 48; // Moderate hidden size for better quality
const EPOCHS: usize = 5;
const BATCH_SIZE: usize = 1024; // Further reduced batch size for speed
const LR: f32 = 0.003;

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
            for (i, &x) in input.iter().enumerate() {
                if x == 0.0 { continue; }
                if out_start >= weights[i].len() { continue; }
                let end_idx = (out_start + out_chunk.len()).min(weights[i].len());
                let weights_row = &weights[i][out_start..end_idx];
                for (out, &w) in out_chunk.iter_mut().take(weights_row.len()).zip(weights_row.iter()) {
                    *out += x * w;
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
    let mut file = File::create(path)?;
    
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simple timing for now - we'll add more sophisticated profiling later
    let _start_time = Instant::now();
    let output_path = "model.txt";
    
    // Set rayon thread count from env (default: all cores)
    if let Ok(num_threads) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(n) = num_threads.parse::<usize>() {
            rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
        }
    }

    // Create a new Tokio runtime for async operations
    let rt = Runtime::new()?;
    
    println!("📁 Loading files asynchronously from data/**/* ...");
    
    // Configuration
    let min_word_freq = 5; // Minimum word frequency for vocabulary
    
    // First pass: build word frequency map
    println!("📚 Building vocabulary...");
    let word_freq = Arc::new(Mutex::new(HashMap::<String, u64>::new()));
    
    // Process files in smaller chunks to save memory
    let chunk_size = 1024 * 1024; // Process 1MB at a time
    let files = glob::glob("data/**/*")?;
    
    for file in files.filter_map(Result::ok) {
        if !file.is_file() { continue; }
        
        let file = File::open(&file)?;
        let reader = BufReader::new(file);
        let mut buffer = String::with_capacity(chunk_size);
        
        for line in reader.lines().filter_map(Result::ok) {
            buffer.push_str(&line);
            buffer.push(' '); // Add space to separate lines
            
            if buffer.len() >= chunk_size {
                let chunk = buffer.drain(..).collect::<String>();
                let mut freq_map = word_freq.lock().unwrap();
                for word in chunk.split_whitespace() {
                    if !word.is_empty() {
                        *freq_map.entry(word.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
        
        // Process remaining content in buffer
        if !buffer.is_empty() {
            let mut freq_map = word_freq.lock().unwrap();
            for word in buffer.split_whitespace() {
                if !word.is_empty() {
                    *freq_map.entry(word.to_string()).or_insert(0) += 1;
                }
            }
            buffer.clear();
        }
    }
    
    // Build vocabulary from word frequencies
    let word_vocab: HashSet<String> = {
        let freq_map = word_freq.lock().unwrap();
        freq_map.iter()
            .filter(|(_, &count)| count >= min_word_freq)
            .map(|(word, _)| word.clone())
            .collect()
    };
    
    println!("📚 Built vocabulary of {} words", word_vocab.len());
    
    // Second pass: tokenize and build final dataset with streaming
    println!("🔠 Tokenizing text...");
    let token_freq = Arc::new(Mutex::new(HashMap::<String, u64>::new()));
    let token_ids = Arc::new(Mutex::new(Vec::new()));
    
    // Process files in the same chunked manner
    let files = glob::glob("data/**/*")?;
    
    for file in files.filter_map(Result::ok) {
        if !file.is_file() { continue; }
        
        let file = File::open(&file)?;
        let reader = BufReader::new(file);
        
        for line in reader.lines().filter_map(Result::ok) {
            let mut local_token_ids = Vec::new();
            let mut local_token_freq = HashMap::new();
            
            for word in line.split_whitespace() {
                if word.is_empty() {
                    continue;
                }
                
                if word_vocab.contains(word) {
                    // Word is in vocabulary, use as is
                    *local_token_freq.entry(word.to_string()).or_insert(0) += 1;
                    local_token_ids.push(word.to_string());
                } else {
                    // Word not in vocabulary, split into characters
                    for ch in word.chars() {
                        let ch_str = ch.to_string();
                        *local_token_freq.entry(ch_str.clone()).or_insert(0) += 1;
                        local_token_ids.push(ch_str);
                    }
                }
            }
            
            // Merge with global frequencies and token IDs
            let mut global_freq = token_freq.lock().unwrap();
            for (token, count) in local_token_freq {
                *global_freq.entry(token).or_insert(0) += count;
            }
            
            // Add to token IDs
            let mut ids = token_ids.lock().unwrap();
            ids.extend(local_token_ids);
        }
    }
    
    // Convert Arc<Mutex<Vec<String>>> to Vec<String>
    let token_ids = Arc::try_unwrap(token_ids).unwrap().into_inner()?;
    
    if token_ids.is_empty() {
        eprintln!("❌ No tokens found in the processed files");
        std::process::exit(1);
    }
    
    println!("✅ Tokenization complete. Total tokens: {}", token_ids.len());
    
    // Build token to ID mapping
    let token_freq = token_freq.lock().unwrap();
    let mut token_id_map = HashMap::new();
    let mut id_token_map = Vec::new();
    
    // Add special tokens
    token_id_map.insert("<PAD>".to_string(), 0);
    id_token_map.push("<PAD>".to_string());
    token_id_map.insert("<UNK>".to_string(), 1);
    id_token_map.push("<UNK>".to_string());
    
    // Add most frequent tokens first
    let mut sorted_tokens: Vec<_> = token_freq.iter().collect();
    sorted_tokens.sort_by(|a, b| b.1.cmp(a.1));
    
    for (token, _) in sorted_tokens {
        let id = id_token_map.len();
        token_id_map.insert(token.clone(), id);
        id_token_map.push(token.clone());
    }
    
    println!("🔢 Built vocabulary of {} tokens", id_token_map.len());
    
    // Convert token strings to IDs
    let token_ids: Vec<usize> = token_ids
        .into_iter()
        .map(|token| *token_id_map.get(&token).unwrap_or(&1)) // 1 = <UNK>
        .collect();
    
    // Create model with the built vocabulary
    let model = TinyRnnModel::new(id_token_map, CONTEXT_SIZE, HIDDEN_SIZE);
    
    println!("📚 Processed {} tokens", token_ids.len());
    
    // Create chunk ranges for parallel processing, ensuring we split at token boundaries
    let mut chunk_ranges = Vec::new();
    let mut start = 0;
    while start < token_ids.len() {
        let end = (start + 1024).min(token_ids.len());
        chunk_ranges.push(start..end);
        start = end;
    }
    
    // Process chunks in parallel to count word frequencies
    let word_freqs = chunk_ranges.par_iter()
        .fold(
            || HashMap::new(),
            |mut freq_map, range| {
                let chunk = &token_ids[range.clone()];
                for &id in chunk {
                    *freq_map.entry(id).or_insert(0) += 1;
                }
                freq_map
            }
        )
        .reduce(
            || HashMap::new(),
            |mut map1, map2| {
                for (id, count) in map2 {
                    *map1.entry(id).or_insert(0) += count;
                }
                map1
            }
        );

    // Build word vocabulary from frequent words
    let word_vocab: HashSet<usize> = word_freqs.into_par_iter()
        .filter(|(_, freq)| *freq >= 100)  // Further increased frequency threshold
        .map(|(id, _)| id)
        .collect();
    
    println!("Word vocab size: {} words", word_vocab.len());

    // Second pass: count token frequencies in parallel
    println!("Second pass: Building token frequencies in parallel...");
    let token_freqs = chunk_ranges.par_iter()
        .fold(
            || HashMap::new(),
            |mut freq_map, range| {
                let chunk = &token_ids[range.clone()];
                for &id in chunk {
                    *freq_map.entry(id).or_insert(0) += 1;
                }
                freq_map
            }
        )
        .reduce(
            || HashMap::new(),
            |mut map1, map2| {
                for (id, count) in map2 {
                    *map1.entry(id).or_insert(0) += count;
                }
                map1
            }
        );

    // Build final vocabulary
    let min_token_freq = 5; // Increased threshold to reduce vocab size
    let mut vocab: Vec<usize> = token_freqs.iter()
        .filter(|(_, &freq)| freq >= min_token_freq)
        .map(|(id, _)| *id)
        .collect();
    vocab.sort();
    drop(token_freqs); // Free memory

    println!("Final vocab size: {} tokens", vocab.len());
    // Convert vocab from Vec<usize> to Vec<String>
    let vocab_strings: Vec<String> = vocab.iter().map(|&id| id.to_string()).collect();
    let mut model = TinyRnnModel::new(vocab_strings, CONTEXT_SIZE, HIDDEN_SIZE);

    // Third pass: Convert text to IDs using parallel processing, with disk cache
    println!("Third pass: Converting to token IDs in parallel (with cache)...");
    let _token_id_cache_path = "token_ids.cache";
    println!("Generating token IDs...");
    let gen_start = Instant::now();
    // Process chunks in parallel and collect results
    let token_chunks: Vec<Vec<usize>> = chunk_ranges.par_iter()
        .map(|range| {
            token_ids[range.clone()].to_vec()
        })
        .collect();
    
    // Flatten the chunks into a single vector
    let token_ids: Vec<usize> = token_chunks.into_iter().flatten().collect();
    let gen_elapsed = gen_start.elapsed().as_secs_f64();
    println!("Token IDs generated in {:.2}s", gen_elapsed);
    // Subsample for benchmarking speed
    let token_ids = if token_ids.len() > 1_000_000 {
        token_ids[..1_000_000].to_vec()
    } else {
        token_ids
    };
    drop(chunk_ranges); // Free the original text
    drop(word_vocab); // Free word vocabulary

    println!("Training for {} epochs...", EPOCHS);

    for epoch in 0..EPOCHS {
        println!("🚀 Epoch {}/{}", epoch + 1, EPOCHS);
        let start = Instant::now();
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let total_batches = ((token_ids.len() - 1 - CONTEXT_SIZE) as f64 / BATCH_SIZE as f64).ceil() as usize;
        
        // Track epoch timing
        let _epoch_start = Instant::now();

        // Pre-allocate thread-local buffers to avoid repeated allocations
        // Use a larger chunk size for better parallel efficiency and cache locality
        let chunk_size = (BATCH_SIZE / rayon::current_num_threads().max(1)).max(256);
        
        // Gradient accumulation buffers
        let mut accum_grad_w = vec![vec![0.0; model.vocab.len()]; model.hidden_size];
        let mut accum_grad_b = vec![0.0; model.vocab.len()];
        let mut accum_count = 0;

        for idx in (CONTEXT_SIZE..token_ids.len() - 1).step_by(BATCH_SIZE) {
            let batch_start = Instant::now();
            let end = (idx + BATCH_SIZE).min(token_ids.len() - 1);

            // Process multiple samples per thread using chunked iteration
            let (grad_w_sum, grad_b_sum, loss_sum) = (idx..end).into_par_iter()
                .chunks(chunk_size)
                .map(|chunk| {
                    // Thread-local buffers allocated once per chunk
                    let mut chunk_grad_w = vec![vec![0.0; model.vocab.len()]; model.hidden_size];
                    let mut chunk_grad_b = vec![0.0; model.vocab.len()];
                    let mut chunk_loss = 0.0;

                    for i in chunk {
                        let context = &token_ids[i - CONTEXT_SIZE..i];
                        let target = token_ids[i];
                        
                        // Skip if target is out of vocabulary
                        if target >= model.vocab.len() {
                            continue;
                        }
                        
                        // Forward pass
                        let mut h = vec![0.0; model.hidden_size];
                        let mut logits = vec![0.0; model.vocab.len()];
                        
                        // Ensure all context tokens are within vocabulary
                        if context.iter().any(|&id| id >= model.vocab.len()) {
                            continue;
                        }
                        
                        model.forward_buffered(context, &mut h, &mut logits);
                        
                        // Compute softmax and loss
                        let mut probs = vec![0.0; model.vocab.len()];
                        softmax_inplace(&mut logits, &mut probs);
                        let safe_prob = probs[target].max(1e-10);
                        chunk_loss += -safe_prob.ln();
                        
                        // Compute gradients
                        for j in 0..model.hidden_size {
                            let h_j = h[j];
                            let grad_w_j = &mut chunk_grad_w[j];
                            for (k, &p) in probs.iter().enumerate() {
                                let grad = if k == target { p - 1.0 } else { p };
                                grad_w_j[k] += grad * h_j;
                            }
                        }
                        
                        // Update bias gradients
                        for (gb, &p) in chunk_grad_b.iter_mut().zip(probs.iter()) {
                            *gb += if p > 1e-10 { p } else { 0.0 };
                        }
                        if target < chunk_grad_b.len() {
                            chunk_grad_b[target] -= 1.0;
                        }
                    }
                    
                    (chunk_grad_w, chunk_grad_b, chunk_loss)
                })
                .reduce(
                    || (
                        vec![vec![0.0; model.vocab.len()]; model.hidden_size],
                        vec![0.0; model.vocab.len()],
                        0.0_f32,
                    ),
                    |(mut acc_w, mut acc_b, acc_loss), (grad_w, grad_b, loss)| {
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

            // Parallel gradient accumulation
            accum_grad_w.par_iter_mut()
                .zip(grad_w_sum.par_iter())
                .for_each(|(acc_row, grad_row)| {
                    acc_row.par_iter_mut()
                        .zip(grad_row.par_iter())
                        .for_each(|(acc, &grad)| *acc += grad);
                });
            
            accum_grad_b.par_iter_mut()
                .zip(grad_b_sum.par_iter())
                .for_each(|(acc, &grad)| *acc += grad);
                
            accum_count += 1;

            // Update weights every ACCUM_STEPS mini-batches
            const ACCUM_STEPS: usize = 4;
            if accum_count == ACCUM_STEPS || batch_count + 1 == total_batches {
                // Apply gradients to the model weights
                let scale = LR / (BATCH_SIZE * accum_count) as f32;
                
                // Update first layer weights
                model.ff1_weights.par_iter_mut()
                    .zip(accum_grad_w.par_iter())
                    .for_each(|(weights_row, grad_row)|
                        weights_row.par_iter_mut()
                            .zip(grad_row.par_iter())
                            .for_each(|(w, &g)| *w -= scale * g)
                    );
                
                // Update second layer weights (ff2_weights is [vocab_size][hidden_size])
                // accum_grad_b is the gradient for the output layer
                model.ff2_weights.par_iter_mut()
                    .zip(accum_grad_b.par_iter())
                    .for_each(|(weights_row, &grad)| {
                        let row_len = weights_row.len() as f32;
                        weights_row.par_iter_mut()
                            .for_each(|w| *w -= scale * grad / row_len)
                    });
                
                // Update second layer biases (ff2_bias is [vocab_size])
                model.ff2_bias.par_iter_mut()
                    .zip(accum_grad_b.par_iter())
                    .for_each(|(b, &g)| *b -= scale * g);
                
                // Reset accumulation buffers
                for row in accum_grad_w.iter_mut() {
                    for v in row.iter_mut() { *v = 0.0; }
                }
                for v in accum_grad_b.iter_mut() { *v = 0.0; }
                accum_count = 0;
            }

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
        Ok(_) => {
            println!("✅ Model saved to {}", output_path);
            Ok(())
        },
        Err(e) => {
            eprintln!("❌ Failed to save model: {}", e);
            Ok(()) // Continue execution even if save fails
        }
    }
}
