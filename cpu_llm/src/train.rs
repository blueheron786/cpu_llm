use cpu_llm::{model::TinyRnnModel, io::save_model};
use cpu_llm::utils::{relu, linear, softmax};
use glob::glob;
use std::fs;

pub fn train(text: &str, output_path: &str) {
    // Limit Rayon to use all but 2 CPU cores
    let num_cpus = num_cpus::get().max(2) - 2;
    rayon::ThreadPoolBuilder::new().num_threads(num_cpus).build_global().ok();

    // Always delete the old model file before training
    let _ = std::fs::remove_file(output_path);
    // Build vocab from training data
    let mut vocab: Vec<char> = text.chars().collect();
    vocab.sort();
    vocab.dedup();
    println!("Training vocabulary ({} chars): {:?}", vocab.len(), vocab);
    let mut model = TinyRnnModel::new(vocab);

    let learning_rate = 0.001;
    let epochs = 3;
    let context_size = model.context_size;

    let text_chars: Vec<char> = text.chars().collect();
    let text_len = text_chars.len();

    use rayon::prelude::*;
    use sysinfo::{System, SystemExt};
    use std::time::Instant;
    let mut sys = System::new_all();
    // RAM limiting: leave at least 2GB free
    let min_free_ram: u64 = 2 * 1024 * 1024 * 1024; // 2GB
    let total_ram: u64 = sys.total_memory() * 1024;
    let target_ram: u64 = total_ram.saturating_sub(min_free_ram);
    let bytes_per_sample = (model.hidden_size * model.vocab_size * 4) + (model.vocab_size * 4) + 4; // grad_w + grad_b + loss
    let mut batch_size = 150_000; // Start high and adapt up/down

    for epoch in 0..epochs {
        let mut total_loss: f32 = 0.0;
        let start = Instant::now();
        let total = text_len - context_size;
        let mut idx = 0;
        while idx < total {
            sys.refresh_memory();
            let free_ram = sys.free_memory() * 1024;
            let used_ram = sys.used_memory() * 1024;
            let max_batch = (target_ram / bytes_per_sample.max(1) as u64).max(1) as usize;
            // If RAM is low, decrease batch size by 10%
            if free_ram < min_free_ram {
                println!("⏸️ RAM usage high (free: {:.1}GB), pausing...", free_ram as f64 / 1e9);
                batch_size = ((batch_size as f64) * 0.98).max(1.0) as usize;
                std::thread::sleep(std::time::Duration::from_secs(2));
                continue;
            }
            batch_size = batch_size.min(max_batch).min(100_000).min(total - idx);
            let end = idx + batch_size;
            let batch_indices: Vec<usize> = (context_size + idx..context_size + end).collect();
            let grads: Vec<(Vec<Vec<f32>>, Vec<f32>, f32)> = batch_indices
                .into_iter()
                .map(|i| {
                    // Prepare context window
                    let context: Vec<usize> = text_chars[i-context_size..i]
                        .iter()
                        .filter_map(|c| model.stoi.get(c).copied())
                        .filter(|&id| id < model.vocab_size)
                        .collect();
                    if context.len() < context_size {
                        return (vec![vec![0.0; model.vocab_size]; model.hidden_size], vec![0.0; model.vocab_size], 0.0);
                    }
                    // Forward pass
                    let mut avg_embed = vec![0.0; model.hidden_size];
                    for &id in &context {
                        for j in 0..model.hidden_size {
                            avg_embed[j] += model.embedding[id][j];
                        }
                    }
                    let n = context.len() as f32;
                    for j in 0..model.hidden_size {
                        avg_embed[j] /= n;
                    }
                    let h = relu(linear(&avg_embed, &model.ff1_weights, &model.ff1_bias));
                    let logits = linear(&h, &model.ff2_weights, &model.ff2_bias);
                    let probs = softmax(logits);
                    // Target
                    let target_char = text_chars[i];
                    let target_idx = match model.stoi.get(&target_char) {
                        Some(idx) if *idx < model.vocab_size => *idx,
                        _ => return (vec![vec![0.0; model.vocab_size]; model.hidden_size], vec![0.0; model.vocab_size], 0.0),
                    };
                    if target_idx >= model.vocab_size {
                        return (vec![vec![0.0; model.vocab_size]; model.hidden_size], vec![0.0; model.vocab_size], 0.0);
                    }
                    // Cross-entropy loss
                    let safe_prob = probs[target_idx].max(1e-8);
                    let loss = -safe_prob.ln();
                    // Gradient for output layer (softmax + cross-entropy)
                    let mut dlogits = probs.clone();
                    dlogits[target_idx] -= 1.0;
                    // Accumulate gradients
                    let mut grad_w = vec![vec![0.0; model.vocab_size]; model.hidden_size];
                    let mut grad_b = vec![0.0; model.vocab_size];
                    for j in 0..model.hidden_size {
                        for k in 0..model.vocab_size {
                            grad_w[j][k] += dlogits[k] * h[j];
                        }
                    }
                    for k in 0..model.vocab_size {
                        grad_b[k] += dlogits[k];
                    }
                    (grad_w, grad_b, loss)
                })
                .collect();
            // Apply accumulated gradients sequentially
            for (grad_w, grad_b, loss) in grads.iter() {
                total_loss += *loss;
                for j in 0..model.hidden_size {
                    for k in 0..model.vocab_size {
                        model.ff2_weights[j][k] -= learning_rate * grad_w[j][k];
                    }
                }
                for k in 0..model.vocab_size {
                    model.ff2_bias[k] -= learning_rate * grad_b[k];
                }
            }
            idx = end;
            // Progress reporting (every batch)
            if idx % 10000 < batch_size {
                let percent = (idx as f64) / (total as f64) * 100.0;
                let elapsed = start.elapsed().as_secs_f64();
                let rate = (idx as f64) / elapsed.max(1.0);
                let remaining = (total - idx) as f64 / rate.max(1.0);
                let eta = format!("ETA: {:.1} min", remaining / 60.0);
                println!("Progress: {}/{} ({:.2}%) {}", idx, total, percent, eta);
            }
        }
        let epoch_time = start.elapsed().as_secs_f64();
        let epochs_left = epochs - (epoch + 1);
        let eta_epochs = epoch_time * (epochs_left as f64);
        println!("Epoch {}/{}: avg loss = {} | ETA: {:.1} min", epoch+1, epochs, total_loss/(text_len as f32), eta_epochs / 60.0);
    }

         match save_model(output_path, &model) {
        Ok(_) => println!("✅ Model saved to {}", output_path),
        Err(e) => eprintln!("❌ Failed to save model: {}", e),
    }
}

fn main() {
    println!("📁 Scanning for training files in data/**/* ...");

    let paths: Vec<_> = glob("data/**/*")
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)
        .filter(|p| p.is_file())
        .collect();

    let contents: Vec<String> = paths.iter()
        .filter_map(|path| {
            fs::read_to_string(path).ok()
        })
        .collect();

    let combined_text = contents.concat();

    if contents.is_empty() {
        eprintln!("❌ No training files found in data/**/*");
        std::process::exit(1);
    }

    println!("📚 Loaded {} files ({} characters total)", contents.len(), combined_text.len());

    train(&combined_text, "model.json");
}
