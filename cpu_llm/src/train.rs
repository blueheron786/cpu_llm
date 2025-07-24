use cpu_llm::{model::TinyRnnModel, io::save_model};
use glob::glob;
use rayon::prelude::*;
use std::fs;

const FIXED_VOCAB: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\n";

pub fn relu(x: Vec<f32>) -> Vec<f32> {
    x.into_iter().map(|v| v.max(0.0)).collect()
}

pub fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
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

pub fn softmax(mut logits: Vec<f32>) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter_mut().map(|l| {
        *l = (*l - max_logit).exp();
        *l
    }).sum();
    logits.iter_mut().for_each(|l| *l /= exp_sum.max(1e-8));
    logits
}

pub fn train(text: &str, output_path: &str) {
    let mut vocab: Vec<char> = FIXED_VOCAB.chars().collect();
    vocab.sort(); // Avoid duplicates but preserve fixed vocab
    let model = TinyRnnModel::new(vocab);
    let vocab_size = model.vocab.len();
    let hidden_size = model.hidden_size;
    let context_size = model.context_size;

    let text_chars: Vec<char> = text.chars().collect();
    let epochs = 5;
    let batch_size = 512;
    let lr = 0.05;

    let mut model = model;
    for epoch in 0..epochs {
        println!("🚀 Epoch {}/{}", epoch + 1, epochs);
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for idx in (context_size..text_chars.len() - 1).step_by(batch_size) {
            let end = (idx + batch_size).min(text_chars.len() - 1);

            // Multithreaded batch loss & gradients
            let (batch_grad_w, batch_grad_b, batch_loss): (Vec<Vec<f32>>, Vec<f32>, f32) =
                (idx..end)
                .into_par_iter()
                .map_init(
                    || {
                        (
                            vec![vec![0.0; vocab_size]; hidden_size], // grad_w
                            vec![0.0; vocab_size],                    // grad_b
                            0.0_f32,                                   // loss
                        )
                    },
                    |(grad_w, grad_b, loss), i| {
                        let context: Vec<usize> = text_chars[i - context_size..i]
                            .iter()
                            .filter_map(|c| model.stoi.get(c).copied())
                            .filter(|&id| id < vocab_size)
                            .collect();

                        if context.len() < context_size {
                            return (grad_w.clone(), grad_b.clone(), *loss);
                        }

                        let mut avg_embed = vec![0.0; hidden_size];
                        for &id in &context {
                            for j in 0..hidden_size {
                                avg_embed[j] += model.embedding[id][j];
                            }
                        }
                        for j in 0..hidden_size {
                            avg_embed[j] /= context.len() as f32;
                        }

                        let h = relu(linear(&avg_embed, &model.ff1_weights, &model.ff1_bias));
                        let logits = linear(&h, &model.ff2_weights, &model.ff2_bias);
                        let probs = softmax(logits);

                        let target_char = text_chars[i];
                        let target_idx = match model.stoi.get(&target_char) {
                            Some(&idx) if idx < vocab_size => idx,
                            _ => return (grad_w.clone(), grad_b.clone(), *loss),
                        };

                        let safe_prob = probs[target_idx].max(1e-8);
                        *loss += -safe_prob.ln();

                        let mut dlogits = probs;
                        dlogits[target_idx] -= 1.0;

                        for j in 0..hidden_size {
                            for k in 0..vocab_size {
                                grad_w[j][k] += dlogits[k] * h[j];
                            }
                        }
                        for k in 0..vocab_size {
                            grad_b[k] += dlogits[k];
                        }

                        (grad_w.clone(), grad_b.clone(), *loss)
                    },
                )
                .reduce(
                    || (vec![vec![0.0; vocab_size]; hidden_size], vec![0.0; vocab_size], 0.0_f32),
                    |(mut acc_w, mut acc_b, acc_loss), (grad_w, grad_b, loss)| {
                        for j in 0..hidden_size {
                            for k in 0..vocab_size {
                                acc_w[j][k] += grad_w[j][k];
                            }
                        }
                        for k in 0..vocab_size {
                            acc_b[k] += grad_b[k];
                        }
                        (acc_w, acc_b, acc_loss + loss)
                    },
                );

            for j in 0..hidden_size {
                for k in 0..vocab_size {
                    model.ff2_weights[j][k] -= lr * batch_grad_w[j][k] / batch_size as f32;
                }
            }
            for k in 0..vocab_size {
                model.ff2_bias[k] -= lr * batch_grad_b[k] / batch_size as f32;
            }

            total_loss += batch_loss;
            batch_count += 1;
            if batch_count % 100 == 0 {
                println!("   🌀 Batch {}, loss = {:.4}", batch_count, batch_loss / batch_size as f32);
            }
        }

        println!("✅ Epoch complete. Avg loss = {:.4}", total_loss / (batch_count as f32 * batch_size as f32));
    }

    match save_model(output_path, &model) {
        Ok(_) => println!("✅ Model saved to {}", output_path),
        Err(e) => eprintln!("❌ Failed to save model: {}", e),
    }
}

fn main() {
    println!("📁 Scanning for training files in data/**/* ...");

    let mut combined_text = String::new();
    let mut file_count = 0;

    for entry in glob("data/**/*").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) if path.is_file() => {
                println!("📄 Loading {}", path.display());
                match fs::read_to_string(&path) {
                    Ok(content) => {
                        combined_text.push_str(&content);
                        file_count += 1;
                    }
                    Err(e) => eprintln!("⚠️ Could not read {}: {}", path.display(), e),
                }
            }
            _ => {}
        }
    }

    if file_count == 0 {
        eprintln!("❌ No training files found in data/**/*");
        std::process::exit(1);
    }

    println!("📚 Loaded {} files ({} characters total)", file_count, combined_text.len());
    train(&combined_text, "model.json");
}
