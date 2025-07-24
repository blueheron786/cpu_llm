use cpu_llm::{model::TinyRnnModel, io::save_model};
use glob::glob;
use rayon::prelude::*;
use std::fs;

const FIXED_VOCAB: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\n";

pub fn train(text: &str, output_path: &str) {
    // Build vocab from text
    // let mut vocab: Vec<char> = text.chars().collect();
    
    // Used a fixed vocab for simplicity, and to avoid index-out-of-bounds errors
    let mut vocab: Vec<char> = FIXED_VOCAB.chars().collect();
    vocab.sort();
    // Don't de-dupe, it removes lower-case variables.

    let model = TinyRnnModel::new(vocab);

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

    let contents: Vec<String> = paths.par_iter()
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
