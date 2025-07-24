use cpu_llm::{model::TinyRnnModel, io::save_model};
use glob::glob;
use std::fs;

pub fn train(text: &str, output_path: &str) {
    // Build vocab from text
    let mut vocab: Vec<char> = text.chars().collect();
    vocab.sort();
    vocab.dedup();

    let model = TinyRnnModel::new(vocab);

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
