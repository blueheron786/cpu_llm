use cpu_llm::{
    model::{run_inference, TinyRnnModel},
    io::load_model,
};
use std::path::Path;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let model_path = "model.txt";
    
    if !Path::new(model_path).exists() {
        println!("No model found at '{}'.", model_path);
        println!("Please train a model first by running:");
        println!("  cargo run --bin hybrid_train");
        return Ok(());
    }

    println!("Loading model from '{}'...", model_path);
    match load_model(model_path) {
        Ok(model) => {
            println!("✅ Model loaded successfully!");
            println!("  Vocabulary size: {}", model.vocab.len());
            println!("  Context size: {}", model.context_size);
            println!("  Hidden size: {}", model.hidden_size);
            
            println!("\nEnter a prompt (or press Ctrl+C to exit):");
            
            // Simple REPL for text generation
            let mut input = String::new();
            loop {
                print!("> ");
                io::stdout().flush()?; // Ensure the prompt is displayed
                
                input.clear();
                match io::stdin().read_line(&mut input) {
                    Ok(0) => break, // EOF (Ctrl+D)
                    Ok(_) => {
                        let prompt = input.trim();
                        if prompt.is_empty() {
                            println!("Please enter a prompt or press Ctrl+C to exit.");
                            continue;
                        }
                        
                        println!("\nGenerating text...");
                        let output = run_inference(&model, prompt, 100);
                        println!("\nGenerated: {}\n", output);
                    }
                    Err(e) => {
                        eprintln!("\nError reading input: {}", e);
                        break;
                    }
                }
            }
        },
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            eprintln!("The model file may be corrupted or in an incompatible format.");
            std::process::exit(1);
        }
    }
    
    Ok(())
}
