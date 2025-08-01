use cpu_llm::{model::{run_inference, run_batch_inference}, io::load_model};
use std::time::Instant;

fn main() {
    // Set thread count to maximum available cores
    let num_threads = rayon::current_num_threads();
    println!("🚀 Using {} threads for maximum parallelization", num_threads);
    
    if std::path::Path::new("model.json").exists() {
        let model = load_model("model.json").expect("Failed to load model.");
        
        // Single inference test
        let prompt = "It was a dark and stormy night. ";
        let start = Instant::now();
        let output = run_inference(prompt, 100, &model);
        let duration = start.elapsed();
        println!("🔥 Single inference completed in {:.2}ms", duration.as_millis());
        println!("Generated: {}", output);
        
        // Batch inference test for maximum parallelization
        let prompts = vec![
            "Once upon a time, ",
            "The quick brown fox ",
            "In a galaxy far away, ",
            "Scientists have discovered ",
            "The weather today is ",
            "Technology has changed ",
            "Love is when you ",
            "The mystery began when ",
        ];
        
        println!("\n🚀 Running batch inference with {} prompts...", prompts.len());
        let start = Instant::now();
        let batch_outputs = run_batch_inference(&prompts, 50, &model);
        let duration = start.elapsed();
        println!("🔥 Batch inference completed in {:.2}ms", duration.as_millis());
        println!("Average per prompt: {:.2}ms", duration.as_millis() as f64 / prompts.len() as f64);
        
        for (i, output) in batch_outputs.iter().enumerate() {
            println!("Prompt {}: {}", i + 1, output);
        }
        
    } else {
        println!("No model.json found. Please run training first.");
    }
}
