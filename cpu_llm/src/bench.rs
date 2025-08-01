// src/bench.rs - Performance benchmark for parallelized inference
use cpu_llm::{model::{run_inference, run_batch_inference}, io::load_model};
use std::time::Instant;

fn benchmark_single_vs_batch(model_path: &str) {
    let model = load_model(model_path).expect("Failed to load model");
    
    let test_prompts = vec![
        "The future of artificial intelligence ",
        "Climate change is affecting ",
        "Space exploration will lead to ",
        "The internet has revolutionized ",
        "Machine learning algorithms can ",
        "Renewable energy sources include ",
        "Virtual reality technology enables ",
        "Quantum computing will solve ",
        "Biotechnology advances allow ",
        "Social media platforms connect ",
        "Autonomous vehicles will transform ",
        "Blockchain technology provides ",
        "Genetic engineering can modify ",
        "Neural networks learn by ",
        "Robotics automation improves ",
        "Cloud computing offers scalable ",
    ];
    
    let num_tokens = 50;
    let num_runs = 3;
    
    println!("🚀 Parallel Inference Benchmark");
    println!("================================");
    println!("Model: {}", model_path);
    println!("Prompts: {}", test_prompts.len());
    println!("Tokens per prompt: {}", num_tokens);
    println!("Threads available: {}", rayon::current_num_threads());
    println!();
    
    // Benchmark sequential inference
    println!("📊 Sequential Inference (baseline):");
    let mut sequential_times = Vec::new();
    
    for run in 1..=num_runs {
        let start = Instant::now();
        let _results: Vec<String> = test_prompts.iter()
            .map(|&prompt| run_inference(prompt, num_tokens, &model))
            .collect();
        let duration = start.elapsed();
        sequential_times.push(duration.as_millis());
        println!("  Run {}: {:.2}ms", run, duration.as_millis());
    }
    
    let sequential_avg = sequential_times.iter().sum::<u128>() as f64 / sequential_times.len() as f64;
    
    // Benchmark parallel batch inference
    println!("\n🔥 Parallel Batch Inference:");
    let mut parallel_times = Vec::new();
    
    for run in 1..=num_runs {
        let start = Instant::now();
        let _results = run_batch_inference(&test_prompts, num_tokens, &model);
        let duration = start.elapsed();
        parallel_times.push(duration.as_millis());
        println!("  Run {}: {:.2}ms", run, duration.as_millis());
    }
    
    let parallel_avg = parallel_times.iter().sum::<u128>() as f64 / parallel_times.len() as f64;
    
    // Results
    println!("\n📈 Performance Results:");
    println!("  Sequential average: {:.2}ms", sequential_avg);
    println!("  Parallel average: {:.2}ms", parallel_avg);
    println!("  Speedup: {:.2}x", sequential_avg / parallel_avg);
    println!("  Efficiency: {:.1}%", (sequential_avg / parallel_avg) / rayon::current_num_threads() as f64 * 100.0);
    
    // Per-prompt analysis
    println!("\n📊 Per-prompt Performance:");
    println!("  Sequential: {:.2}ms per prompt", sequential_avg / test_prompts.len() as f64);
    println!("  Parallel: {:.2}ms per prompt", parallel_avg / test_prompts.len() as f64);
    
    // Theoretical vs actual
    let theoretical_speedup = rayon::current_num_threads() as f64;
    let actual_speedup = sequential_avg / parallel_avg;
    println!("\n🎯 Parallelization Analysis:");
    println!("  Theoretical max speedup: {:.2}x", theoretical_speedup);
    println!("  Actual speedup: {:.2}x", actual_speedup);
    println!("  Parallel efficiency: {:.1}%", (actual_speedup / theoretical_speedup) * 100.0);
}

fn main() {
    // Set up maximum parallelization
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(rayon::current_num_threads().max(1));
    
    // Try to build thread pool, but don't panic if already initialized
    match rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global() {
        Ok(()) => println!("🧵 Initialized with {} threads for maximum performance\n", max_threads),
        Err(_) => println!("🧵 Using existing thread pool with {} threads\n", rayon::current_num_threads()),
    }
    
    // Run benchmark
    if std::path::Path::new("model.json").exists() {
        benchmark_single_vs_batch("model.json");
    } else {
        println!("❌ No model.json found. Please run training first.");
        println!("   Run: cargo run --release --bin train");
    }
}
