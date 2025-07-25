use cpu_llm::{model::{run_inference}, io::load_model};

fn main() {
    if std::path::Path::new("model.json").exists() {
        let model = load_model("model.json").expect("Failed to load model.");
        let prompt = "It was a dark and stormy night. ";
        let output = run_inference(&model, prompt, 100);
        println!("Generated: {}", output);
    } else {
        println!("No model.json found. Please run training first.");
    }
}
