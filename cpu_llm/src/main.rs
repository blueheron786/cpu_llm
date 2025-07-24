use cpu_llm::{model::{run_inference}, io::load_model};

fn main() {
    let model = load_model("model.json").expect("Failed to load model.");

    let prompt = "It was a dark and stormy night. ";
    let output = run_inference(prompt, 100, &model);
    println!("Generated: {}", output);
}
