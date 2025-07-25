# CPU LLM

## :warning: Highly experimental :warning:

LLM built from the ground-up to use CPUs (multiple threads) and CPU RAM instead of GPUs and GPU RAM.

Unlike other existing solutions (circa 2025), the aim is not to take GPU-centric models and limit them to work on CPUs; it's to build something from the ground-up that's made to run efficiently on CPUs.

As of right now, it's optimized to run as fast as possible, while trying to keep memory from being consumed in an unbound manner. It generally stays within 2GB of memory while training, albeit in some places, memory usage increases logarithmetically.

This is a rapidly-changing, highly experimental fun project. Feel free to open a PR and add your personal touch!

## Usage

- Download TinyStories v2 (training data) and put it in `cpu_llm/data`
- Drop a bunch of text files into the `data` directory
- Run `cargo run --release --bin train` to generate the training model using maximum efficiency.
- Run `cargo run --bin main` to generate text based on a fixed prompt

Add your training data files to `data` as plaintext. Feel free to organize it into subfolders as you see fit. Ensure the text files contain plain text, preferably representative of your target language or domain.

To retrain, simply add/remove files and run the training again.

## Architecture

- `hybrid_train.rs`: Entry point, runs training and generates the model using texts in `data`. The hybrid training uses a hybrid tokenizer that uses a character and wordpiece (subword) approach.
- `main.rs`: Entry point, runs inference (prompt -> response)
- `model.rs`: It's the model (structure and weights)
- `io.rs`: Model persistence (save/load weights)

`train.rs` uses the old, character-based tokenizer. 

