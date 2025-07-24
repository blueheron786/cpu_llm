# CPU LLM

## :warning: Highly experimental :warning

LLM built from the ground-up to use CPUs (multiple threads) and CPU RAM instead of GPUs and GPU RAM.

Unlike other existing solutions (circa 2025), the aim is not to take GPU-centric models and limit them to work on CPUs; it's to build something from the ground-up that's made to run efficiently on CPUs.

This is a rapidly-changing, highly experimental fun project. Feel free to open a PR and add your personal touch!

## Usage

- Drop a bunch of text files into the `data` directory
- Run `cargo run --bin train` to generate the training model
- Run `cargo run --bin main` to generate text based on a fixed prompt

Add your training data files to `data` as plaintext. Feel free to organize it into subfolders as you see fit. Ensure the text files contain plain text, preferably representative of your target language or domain.

To retrain, simply add/remove files and run the training again.

## Architecture

- `train.rs`: Entry point, runs training and generates the model using texts in `data`
- `main.rs`: Entry point, runs inference (prompt -> response)
- `model.rs`: It's the model (structure and weights)
- `io.rs`: Model persistence (save/load weights)

