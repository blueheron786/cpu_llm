# CPU LLM

A lightweight, CPU-native language model built from scratch in Rust. Features a custom hybrid tokenizer (word-level backed by character-level fallback), an RNN-based architecture, and multi-threaded training support using Rayon.

## Features

- ✅ Written entirely in Rust
- ✅ Hybrid word+char tokenization
- ✅ Custom RNN architecture
- ✅ Parallel training using Rayon
- ✅ Minimal external dependencies
- ✅ Designed for small-scale, low-resource environments

## Why?

Most LLMs today require GPUs and massive datasets. This project explores how far we can go with simple architectures, hand-crafted logic, and commodity hardware.

## Usage

```bash
cargo run --release --bin train
```

To run inference:

```bash
cargo run --release --bin infer
```

## Project Structure

- src/tokenizer.rs: Hybrid tokenizer
- src/model.rs: TinyRnnModel definition and logic
- src/train.rs: Training loop
- src/infer.rs: Inference code
- data/: Training corpus
- output/: Saved models

## Requirements

- Rust (latest stable)
- Dataset in plain text (UTF-8)

## Limitations

- Model can't generate very complex output or long text
- Tokenizing will probably eat all your RAM while it's running (takes a couple of minutes)

## License
MIT