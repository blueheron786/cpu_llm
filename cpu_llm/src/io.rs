use std::fs::{File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use crate::model::TinyRnnModel;
use serde_json;

pub fn save_model<P: AsRef<Path>>(path: P, model: &TinyRnnModel) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, model)?;
    Ok(())
}

pub fn load_model<P: AsRef<Path>>(path: P) -> std::io::Result<TinyRnnModel> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model = serde_json::from_reader(reader)?;
    Ok(model)
}