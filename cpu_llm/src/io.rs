use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use crate::model::TinyRnnModel;

pub fn save_model<P: AsRef<Path>>(path: P, model: &TinyRnnModel) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write vocab
    for token in &model.vocab {
        writeln!(writer, "{}", token)?;
    }
    writeln!(writer, "---")?;  // Separator between vocab and weights
    
    // Write embeddings
    for row in &model.embedding {
        for &val in row {
            write!(writer, "{} ", val)?;
        }
        writeln!(writer)?;
    }
    
    // Write ff1 weights
    for row in &model.ff1_weights {
        for &val in row {
            write!(writer, "{} ", val)?;
        }
        writeln!(writer)?;
    }
    
    // Write ff1 bias
    for &val in &model.ff1_bias {
        write!(writer, "{} ", val)?;
    }
    writeln!(writer)?;
    
    // Write ff2 weights
    for row in &model.ff2_weights {
        for &val in row {
            write!(writer, "{} ", val)?;
        }
        writeln!(writer)?;
    }
    
    // Write ff2 bias
    for &val in &model.ff2_bias {
        write!(writer, "{} ", val)?;
    }
    writeln!(writer)?;
    
    Ok(())
}

pub fn load_model<P: AsRef<Path>>(path: P) -> std::io::Result<TinyRnnModel> {
    let path = path.as_ref(); // Convert to &Path once
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines().filter_map(Result::ok);
    
    // Read vocab
    let mut vocab = Vec::new();
    while let Some(line) = lines.next() {
        if line == "---" {
            break;
        }
        if !line.trim().is_empty() {
            vocab.push(line);
        }
    }
    
    if vocab.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No vocabulary found in model file",
        ));
    }
    
    // Initialize model with vocab
    let vocab_size = vocab.len();
    let hidden_size = if let Some(line) = lines.next() {
        line.split_whitespace().count()
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid model format: missing embedding data",
        ));
    };
    
    let context_size = 6; // Default context size, adjust as needed
    let mut model = TinyRnnModel::new(vocab, context_size, hidden_size);
    
    // Reset lines iterator to read the file again
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines().filter_map(Result::ok);
    
    // Skip vocab section
    while let Some(line) = lines.next() {
        if line == "---" {
            break;
        }
    }
    
    // Read embeddings
    for i in 0..vocab_size {
        if let Some(line) = lines.next() {
            let values: Vec<f32> = line
                .split_whitespace()
                .filter_map(|s| f32::from_str(s).ok())
                .collect();
            
            if values.len() != hidden_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid embedding size: expected {}, got {}", hidden_size, values.len()),
                ));
            }
            
            for (j, &val) in values.iter().enumerate() {
                model.embedding[i][j] = val;
            }
        }
    }
    
    // Read ff1 weights
    for i in 0..hidden_size {
        if let Some(line) = lines.next() {
            let values: Vec<f32> = line
                .split_whitespace()
                .filter_map(|s| f32::from_str(s).ok())
                .collect();
            
            if values.len() != hidden_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid ff1_weights size at row {}: expected {}, got {}", 
                           i, hidden_size, values.len()),
                ));
            }
            
            for (j, &val) in values.iter().enumerate() {
                model.ff1_weights[i][j] = val;
            }
        }
    }
    
    // Read ff1 bias
    if let Some(line) = lines.next() {
        let values: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| f32::from_str(s).ok())
            .collect();
        
        if values.len() != hidden_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid ff1_bias size: expected {}, got {}", hidden_size, values.len()),
            ));
        }
        
        for (i, &val) in values.iter().enumerate() {
            model.ff1_bias[i] = val;
        }
    }
    
    // Read ff2 weights (stored as [vocab_size][hidden_size] but need [hidden_size][vocab_size])
    let mut ff2_weights_transposed = vec![vec![0.0; vocab_size]; hidden_size];
    for i in 0..vocab_size {
        if let Some(line) = lines.next() {
            let values: Vec<f32> = line
                .split_whitespace()
                .filter_map(|s| f32::from_str(s).ok())
                .collect();
            
            if values.len() != hidden_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid ff2_weights size at row {}: expected {}, got {}", 
                           i, hidden_size, values.len()),
                ));
            }
            
            // Store values in transposed order
            for j in 0..hidden_size {
                ff2_weights_transposed[j][i] = values[j];
            }
        }
    }
    // Copy transposed weights to model
    model.ff2_weights = ff2_weights_transposed;
    
    // Read ff2 bias
    if let Some(line) = lines.next() {
        let values: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| f32::from_str(s).ok())
            .collect();
        
        if values.len() != vocab_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid ff2_bias size: expected {}, got {}", vocab_size, values.len()),
            ));
        }
        
        for (i, &val) in values.iter().enumerate() {
            model.ff2_bias[i] = val;
        }
    }
    
    Ok(model)
}