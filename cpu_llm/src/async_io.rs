use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use futures_util::Stream;
use futures_util::StreamExt;
use glob::glob;
use thiserror::Error;
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, BufReader};
use tokio::sync::Mutex;

/// The default chunk size for reading files (1MB)
const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

#[derive(Error, Debug)]
pub enum AsyncIoError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Glob pattern error: {0}")]
    Glob(#[from] glob::PatternError),
    
    #[error("Glob error: {0}")]
    GlobError(#[from] glob::GlobError),
    
    #[error("No files found matching pattern")]
    NoFilesFound,
    
    #[error("UTF-8 error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
}

/// Reads a file and returns its contents as a String
async fn read_file(path: &Path) -> Result<String, AsyncIoError> {
    let mut file = File::open(path).await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    Ok(contents)
}

/// Processes files matching the given pattern and yields cleaned text chunks
pub async fn process_files(
    pattern: &str,
) -> Result<Vec<String>, AsyncIoError> {
    let start_time = Instant::now();
    let mut results = Vec::new();
    
    // Get all matching files
    let paths: Vec<_> = glob(pattern)?.filter_map(Result::ok).collect();
    
    if paths.is_empty() {
        return Err(AsyncIoError::NoFilesFound);
    }
    
    // Process each file
    for path in paths {
        let content = read_file(&path).await?;
        let cleaned = clean_text(&content);
        results.push(cleaned);
    }
    
    let elapsed = start_time.elapsed();
    let total_bytes: usize = results.iter().map(|s| s.len()).sum();
    let mb_processed = total_bytes as f64 / (1024.0 * 1024.0);
    
    println!(
        "Processed {} files ({:.2} MB) in {:.2?}",
        results.len(),
        mb_processed,
        elapsed
    );
    
    Ok(results)
}

/// Finds a good split point in the text, preferably at whitespace
fn find_split_point(text: &str, max_len: usize) -> usize {
    if text.len() <= max_len {
        return text.len();
    }
    
    // Try to find a whitespace to split at
    if let Some(pos) = text[..max_len].rfind(|c: char| c.is_whitespace()) {
        return pos + 1; // Include the whitespace in the first part
    }
    
    // If no whitespace found, split at max_len (may split in the middle of a word)
    max_len
}

/// Cleans and normalizes text for processing
pub fn clean_text(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut prev_was_space = false;

    for c in text.chars() {
        // Normalize newlines and other whitespace
        let c = match c {
            '\n' | '\r' | '\t' | '\u{00A0}' => ' ', // Common whitespace chars
            _ if c.is_whitespace() => ' ', // Any other whitespace
            _ => c.to_ascii_lowercase(), // Convert to lowercase
        };

        if c == ' ' {
            if !prev_was_space {
                cleaned.push(' ');
                prev_was_space = true;
            }
        } else {
            cleaned.push(c);
            prev_was_space = false;
        }
    }

    cleaned.trim().to_string()
}
