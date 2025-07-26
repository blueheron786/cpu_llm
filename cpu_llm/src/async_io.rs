use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use async_stream::stream;
use futures_util::Stream;
use futures_util::StreamExt;
use glob::glob;
use thiserror::Error;
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt};
use tokio::sync::Semaphore;

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
    
    #[error("Semaphore error: {0}")]
    Semaphore(String),
}

/// Asynchronously reads a file and returns its contents as a String.
async fn read_file_async(path: &Path) -> Result<String, AsyncIoError> {
    let mut file = File::open(path).await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    Ok(contents)
}

/// Streams file contents as they're being read, with a configurable concurrency level.
pub fn stream_files(
    pattern: &'static str,
    concurrency: usize,
) -> impl Stream<Item = Result<(PathBuf, String), AsyncIoError>> + 'static {
    let pattern = Arc::new(pattern.to_string());
    
    stream! {
        // First, collect all file paths that match the pattern
        let paths: Vec<_> = match glob(pattern.as_str()) {
            Ok(paths) => paths.filter_map(Result::ok).collect(),
            Err(e) => {
                yield Err(AsyncIoError::from(e));
                return;
            }
        };

        if paths.is_empty() {
            yield Err(AsyncIoError::NoFilesFound);
            return;
        }

        // Process files with the given concurrency level
        let semaphore = Arc::new(Semaphore::new(concurrency));
        
        // Create a stream of file processing tasks
        let mut tasks = futures::stream::FuturesUnordered::new();
        
        // Spawn initial set of tasks up to concurrency limit
        for path in paths.into_iter().take(concurrency) {
            let semaphore = semaphore.clone();
            tasks.push(tokio::spawn(async move {
                let _permit = semaphore.acquire_owned().await
                    .map_err(|e| AsyncIoError::Semaphore(e.to_string()))?;
                let content = read_file_async(&path).await?;
                Ok((path, content))
            }));
        }
        
        // Process tasks as they complete
        while let Some(result) = tasks.next().await {
            match result {
                Ok(Ok((path, content))) => {
                    yield Ok((path, content));
                }
                Ok(Err(e)) => {
                    yield Err(e);
                }
                Err(join_err) => {
                    yield Err(AsyncIoError::Io(io::Error::new(
                        io::ErrorKind::Other,
                        join_err.to_string(),
                    )));
                }
            }
        }
    }
}

/// Processes files in chunks and yields cleaned text chunks
pub async fn process_files(
    pattern: &'static str,
    chunk_size: usize,
    concurrency: usize,
) -> impl Stream<Item = String> + 'static {
    stream! {
        let start_time = Instant::now();
        let mut file_count = 0;
        let mut total_bytes = 0;
        let mut current_chunk = String::with_capacity(chunk_size);
        
        let mut stream = Box::pin(stream_files(pattern, concurrency));
        
        while let Some(result) = stream.next().await {
            match result {
                Ok((path, content)) => {
                    file_count += 1;
                    total_bytes += content.len();
                    
                    // Clean the text
                    let cleaned = clean_text(&content);
                    
                    // Add to current chunk
                    current_chunk.push_str(&cleaned);
                    current_chunk.push(' '); // Separate files with space
                    
                    // Yield chunks of the desired size, ensuring we don't split UTF-8 characters
                    while current_chunk.len() >= chunk_size {
                        // Find the last character boundary before or at chunk_size
                        let split_point = current_chunk
                            .char_indices()
                            .take_while(|&(i, _)| i <= chunk_size)
                            .last()
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        
                        // Split at the character boundary
                        let chunk = current_chunk.drain(..split_point).collect::<String>();
                        if !chunk.is_empty() {
                            yield chunk;
                        }
                    }
                    
                    if file_count % 100 == 0 {
                        let elapsed = start_time.elapsed();
                        let mbps = (total_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();
                        println!(
                            "Processed {} files ({} MB, {:.2} MB/s)",
                            file_count,
                            total_bytes / 1024 / 1024,
                            mbps
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Error processing file: {}", e);
                }
            }
        }
        
        // Yield any remaining data
        if !current_chunk.is_empty() {
            yield current_chunk;
        }
        
        let elapsed = start_time.elapsed();
        let mbps = (total_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64().max(0.001);
        println!(
            "Processed {} files ({} MB, {:.2} MB/s, {:.2?} total)",
            file_count,
            total_bytes / 1024 / 1024,
            mbps,
            elapsed
        );
    }
}

// Clean text by normalizing spaces and lowercasing
pub(crate) fn clean_text(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut prev_was_space = false;

    for c in text.chars() {
        let c = match c {
            '\n' | '\r' => ' ',
            _ => c.to_ascii_lowercase(),
        };

        if c.is_whitespace() {
            if !prev_was_space {
                cleaned.push(' ');
                prev_was_space = true;
            }
        } else {
            cleaned.push(c);
            prev_was_space = false;
        }
    }

    // Trim trailing space if any
    if cleaned.ends_with(' ') {
        cleaned.pop();
    }

    cleaned
}
