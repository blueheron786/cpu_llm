import glob
import os
from multiprocessing import Pool
from functools import partial

def analyze_file(filepath):
    try:
        # Read in chunks for better memory usage
        chunk_size = 1024 * 1024  # 1MB chunks
        chars = 0
        words = 0
        lines = 0
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                    
                chars += len(chunk)
                # Quick word count - split on spaces, less accurate but faster
                words += len(chunk.split())
                lines += chunk.count('\n')
        
        return {
            'filepath': os.path.relpath(filepath),
            'chars': chars,
            'words': words,
            'lines': lines + 1  # +1 for last line without newline
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def analyze_files(pattern):
    files = glob.glob(pattern, recursive=True)
    
    print("📊 Fast Data Analysis:")
    print("-" * 50)
    print(f"Processing {len(files)} files in parallel...")
    
    # Use all available CPU cores for parallel processing
    with Pool() as pool:
        results = pool.map(analyze_file, files)
    
    # Filter out None results from errors
    results = [r for r in results if r]
    
    # Calculate totals
    total_chars = sum(r['chars'] for r in results)
    total_words = sum(r['words'] for r in results)
    total_lines = sum(r['lines'] for r in results)
    
    # Print individual file stats
    for r in results:
        print(f"\nFile: {r['filepath']}")
        print(f"  - Characters: {r['chars']:,}")
        print(f"  - Words: {r['words']:,}")
        print(f"  - Lines: {r['lines']:,}")
    
    print("\n" + "=" * 50)
    print(f"Total Statistics:")
    print(f"  - Files: {len(results)}")
    print(f"  - Total Characters: {total_chars:,}")
    print(f"  - Total Words: {total_words:,}")
    print(f"  - Total Lines: {total_lines:,}")
    print(f"  - Average Words per Line: {total_words/total_lines:.1f}")
    print("=" * 50)

if __name__ == '__main__':
    # Use absolute path
    print("Starting analysis...")
    analyze_files("data/**/*.txt")
