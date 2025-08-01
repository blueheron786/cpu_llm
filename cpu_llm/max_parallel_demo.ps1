# max_parallel_demo.ps1 - Demonstrate maximum parallelization capabilities

Write-Host "🚀 CPU LLM Maximum Parallelization Demo" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

# Get system information
$cores = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfLogicalProcessors
$physicalCores = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfProcessors
Write-Host "💻 System Info:" -ForegroundColor Cyan
Write-Host "   Physical CPU cores: $physicalCores" -ForegroundColor Yellow
Write-Host "   Logical CPU cores: $cores" -ForegroundColor Yellow
Write-Host "   Available for parallelization: $cores threads" -ForegroundColor Yellow
Write-Host ""

# Check if model exists
if (-not (Test-Path "model.json")) {
    Write-Host "📝 No trained model found. Training with maximum parallelization..." -ForegroundColor Yellow
    Write-Host "   Using all $cores CPU threads for training..." -ForegroundColor Yellow
    
    # Set environment variable for maximum threads
    $env:RAYON_NUM_THREADS = $cores
    
    # Run training with release optimizations
    Write-Host "🏃‍♂️ Running: cargo run --release --bin train" -ForegroundColor Green
    cargo run --release --bin train
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Training failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Training completed!" -ForegroundColor Green
    Write-Host ""
}

# Run inference benchmark
Write-Host "🔥 Running inference benchmark with maximum parallelization..." -ForegroundColor Yellow
Write-Host "   Comparing sequential vs parallel batch inference..." -ForegroundColor Yellow
Write-Host ""

# Set environment variable for maximum threads
$env:RAYON_NUM_THREADS = $cores

Write-Host "🏃‍♂️ Running: cargo run --release --bin bench" -ForegroundColor Green
cargo run --release --bin bench

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Benchmark failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎯 Demo completed! Your CPU LLM is now using maximum parallelization." -ForegroundColor Green
Write-Host "   • Training: Parallel batch processing with $cores threads" -ForegroundColor Yellow
Write-Host "   • Inference: Parallel matrix operations and batch inference" -ForegroundColor Yellow
Write-Host "   • Memory: Optimized buffer reuse and cache-friendly operations" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 Tips for maximum performance:" -ForegroundColor Cyan
Write-Host "   • Use batch inference for multiple prompts" -ForegroundColor White
Write-Host "   • Set RAYON_NUM_THREADS=$cores for consistent performance" -ForegroundColor White
Write-Host "   • Use --release flag for production builds" -ForegroundColor White
Write-Host "   • Monitor CPU usage to verify full utilization" -ForegroundColor White
