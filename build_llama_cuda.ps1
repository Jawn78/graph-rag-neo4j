<#
PowerShell helper to build llama.cpp with CUDA and copy GPU-enabled DLLs into the server venv.
Run from an admin/developer PowerShell where Visual Studio & CUDA are in PATH.
#>

# Configuration
$RepoUrl = 'https://github.com/ggerganov/llama.cpp.git'
$CloneDir = Join-Path $PWD 'llama.cpp'
$VenvLibDir = Join-Path $PWD '.venv-server\Lib\site-packages\llama_cpp\lib'
$CudaArch = '75;80;86'  # RTX 2080 Super = 75, RTX 30xx = 86, RTX 40xx = 89

# Error handling function
function Exit-WithError {
    param([string]$Message)
    Write-Error $Message
    if ($PSCommandPath -eq $MyInvocation.ScriptName) {
        exit 1
    } else {
        return $false
    }
}

# Check CUDA installation
Write-Host "Checking CUDA installation..."
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Exit-WithError "NVCC not found. Please ensure CUDA Toolkit is installed and in PATH."
}
Write-Host "Found CUDA: $($nvcc.Version)"

# Check for other required commands
$requiredCommands = @("cmake", "git", "cl")
foreach ($cmd in $requiredCommands) {
    $cmdPath = Get-Command $cmd -ErrorAction SilentlyContinue
    if (-not $cmdPath) {
        Exit-WithError "Required command '$cmd' not found in PATH. Please install it and re-run."
    }
}

# Clone or update repo
if (-not (Test-Path $CloneDir)) {
    Write-Host "Cloning llama.cpp repository..."
    git clone $RepoUrl $CloneDir
    if (-not $?) {
        Exit-WithError "Failed to clone repository."
    }
} else {
    Write-Host "Updating existing llama.cpp repository..."
    Push-Location $CloneDir
    git fetch origin
    git reset --hard origin/master
    if (-not $?) {
        Pop-Location
        Exit-WithError "Failed to update repository."
    }
    Pop-Location
}

# Clean and rebuild
Push-Location $CloneDir
Write-Host "Cleaning previous build..."
Remove-Item -Path build -Recurse -Force -ErrorAction SilentlyContinue
mkdir build | Out-Null

Set-Location build

# Set up CUDA environment
$Env:CUDACXX = "nvcc"
$Env:CUDAHOSTCXX = "cl"
$Env:CUDAFLAGS = "--allow-unsupported-compiler"

$cmakeArgs = @(
    '..',
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DUSE_CUBLAS=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler",
    "-DLLAMA_CUBLAS=ON"
)

# Configure with CMake
Write-Host "Configuring with CMake: cmake $($cmakeArgs -join ' ')"
cmake @cmakeArgs
if (-not $?) {
    Exit-WithError "CMake configuration failed."
}

# Build
Write-Host "Building (cmake --build . --config Release -j)"
cmake --build . --config Release -j
if (-not $?) {
    Exit-WithError "Build failed."
}

# Verify CUDA support was enabled
Write-Host "Verifying CUDA support..."
$mainExe = Join-Path (Get-Location) "bin\Release\main.exe"
if (-not (Test-Path $mainExe)) {
    Exit-WithError "Main executable not found. Build may have failed."
}

# Find and copy built DLLs (check Release folder first)
$searchPaths = @(
    "bin\Release",
    "Release",
    "ggml-cuda\Release",
    "."
)

$requiredDlls = @(
    "ggml-cuda.dll",
    "llama.dll"
)

$foundDlls = @{}
foreach ($path in $searchPaths) {
    $fullPath = Join-Path (Get-Location) $path
    Write-Host "Searching for DLLs in: $fullPath"
    $dlls = Get-ChildItem -Path $fullPath -Filter *.dll -ErrorAction SilentlyContinue
    if ($dlls) {
        foreach ($dll in $dlls) {
            if ($dll.Name -in $requiredDlls -and -not $foundDlls.ContainsKey($dll.Name)) {
                $foundDlls[$dll.Name] = $dll.FullName
            }
        }
    }
}

# Verify we found all required DLLs
$missingDlls = $requiredDlls | Where-Object { -not $foundDlls.ContainsKey($_) }
if ($missingDlls) {
    Exit-WithError "Missing required DLLs: $($missingDlls -join ', ')"
}

# Create venv lib directory if it doesn't exist
if (-not (Test-Path $VenvLibDir)) {
    Write-Host "Creating venv lib dir: $VenvLibDir"
    New-Item -ItemType Directory -Path $VenvLibDir -Force | Out-Null
}

# Backup existing DLLs
$backupDir = "$VenvLibDir.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (Test-Path "$VenvLibDir\*.dll") {
    Write-Host "Backing up existing DLLs to $backupDir"
    Copy-Item -Path $VenvLibDir -Destination $backupDir -Recurse -Force
}

# Copy new DLLs
foreach ($dll in $foundDlls.Keys) {
    $dest = Join-Path $VenvLibDir $dll
    Write-Host "Copying $($foundDlls[$dll]) -> $dest"
    Copy-Item -Path $foundDlls[$dll] -Destination $dest -Force
}

Pop-Location

Write-Host "`nBuild completed successfully!"
Write-Host "To use with Python, activate your venv and run:"
Write-Host "pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121"