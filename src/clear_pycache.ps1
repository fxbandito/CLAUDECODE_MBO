# PowerShell script to clear all __pycache__ and .mypy_cache directories
# Run this script from the src folder

# Get current directory
$currentPath = Get-Location
Write-Host "Clearing cache directories in: $currentPath" -ForegroundColor Cyan
Write-Host ""

# ============================================
# 1. Clear __pycache__ directories
# ============================================
Write-Host "=== __pycache__ ===" -ForegroundColor Cyan
$pycacheDirectories = Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory

if ($pycacheDirectories.Count -eq 0) {
    Write-Host "No __pycache__ directories found." -ForegroundColor Yellow
} else {
    Write-Host "Found $($pycacheDirectories.Count) __pycache__ directories:" -ForegroundColor Green
    $pycacheDirectories | ForEach-Object { Write-Host "  - $($_.FullName)" -ForegroundColor Gray }

    $pycacheDirectories | Remove-Item -Recurse -Force
    Write-Host "All __pycache__ directories removed successfully!" -ForegroundColor Green
}

Write-Host ""

# ============================================
# 2. Clear .mypy_cache directories
# ============================================
Write-Host "=== .mypy_cache ===" -ForegroundColor Cyan

# Check in current directory and parent directory (project root)
$mypyCachePaths = @()

# Current directory
$currentMypyCache = Join-Path -Path $currentPath -ChildPath ".mypy_cache"
if (Test-Path $currentMypyCache) {
    $mypyCachePaths += $currentMypyCache
}

# Parent directory (project root)
$parentPath = Split-Path -Path $currentPath -Parent
$parentMypyCache = Join-Path -Path $parentPath -ChildPath ".mypy_cache"
if (Test-Path $parentMypyCache) {
    $mypyCachePaths += $parentMypyCache
}

# Also search recursively for any .mypy_cache
$recursiveMypyCache = Get-ChildItem -Path . -Include ".mypy_cache" -Recurse -Directory -ErrorAction SilentlyContinue
if ($recursiveMypyCache) {
    $mypyCachePaths += $recursiveMypyCache.FullName
}

# Remove duplicates
$mypyCachePaths = $mypyCachePaths | Select-Object -Unique

if ($mypyCachePaths.Count -eq 0) {
    Write-Host "No .mypy_cache directories found." -ForegroundColor Yellow
} else {
    Write-Host "Found $($mypyCachePaths.Count) .mypy_cache directories:" -ForegroundColor Green
    $mypyCachePaths | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }

    foreach ($cachePath in $mypyCachePaths) {
        Remove-Item -Path $cachePath -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "All .mypy_cache directories removed successfully!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Cache cleanup complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
