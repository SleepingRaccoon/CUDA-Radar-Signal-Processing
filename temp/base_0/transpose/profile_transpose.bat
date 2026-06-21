@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Transpose Kernel Profiling Script
echo   Tools: Nsight Compute (ncu) + Nsight Systems (nsys)
echo ============================================================
echo.

REM -----------------------------------------------------------------
REM 1. Check tools
REM -----------------------------------------------------------------
where ncu >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] ncu not found in PATH.
    echo         Install CUDA Toolkit 11.0+ or add Nsight Compute to PATH.
    exit /b 1
)

where nsys >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN]  nsys not found in PATH. Timeline profiling skipped.
    set HAS_NSYS=0
) else (
    set HAS_NSYS=1
)

echo ncu  : found
if !HAS_NSYS! equ 1 echo nsys : found
echo.

REM -----------------------------------------------------------------
REM 2. Paths
REM -----------------------------------------------------------------
set "TEMP_DIR=..\..\..\temp"
set "PROF_DIR=profiling"
set "SCALAR_EXE=%TEMP_DIR%\transpose.exe"
set "F4_EXE=%TEMP_DIR%\demo1.exe"

if not exist "%PROF_DIR%" mkdir "%PROF_DIR%"

REM -----------------------------------------------------------------
REM 3. Build profiling binaries (if not already built)
REM -----------------------------------------------------------------
echo [1/4] Checking binaries...
echo.

if not exist "%SCALAR_EXE%" (
    echo   Building scalar transpose...
    pushd "%TEMP_DIR%"
    nvcc -o transpose.exe transpose.cu
    if %errorlevel% neq 0 (
        echo   [ERROR] Failed to build scalar transpose
        popd
        exit /b 1
    )
    popd
    echo   Done.
) else (
    echo   Scalar binary:  found
)

if not exist "%F4_EXE%" (
    echo   Building float4 transpose...
    pushd "%TEMP_DIR%"
    nvcc -o demo1.exe demo1.cu
    if %errorlevel% neq 0 (
        echo   [ERROR] Failed to build float4 transpose
        popd
        exit /b 1
    )
    popd
    echo   Done.
) else (
    echo   Float4 binary:  found
)

echo.

REM =================================================================
REM 4. Nsight Compute -- per-kernel detailed analysis
REM =================================================================
REM
REM   --kernel-name regex: selects specific kernel(s) to profile.
REM   --section:          picks which metric sections to collect.
REM   -o <file>:          output .ncu-rep report file.
REM   --launch-count 1:   only profile the first matching launch
REM                         (after warm-up, which the binary handles).
REM
REM   MemoryWorkloadAnalysis:  L1/L2/DRAM throughput, cache hit rate
REM   Occupancy:               theoretical vs achieved occupancy
REM   SpeedOfLight:            fraction of peak compute/memory bandwidth
REM
REM   Output files go into profiling/*.ncu-rep
REM =================================================================
echo [2/4] Nsight Compute -- scalar kernels...
echo.

REM Profile each scalar kernel variant individually for clean reports.
REM The binary runs them in order: v11, v12, v2, v3, v4.
REM Each is launched once (after warm-up), so --launch-count 1 works.

set KERNELS=v11 v12 v2 v3 v4
for %%k in (%KERNELS%) do (
    echo   Profiling transpose_%%k ...
    ncu ^
        --kernel-name "regex:transpose_%%k" ^
        --launch-count 1 ^
        --launch-skip 0 ^
        --set full ^
        --section MemoryWorkloadAnalysis ^
        --section Occupancy ^
        --section SpeedOfLight ^
        -o "%PROF_DIR%\scalar_%%k" ^
        "%SCALAR_EXE%" >nul 2>&1

    if %errorlevel% equ 0 (
        echo     ^> %PROF_DIR%\scalar_%%k.ncu-rep
    ) else (
        echo     [WARN] ncu failed for transpose_%%k
    )
)

echo.

REM -------- float4 kernels --------
echo [3/4] Nsight Compute -- float4 kernels...
echo.

set F4_KERNELS=v1 v2 v3
for %%k in (%F4_KERNELS%) do (
    echo   Profiling transpose_float4_%%k ...
    ncu ^
        --kernel-name "regex:transpose_float4_%%k" ^
        --launch-count 1 ^
        --launch-skip 0 ^
        --set full ^
        --section MemoryWorkloadAnalysis ^
        --section Occupancy ^
        --section SpeedOfLight ^
        -o "%PROF_DIR%\float4_%%k" ^
        "%F4_EXE%" >nul 2>&1

    if %errorlevel% equ 0 (
        echo     ^> %PROF_DIR%\float4_%%k.ncu-rep
    ) else (
        echo     [WARN] ncu failed for transpose_float4_%%k
    )
)

echo.

REM =================================================================
REM 5. Nsight Systems -- timeline trace
REM =================================================================
if !HAS_NSYS! equ 1 (
    echo [4/4] Nsight Systems -- timeline traces...
    echo.

    echo   Profiling scalar binary (full timeline)...
    nsys profile ^
        --stats=true ^
        --trace=cuda,nvtx,osrt ^
        -o "%PROF_DIR%\timeline_scalar" ^
        "%SCALAR_EXE%" >nul 2>&1

    if %errorlevel% equ 0 (
        echo     ^> %PROF_DIR%\timeline_scalar.nsys-rep
    ) else (
        echo     [WARN] nsys failed for scalar binary
    )

    echo   Profiling float4 binary (full timeline)...
    nsys profile ^
        --stats=true ^
        --trace=cuda,nvtx,osrt ^
        -o "%PROF_DIR%\timeline_float4" ^
        "%F4_EXE%" >nul 2>&1

    if %errorlevel% equ 0 (
        echo     ^> %PROF_DIR%\timeline_float4.nsys-rep
    ) else (
        echo     [WARN] nsys failed for float4 binary
    )

    echo.
)

REM =================================================================
REM 6. Summary
REM =================================================================
echo ============================================================
echo   Profiling complete.
echo ============================================================
echo.
echo   Output files in: %PROF_DIR%\
echo.
echo   .ncu-rep files -- open with Nsight Compute GUI:
echo     ncu-ui %PROF_DIR%\scalar_v3.ncu-rep
echo.
echo   .nsys-rep files -- open with Nsight Systems GUI:
echo     nsys-ui %PROF_DIR%\timeline_scalar.nsys-rep
echo.
echo   Quick command-line summary:
echo     ncu --print-summary all %%PROF_DIR%%\scalar_v3.ncu-rep
echo.

exit /b 0
