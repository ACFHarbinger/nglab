@echo off
setlocal enabledelayedexpansion

:: Default values
set VERBOSE=1
set MANAGER=uv

:: Simple argument parsing
:parse_args
if "%~1"=="" goto execute
if "%~1"=="-silent" (
    set VERBOSE=0
) else (
    set MANAGER=%~1
)
shift
goto parse_args

:execute
:: Set echo based on verbose mode
if %VERBOSE% equ 1 (
    @echo on
) else (
    @echo off
)

echo Using manager: %MANAGER%

:: UV manager section
if "%MANAGER%"=="uv" goto uv_section
if "%MANAGER%"=="conda" goto conda_section  
if "%MANAGER%"=="venv" goto venv_section

echo Error: unknown manager selected: %MANAGER%
exit /b 1

:uv_section
echo Setting up with UV...
uv --version >nul 2>&1
if !errorlevel! == 1 (
    echo Warning: uv is not installed or not in PATH
    echo Installing uv...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.cargo\env.bat" call "%USERPROFILE%\.cargo\env.bat"
)

uv venv env\wsr
call env\wsr\Scripts\activate.bat
uv add -r env\requirements.txt
goto success

:conda_section
echo Setting up with Conda...
conda --version >nul 2>&1
if !errorlevel! == 1 (
    echo Warning: conda is not installed or not in PATH
    echo Installing conda...
    
    echo Downloading Anaconda installer...
    powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe' -OutFile 'Anaconda3-installer.exe'"
    
    if exist Anaconda3-installer.exe (
        echo Installing Anaconda (this may take a few minutes)...
        Anaconda3-installer.exe /S /D=%USERPROFILE%\Anaconda3
    ) else (
        echo Error: Failed to download Anaconda installer
        exit /b 1
    )
    
    if exist "%USERPROFILE%\Anaconda3" (
        set "PATH=%USERPROFILE%\Anaconda3;%USERPROFILE%\Anaconda3\Scripts;%USERPROFILE%\Anaconda3\Library\bin;%PATH%"
        echo Initializing conda...
        call conda init cmd.exe
    ) else (
        echo Error: Anaconda installation failed
        exit /b 1
    )
    
    if exist Anaconda3-installer.exe del Anaconda3-installer.exe
)

if exist "env\environment.yml" (
    conda env create --file env\environment.yml -y
) else (
    conda create --name wsr python=3.11 -y
)

call conda activate wsr
goto success

:venv_section
echo Setting up with venv...
python --version >nul 2>&1
if !errorlevel! == 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

python -m venv env\.wsr
call env\.wsr\Scripts\activate.bat
pip install -r env\requirements.txt
goto success

:success
echo Setup completed successfully with %MANAGER%
endlocal