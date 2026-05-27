@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

:: ========================================================
:: 1. Environment Check and Activation
:: ========================================================
if /i "%KINESIS_DEBUG%"=="1" echo [DEBUG] enter run.bat
set "EXIT_CODE=0"
set "AUTO_PAUSE=1"
set "DO_INSTALL=0"
set "VENV_DIR=%CD%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_CREATED=0"

:parse_global_args
if /i "%KINESIS_DEBUG%"=="1" echo [DEBUG] parse_global_args: 1="%~1" 2="%~2"
if /i "%~1"=="--no-pause" (
    set "AUTO_PAUSE=0"
    shift
    goto parse_global_args
)
if /i "%~1"=="--pause" (
    set "AUTO_PAUSE=1"
    shift
    goto parse_global_args
)
if /i "%~1"=="--install" (
    set "DO_INSTALL=1"
    shift
    goto parse_global_args
)

if /i "%KINESIS_DEBUG%"=="1" echo [DEBUG] after global args: 1="%~1"
if not exist "%PYTHON%" (
    echo [INFO] Virtual environment not found, creating...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create or activate virtual environment.
        set "EXIT_CODE=1"
        goto end
    )
    set "VENV_CREATED=1"
)

if /i "%KINESIS_DEBUG%"=="1" echo [DEBUG] python path: "%PYTHON%"
if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment is missing: %PYTHON%
    set "EXIT_CODE=1"
    goto end
)

if /i "%KINESIS_DEBUG%"=="1" echo [DEBUG] venv ok
if "%VENV_CREATED%"=="1" set "DO_INSTALL=1"
if "%DO_INSTALL%"=="1" (
    call :install_deps
    if errorlevel 1 (
        set "EXIT_CODE=1"
        goto end
    )
)

:: ========================================================
:: 2. Task Distribution
:: ========================================================
set TASK=%~1
set ARG1=%~1

:: If no args, run default imitation test
if "%TASK%"=="" goto run_imitation_default

:: If first arg starts with "-" (e.g. --mode), assume imitation task args
if "!ARG1:~0,1!"=="-" goto run_imitation_args

:: Remove the first argument (task name)
shift

if /i "%TASK%"=="imitation" goto run_imitation_args
if /i "%TASK%"=="directional" goto run_directional
if /i "%TASK%"=="target" goto run_target
if /i "%TASK%"=="t2m" goto run_t2m
if /i "%TASK%"=="help" goto show_help

:: Unknown task
echo [ERROR] Unknown task: %TASK%
set "EXIT_CODE=1"
goto show_help

:: ========================================================
:: 3. Task Logic
:: ========================================================

:run_imitation_default
set MODE=test
set HEADLESS=False
goto run_imitation_exec

:run_imitation_args
:: Parse arguments
set MODE=test
set HEADLESS=False

:parse_imitation_loop
if "%~1"=="" goto run_imitation_exec
if "%~1"=="--mode" (
    set MODE=%~2
    shift
    shift
    goto parse_imitation_loop
)
if "%~1"=="--headless" (
    set HEADLESS=%~2
    shift
    shift
    goto parse_imitation_loop
)
:: If argument is not recognized, skip it (or pass it through if we were smarter)
shift
goto parse_imitation_loop

:run_imitation_exec
if "%MODE%"=="train" (
    set MOTION_FILE=data/kit_train_motion_dict.pkl
    set INITIAL_POSE_FILE=data/initial_pose/initial_pose_train.pkl
) else (
    if "%MODE%"=="test" (
        set MOTION_FILE=data/kit_test_motion_dict.pkl
        set INITIAL_POSE_FILE=data/initial_pose/initial_pose_test.pkl
    ) else (
        echo [ERROR] Invalid mode: %MODE%. Use 'train' or 'test'.
        set "EXIT_CODE=1"
        goto end
    )
)

echo [INFO] Running Imitation Task
echo   Mode: %MODE%
echo   Headless: %HEADLESS%
echo   Motion File: %MOTION_FILE%

"%PYTHON%" src/run.py exp_name=kinesis-moe-imitation ^
    epoch=-1 ^
    run=eval_run ^
    run.headless=%HEADLESS% ^
    run.motion_file=%MOTION_FILE% ^
    run.initial_pose_file=%INITIAL_POSE_FILE% ^
    env.termination_distance=0.5

if errorlevel 1 (
    echo [ERROR] Execution failed.
    set "EXIT_CODE=1"
    goto end
)

goto end

:run_directional
echo [INFO] Running Directional Task
"%PYTHON%" src/run.py exp_name=kinesis-target-goal-reach ^
    run=eval_run ^
    learning=directional ^
    epoch=-1 ^
    run.headless=False ^
    run.im_eval=False

if errorlevel 1 (
    echo [ERROR] Execution failed.
    set "EXIT_CODE=1"
    goto end
)
goto end

:run_target
echo [INFO] Running Target Reach Task
"%PYTHON%" src/run.py exp_name=kinesis-target-goal-reach ^
    run=eval_run ^
    learning=pointgoal ^
    epoch=-1 ^
    run.headless=False

if errorlevel 1 (
    echo [ERROR] Execution failed.
    set "EXIT_CODE=1"
    goto end
)
goto end

:run_t2m
set MOTION_FILE=%~1
if "%MOTION_FILE%"=="" (
    echo [ERROR] Motion file required for t2m task.
    echo Usage: run.bat t2m [motion_file]
    set "EXIT_CODE=1"
    goto end
)
echo [INFO] Running Text-to-Motion Task
echo   Motion File: %MOTION_FILE%
"%PYTHON%" src/run.py exp_name=kinesis-moe-imitation ^
    epoch=-1 ^
    run=t2m ^
    run.motion_file=%MOTION_FILE% ^
    env.termination_distance=0.5

if errorlevel 1 (
    echo [ERROR] Execution failed.
    set "EXIT_CODE=1"
    goto end
)
goto end

:show_help
echo Usage: run.bat [task] [options]
echo.
echo Tasks:
echo   imitation (default)   Run imitation task. Options: --mode [train^|test] --headless [True^|False]
echo   directional           Run directional task
echo   target                Run target reach task
echo   t2m                   Run text-to-motion task
echo.
echo Examples:
echo   run.bat
echo   run.bat imitation --mode train
echo   run.bat directional
goto end

:install_deps
echo [INFO] Installing dependencies...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    exit /b 1
)
"%PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch (cu118).
    exit /b 1
)
echo [INFO] Environment installation complete.
echo.
exit /b 0

:end
if "%AUTO_PAUSE%"=="1" pause
endlocal & exit /b %EXIT_CODE%
