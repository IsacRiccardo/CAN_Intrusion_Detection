@echo off
:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python before running this script.
    exit /b
)

:: Create a virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    exit /b
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    exit /b
)

:: Install the requirements
echo Installing requirements from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements.
    exit /b
)

:: Run the Python script
echo Running the Python script...
python Scripts/main.py
if %ERRORLEVEL% NEQ 0 (
    echo Script execution failed.
    exit /b
)

:: Deactivate the virtual environment
echo Deactivating virtual environment...
deactivate

:: Done
echo Setup and script execution complete!
pause
