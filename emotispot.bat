@echo off

REM Change directory to Desktop
cd %USERPROFILE%\Desktop\EmotiSpot


REM Activate the conda environment
CALL conda activate

REM Check if the environment was activated successfully
IF ERRORLEVEL 1 (
    echo Failed to activate Conda environment. Ensure it exists and Conda is installed correctly.
    EXIT /B 1
)

REM Install the requirements
pip install -r requirements.txt

REM Check if the requirements were installed successfully
IF ERRORLEVEL 1 (
    echo Failed to install requirements. Ensure the requirements.txt file is correct and all dependencies are available.
    EXIT /B 1
)

REM Run the Python script
python emotion_app.py

REM Check if the script ran successfully
IF ERRORLEVEL 1 (
    echo Failed to run the Python script. Ensure the script is correct and all dependencies are available.
    EXIT /B 1
)

echo Script executed successfully.
