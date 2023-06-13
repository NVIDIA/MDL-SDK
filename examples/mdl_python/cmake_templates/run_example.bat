::*****************************************************************************
:: Copyright 2023 NVIDIA Corporation. All rights reserved.
::*****************************************************************************
@echo off
@IGNORE_THE_FOLLOWING_COMMENT@
:: This file is generated during CMake Configuration.
:: If you want to regenerate it, delete this file and run CMake Configure again.

:: On Windows this batch file can be used to quickly run the example.
:: Alternatively, on every platform, you can open the Visual Studio Code workspace:
::   @CMAKE_BINARY_DIR@/mdl_python_examples.code-workspace

call @CMAKE_CURRENT_BINARY_DIR@/../env/@_CONFIG@/setup_env.bat

echo Python:     %PYTHON_BINARY%
for /f "tokens=1-2" %%i in ('%PYTHON_BINARY% --version') do set PYTHON_VERSION=%%i %%j
echo Version:    %PYTHON_VERSION%
:: echo PYTHONPATH: %PYTHONPATH%
:: echo @ENV_LIB_PATH@:       %@ENV_LIB_PATH@%
echo.
echo.

%PYTHON_BINARY% @CMAKE_CURRENT_SOURCE_DIR@/@CREATE_FROM_PYTHON_PRESET_MAIN@ %*