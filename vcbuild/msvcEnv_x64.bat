@ECHO OFF
SETLOCAL EnableDelayedExpansion

REM Environment already set up?
IF NOT "%VSINSTALLDIR%"=="" GOTO run

REM Clear an existing LDC_VSDIR environment variable if the directory doesn't exist
IF NOT "%LDC_VSDIR%"=="" IF NOT EXIST "%LDC_VSDIR%" SET LDC_VSDIR=

REM LDC_VSDIR not set? Then try to detect the latest VS installation directory
IF "%LDC_VSDIR%"=="" FOR /F "delims=" %%i IN ('dir "%ProgramFiles(x86)%\Microsoft Visual Studio 1*" /b /ad-h /on 2^> nul') DO SET LDC_VSDIR=%ProgramFiles(x86)%\%%i
IF "%LDC_VSDIR%"=="" (
    ECHO WARNING: No Visual Studio installation detected!
    GOTO run
)

REM Let MSVC set up environment variables
IF NOT EXIST "%LDC_VSDIR%\VC\vcvarsall.bat" (
    ECHO WARNING: Could not find !LDC_VSDIR!\VC\vcvarsall.bat
    GOTO run
)
ECHO Using Visual Studio: %LDC_VSDIR%
CALL "%LDC_VSDIR%\VC\vcvarsall.bat" amd64

:run
REM Invoke the inner command represented by all args
%*
ENDLOCAL
