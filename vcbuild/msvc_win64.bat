@ECHO OFF
REM Environment already set up?
IF NOT "%VSINSTALLDIR%"=="" GOTO :EOF

REM Clear an existing LDC_VSDIR environment variable if the directory doesn't exist
IF NOT "%LDC_VSDIR%"=="" IF NOT EXIST "%LDC_VSDIR%" SET LDC_VSDIR=

REM LDC_VSDIR not set? Then try to detect the latest VS installation directory
IF "%LDC_VSDIR%"=="" FOR /F "delims=" %%i IN ('dir "%ProgramFiles(x86)%\Microsoft Visual Studio 1*" /b /ad-h /on 2^> nul') DO SET LDC_VSDIR=%ProgramFiles(x86)%\%%i
IF "%LDC_VSDIR%"=="" (
    ECHO WARNING: No Visual Studio installation detected!
    ECHO.
    GOTO :EOF
)

REM Let MSVC set up environment variables
ECHO Using Visual Studio installation:
ECHO   %LDC_VSDIR%
ECHO.
CALL "%LDC_VSDIR%\VC\vcvarsall.bat" amd64
