@echo off

:: Environment already set up?
if not "%VSINSTALLDIR%"=="" goto :eof

:: Clear an existing LDC_VSDIR environment variable if the directory doesn't exist
if not "%LDC_VSDIR%"=="" if not exist "%LDC_VSDIR%" set LDC_VSDIR=

:: Try to detect the latest VS installation directory if LDC_VSDIR is not set
if not "%LDC_VSDIR%"=="" goto setup
for /F "tokens=1,2*" %%i in ('reg query HKCU\Software\Microsoft\VisualStudio\12.0_Config /v ShellFolder 2^> nul') do set LDC_VSDIR=%%k
for /F "tokens=1,2*" %%i in ('reg query HKCU\Software\Microsoft\VisualStudio\14.0_Config /v ShellFolder 2^> nul') do set LDC_VSDIR=%%k
if "%LDC_VSDIR%"=="" (
    echo WARNING: no Visual Studio installation detected
    goto :eof
)

:: Let MSVC set up environment variables
:setup
echo Using Visual Studio: %LDC_VSDIR%
if not exist "%LDC_VSDIR%VC\vcvarsall.bat" (
    echo WARNING: could not find VC\vcvarsall.bat
    goto :eof
)
:: Forward the first arg to the MS batch file
call "%LDC_VSDIR%VC\vcvarsall.bat" %1
