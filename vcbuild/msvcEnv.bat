@echo off

:: Environment already set up?
if not "%VSINSTALLDIR%"=="" goto :eof

if "%LDC_VSDIR%"=="" goto detect

:: Check if the existing LDC_VSDIR environment variable points to a VS/VC installation folder
if not "%LDC_VSDIR:~-1%"=="\" set LDC_VSDIR=%LDC_VSDIR%\
if not "%LDC_VSDIR:~-4%"=="\VC\" set LDC_VSDIR=%LDC_VSDIR%VC\
if exist "%LDC_VSDIR%vcvarsall.bat" goto setup

:: Try to detect the latest VC installation directory
:detect
set LDC_VSDIR=
for /F "tokens=1,2*" %%i in ('reg query HKLM\SOFTWARE\Microsoft\VisualStudio\12.0\Setup\VC /v ProductDir 2^> nul') do set LDC_VSDIR=%%k
for /F "tokens=1,2*" %%i in ('reg query HKLM\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\12.0\Setup\VC /v ProductDir 2^> nul') do set LDC_VSDIR=%%k
for /F "tokens=1,2*" %%i in ('reg query HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\Setup\VC /v ProductDir 2^> nul') do set LDC_VSDIR=%%k
for /F "tokens=1,2*" %%i in ('reg query HKLM\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\Setup\VC /v ProductDir 2^> nul') do set LDC_VSDIR=%%k
if "%LDC_VSDIR%"=="" (
    echo WARNING: no Visual C++ installation detected
    goto :eof
)

:: Let MSVC set up environment variables
:setup
echo Using Visual C++: %LDC_VSDIR:~0,-1%
if not exist "%LDC_VSDIR%vcvarsall.bat" (
    echo WARNING: could not find vcvarsall.bat
    goto :eof
)
:: Forward the first arg to the MS batch file
call "%LDC_VSDIR%vcvarsall.bat" %1
