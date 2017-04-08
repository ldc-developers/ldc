@echo off
setlocal
:: The single arg specifies the architecture (x86/amd64).
call "%~dp0msvcEnv.bat" %1 > nul
:: Dump all environment variables to stdout.
set
endlocal
