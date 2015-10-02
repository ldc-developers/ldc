@echo off
setlocal EnableDelayedExpansion
call "%~dp0msvcEnv.bat" x86
:: Invoke the actual command, represented by all args
%*
endlocal
