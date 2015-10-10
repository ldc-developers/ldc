@echo off
setlocal EnableDelayedExpansion
call "%~dp0msvcEnv.bat" amd64
:: Invoke the actual command, represented by all args
%*
endlocal
