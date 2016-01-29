@echo off
setlocal
call "%~dp0msvcEnv.bat" x86
:: Invoke the actual command, represented by all args
%*
endlocal && exit /b %ERRORLEVEL%
