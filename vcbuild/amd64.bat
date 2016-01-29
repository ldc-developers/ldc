@echo off
setlocal
call "%~dp0msvcEnv.bat" amd64
:: Invoke the actual command, represented by all args
%*
endlocal && exit /b %ERRORLEVEL%
