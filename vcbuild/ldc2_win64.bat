@ECHO OFF
SETLOCAL
CALL "%~dp0msvc_win64.bat"
"%~dp0ldc2.exe" %*
ENDLOCAL
