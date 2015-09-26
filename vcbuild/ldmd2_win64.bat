@ECHO OFF
SETLOCAL
CALL "%~dp0msvc_win64.bat"
"%~dp0ldmd2.exe" %*
ENDLOCAL
