This is a standalone (DMD-style) binary package for LDC, the LLVM-based D
compiler.

No installation is required, simply use the executables in the 'bin' subfolder.

The compiler configuration file is etc\ldc2.conf and can be easily customized
to your liking, e.g., adding implicit command-line options and setting up cross-
compilation.

If you have an installed Visual C++ toolchain (Visual Studio/Build Tools 2015 or
newer), LDC defaults to using linker and libraries of the latest Visual C++
installation it can find.
You can set the LDC_VSDIR environment variable to select a specific version,
e.g., 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community'.
MSVC toolchain detection and setup is skipped if LDC is run inside a
'VS Native/Cross Tools Command Prompt' (more precisely, if the VSINSTALLDIR
environment variable is set).

If you don't have a Visual C++ installation, LDC falls back to LLD (the LLVM
linker) and the bundled WinSDK & Visual C++ runtime (import) libraries based on
MinGW-w64. In that case, the generated executables and DLLs depend on an
installed (redistributable) Visual C++ 2015+ runtime (vcruntime140.dll,
ucrtbase.dll etc.).

For further information, including on how to report bugs, please refer to the
LDC wiki: http://wiki.dlang.org/LDC.
