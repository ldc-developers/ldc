This is a standalone (DMD-style) binary package for LDC,
the LLVM-based D compiler.

No installation is required, simply use the executables in
the bin\ subfolder.
Just make sure you have a Microsoft Visual C++ 2015 or 2017
installation, either via Visual Studio or via the stand-
alone Visual C++ Build Tools, both freely available from
Microsoft. LDC relies on the MS linker and MSVCRT + WinSDK
libs.

The compiler configuration file is etc\ldc2.conf, and by
default only import\ resp. import\ldc\ are on the module
search path.

The LDC package is portable and should be able to detect
your (latest) Visual C++ installation automatically.
By setting the LDC_VSDIR environment variable to an
existing Visual Studio directory, you can instruct LDC to
use a specific Visual C++ installation.
If run in a 'VS Native/Cross Tools Command Prompt' (i.e.,
if the environment variable VSINSTALLDIR is set), LDC skips
the Visual C++ detection. This saves about one second for
each linking operation, but linking will be restricted to
the selected target (=> no cross-linking support via
'-m32' in a x64 command prompt).

DUB

DUB 1.0+ works seamlessly together with LDC. If you have
different D compilers installed then you need to use the
--compiler=<path to ldc2.exe or ldmd2.exe> option with DUB.

For further information, including on how to report bugs,
please refer to the LDC wiki: http://wiki.dlang.org/LDC.
