This is a standalone (DMD-style) binary package for LDC, the LLVM-based D
compiler.

No installation is required, simply use the executables in the 'bin' subfolder.

The compiler configuration file is etc\ldc2.conf and can be easily customized
to your liking, e.g., adding implicit command-line options and setting up cross-
compilation.

The LDC package is portable and ships with LLD, the LLVM linker, as well as
WinSDK & Visual C++ runtime (import) libraries based on MinGW-w64. In order to
run the generated binaries, a Visual C++ 2015 runtime installation is required
(vcruntime140.dll, ucrtbase.dll etc.).

In case you prefer an official Microsoft toolchain for linking (Visual C++ 2015
or newer), e.g., to link with the static Microsoft libraries (and thus avoid the
dependency on the Visual C++ runtime installation for your users), you have the
following options:

* Run LDC in a 'VS Native/Cross Tools Command Prompt' (LDC checks whether the
  VSINSTALLDIR environment variable is set).
  LDC assumes the environment variables are all set up appropriately.
* Set the LDC_VSDIR environment variable to some Visual Studio/Visual C++ Build
  Tools installation directory, e.g.,
  'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community'.
  LDC will invoke a batch file provided by VS to set up the environment
  variables for the selected 32/64-bit target platform, which adds an overhead
  of about 1 second for each linking operation.
  You can also set LDC_VSDIR to some non-existing dummy path; LDC will try to
  auto-detect your latest Visual C++ installation in that case.
* Set up the etc\ldc2.conf config file and specify the path to the linker
  ('-linker=<path>', or use '-link-internally') as well as the directories
  containing the MS libs ('-L/LIBPATH:<path1> -L/LIBPATH:<path2> ...'; check out
  the LIB environment variable in a VS tools command prompt).

For further information, including on how to report bugs, please refer to the
LDC wiki: http://wiki.dlang.org/LDC.
