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
* Or set the LDC_VSDIR environment variable to some Visual Studio/Visual C++
  Build Tools installation directory, e.g.,
  'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community'.
  LDC will invoke a batch file provided by VS to set up the environment
  variables for the selected 32/64-bit target platform, which adds an overhead
  of about 1 second for each linking operation.
  You can also set LDC_VSDIR_FORCE (to some non-empty value); LDC will then try
  to auto-detect your latest Visual C++ installation if you haven't set
  LDC_VSDIR, and won't skip the environment setup if VSINSTALLDIR is pre-set.
* Or set up the etc\ldc2.conf config file and specify the directories containing
  the MS libs (appending them to the 'lib-dirs' array; check out the LIB
  environment variable in a VS tools command prompt) as well as the C runtime
  flavor (e.g., appending '-mscrtlib=libcmt' to the 'switches' array).
  In case you prefer the MS linker over LLD, add the switch
  '-linker=<path\to\link.exe>'.

For further information, including on how to report bugs, please refer to the
LDC wiki: http://wiki.dlang.org/LDC.
