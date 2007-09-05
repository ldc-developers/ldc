LLVM D Compiler (LLVMDC)

This compiler is based on the Digital Mars D (DMD)[1] compiler frontend, and
the LLVM[2] compiler toolkit. It is licensed under the same licence as the DMD
compiler frontend. See dmd/readme.txt for more details.

premake[3] is used to generate a makefile so the project can be built. So far
only Linux is tested so use the command: 'premake --target gnu' to generate a
Makefile, then just type 'make'.

You need LLVM 2.1 which is not yet released, so LLVM from SVN is required.
Current development has been done against the 20070814 revision, newer will
probably work, later probably wont...

Many thing are still not implemented. For more information visit the website:
http://www.dsource.org/projects/llvmdc

[1] http://www.digitalmars.com/d
[2] http://www.llvm.org
[3] http://premake.sourceforge.net
