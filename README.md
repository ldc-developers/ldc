LDC – the LLVM-based D Compiler
===============================

[![Build Status](https://circleci.com/gh/ldc-developers/ldc/tree/master.svg?style=svg)][1]
[![Build Status](https://semaphoreci.com/api/v1/ldc-developers/ldc/branches/master/shields_badge.svg)][2]
[![Build Status](https://travis-ci.org/ldc-developers/ldc.png?branch=master)][3]
[![Build Status](https://ci.appveyor.com/api/projects/status/2cfhvg79782n4nth/branch/master?svg=true)][4]
[![Bountysource](https://www.bountysource.com/badge/tracker?tracker_id=283332)][5]

The LDC project provides a portable D programming language compiler
with modern optimization and code generation capabilities.

The compiler uses the official DMD frontend to support the latest
version of D2, and relies on the LLVM Core libraries for code
generation.

LDC is fully Open Source; the parts of the source code not taken/adapted from
other projects are BSD-licensed (see the LICENSE file for details).

Please consult the D wiki for further information:
https://wiki.dlang.org/LDC

D1 is no longer available; see the `d1` Git branch for the last
version supporting it.


Installation
------------

### From a pre-built package

#### Linux and OS X

For several platforms, there are stand-alone binary builds available at the
[GitHub release page](https://github.com/ldc-developers/ldc/releases).

For bleeding-edge users, we also provide the [latest successful
Continuous Integration builds](https://github.com/ldc-developers/ldc/releases/tag/CI)
with enabled LLVM & LDC assertions (significantly increasing compile times).

The [dlang.org install script](https://dlang.org/install.html)
can also be used to install LDC:

    curl -fsS https://dlang.org/install.sh | bash -s ldc

In addition, LDC is available from various package managers:

|              | Command                                      |
| ------------ | -------------------------------------------- |
| Arch Linux   | `pacman -S ldc`                              |
| Debian       | `apt install ldc`                            |
| Fedora       | `dnf install ldc`                            |
| Gentoo       | `layman -a ldc`                              |
| Homebrew     | `brew install ldc`                           |
| Ubuntu       | `apt install ldc`                            |
| Snap         | `snap install --classic --channel=edge ldc2` |
| Nix/NixOS    | `nix-env -i ldc`                             |
| Chocolatey   | `choco ldc`                                  |
| Docker       | `docker pull dlanguage/ldc`                  |

Note that these packages **might be outdated** as they are not
currently integrated into the project release process.


#### Windows

The latest official releases can be downloaded from the
[GitHub release page](https://github.com/ldc-developers/ldc/releases).

For bleeding-edge users, we also provide the [latest successful
Continuous Integration builds](https://github.com/ldc-developers/ldc/releases/tag/CI)
with enabled LLVM & LDC assertions (significantly increasing compile times).

LDC for Windows relies on the Microsoft linker and runtime libraries,
which can be obtained by either installing
[Visual Studio](https://www.visualstudio.com/downloads/) 2015 or 2017
with Visual C++, or the stand-alone
[Visual C++ Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools).


#### Android

Native packages for Android are available in the
[Termux app](https://play.google.com/store/apps/details?id=com.termux&hl=en), easily
installed with `pkg install ldc`. You can find full instructions on cross-compiling
or natively compiling for Android
[on the wiki](https://wiki.dlang.org/Build_D_for_Android).

### Building from source

In-depth material on building and installing LDC and the standard
libraries is available on the project wiki for
[Linux, macOS, BSD, and Android](http://wiki.dlang.org/Building_LDC_from_source) and
[Windows](http://wiki.dlang.org/Building_and_hacking_LDC_on_Windows_using_MSVC).

If you have a working C++/D build environment, CMake, and a current LLVM
version (≥ 3.7) available, there should be no big surprises. Do not
forget to make sure all the submodules (druntime, phobos, dmd-testsuite)
are up to date:

    $ cd ldc
    $ git submodule update --init

(DMD and LDC are supported as host compilers. For bootstrapping
purposes, LDC 0.17, the last version not to require a D compiler, is
maintained in the `ltsmaster` branch).

### Cross-compiling

We've recently added a cross-compilation tool to make it easier to build the D
runtime and standard library for other platforms, `ldc-build-runtime`. Full
instructions and example invocations are provided
[on its wiki page](https://wiki.dlang.org/Building_LDC_runtime_libraries).

Contact
-------

The best way to get in touch with the developers is either via the
[digitalmars.D.ldc forum/newsgroup/mailing list](https://forum.dlang.org)
or our [Gitter chat](http://gitter.im/ldc-developers/main).
There is also the #ldc IRC channel on FreeNode.

For further documentation, contributor information, etc. please see
[the D wiki](https://wiki.dlang.org/LDC).

Feedback of any kind is very much appreciated!


[1]: https://circleci.com/gh/ldc-developers/ldc/tree/master "Circle CI Build Status"
[2]: https://semaphoreci.com/ldc-developers/ldc "Semaphore CI Build Status"
[3]: https://travis-ci.org/ldc-developers/ldc "Travis CI Build Status"
[4]: https://ci.appveyor.com/project/kinke/ldc/history "AppVeyor CI Build Status"
[5]: https://www.bountysource.com/teams/ldc-developers/issues "Bountysource"
