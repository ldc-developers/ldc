LDC – the LLVM-based D Compiler
===============================

[![Build Status](https://travis-ci.org/ldc-developers/ldc.png?branch=master)][1]
[![Test Coverage](https://coveralls.io/repos/ldc-developers/ldc/badge.svg)][2]
[![Bountysource](https://www.bountysource.com/badge/tracker?tracker_id=283332)][3]

The LDC project aims to provide a portable D programming language
compiler with modern optimization and code generation capabilities.

The compiler uses the official DMD frontends to support the latest
version of D2, and relies on the LLVM Core libraries for code
generation.

LDC is fully Open Source; the parts of the code not taken/adapted from
other projects are BSD-licensed (see the LICENSE file for details).

Please consult the D wiki for further information:
http://wiki.dlang.org/LDC

D1 is no longer available; see the 'd1' Git branch for the last
version supporting it.


Installation
------------

### From a pre-built package

Some Linux and OS X package managers include recent versions of LDC, so
manually installing it might not be necessary. For several platforms,
there are also stand-alone binary builds available at the
[GitHub release page](https://github.com/ldc-developers/ldc/releases).

| Distribution | Command               |
| ------------ | --------------------- |
| Arch Linux   | `pacman -S ldc`       |
| Debian       | `apt-get install ldc` |
| Fedora       | `yum install ldc`     |
| Gentoo       | `layman -a ldc`       |
| HomeBrew     | `brew install ldc`    |
| Ubuntu       | `apt-get install ldc` |

### Building from source

In-depth material on building and installing LDC and the standard
libraries, including experimental instructions for running LDC on
Windows, is available on the project wiki at
http://wiki.dlang.org/Building_LDC_from_source.

If you have a working C++ build environment, CMake, a current LLVM (>= 3.5),
and [libconfig](http://hyperrealm.com/libconfig/libconfig.html) available
there should be no big surprises.
Building LDC also requires a working D compiler, DMD and LDC are supported.
(LDC 0.17 is the last version that does not need a D compiler,
and for that reason we try to maintain it in the 'ltsmaster' branch).

Do not forget to make sure all the submodules (druntime, phobos, dmd-testsuite)
are up to date:

    $ cd ldc
    $ git submodule update --recursive --init

Contact
-------

The best way to get in touch with the developers is either via the
digitalmars.D.ldc forum/newsgroup/mailing list
(http://forum.dlang.org) or our [Gitter chat](http://gitter.im/ldc-developers/main).
There is also the #ldc IRC channel on FreeNode.

For further documentation, contributor information, etc. please see
the D wiki: http://wiki.dlang.org/LDC

Feedback of any kind is very much appreciated!


[1]: https://travis-ci.org/ldc-developers/ldc "Build Status"
[2]: https://coveralls.io/r/ldc-developers/ldc "Test Coverage"
[3]: https://www.bountysource.com/trackers/283332-ldc?utm_source=283332&utm_medium=shield&utm_campaign=TRACKER_BADGE "Bountysource"
