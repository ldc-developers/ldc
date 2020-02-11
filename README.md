LDC – the LLVM-based D Compiler
===============================

[![Latest stable release](https://img.shields.io/github/release/ldc-developers/ldc.svg?label=stable)][0]
[![Build status](https://img.shields.io/azure-devops/build/ldc-developers/ldc/Azure%20Pipeline/master.svg?label=Azure%20Pipelines&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAACEFBMVEUAAAA6XqwmPnQdMV1FddwcOHgaRKEiV8kaTsB5ksgXTL8USr5tkdwQRro2WKFIdNVMe%2BEAAABDbsxKe%2BM7Y7xGdt4vXL4yYccyYccxYMYzY8lAcdovXsU%2BcuM7b%2BE4aNIjUrgkVsI3a9wzYskoTZwaQJIfT7UpXMk1a98xZtYsWbkAAAAXPZAcPopRfNsgVMSSr%2B10irtEWYltj9tOetsvZt4aTsCRruxxiLhab5wYPIofVMYXTL%2BQru12j8Nuhrd0iroJL4ITRbITSb4USr6RsPGOre2OrOyOrOwNNY0QQ7MQRLVOfuVNfeZMfeZNfeZOfuZLfOZVg%2BdjjulVg%2BdLfOVLfOZIeeVSgeeMrPChvPSLq%2FBSgeZIeeVFd%2BREduRdieifuvSivPSeufNciOhDduQvX8YvX8YvX8Y3adNBdeNCdeRBdORMfeaHqPCduPOGqO9LfOZBdeQrXMUrXMUrXMUwYcs9cOA%2Fc%2BQ%2Fc%2BRCduU%2Fc%2BRJeuVWhOhIeuU%2Fc%2BQoWcQoWcQpWsY2adk8ceM8ceNLfelVhe0%2Fc%2BQ7cOM7cOM7cOMtYdE4buM5buNIe%2BlYiO9Heug5b%2BM5buM5buNFeehVhu9EeOg1bOI1bOI1a%2BFUhe9BdugzauIyauIwZt4mW88vZ%2BEpX9cfU8caTsAgVs0YTcEXS78XTL8USb4USb4USr4QR70QR70QR7z%2F%2F%2F%2FAAfjuAAAATHRSTlMAAAAAAAAAAAAAAAAAABOy8gBx%2BjfhdIB%2Ff3%2FD%2Ff34rsD932wQI7LY7sEyARspq4B3HAuTqe5%2F7joJJtd%2F8mo8HhWs%2Fn%2F%2B8%2FN4IL9yt9ocBQAAAH9JREFUCB0FwSE2BkAUBtDvzrzzZsp%2FNFkhWIBGoNmARWArBEn516JbAEezDM29kiQOruFDkowTbkAlmaZbwCm4A1TfA0A9AMCxDgB4G848ArzgHM%2FglZ%2F6vaBBf%2F%2BlLvHuCUsy1vrq7qqqHsnovlprHeecWzL23vtzrTnnGsk%2F9ugOvKlHC3gAAAAASUVORK5CYII%3D)][1]
[![Build status](https://img.shields.io/circleci/project/github/ldc-developers/ldc/master.svg?logo=circleci&label=CircleCI)][2]
[![Build status](https://img.shields.io/cirrus/github/ldc-developers/ldc/master?label=Cirrus%20CI&logo=Cirrus%20CI)][3]
[![Build status](https://img.shields.io/shippable/5b6826467dafeb0700d693e5/master.svg?label=Shippable&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAAB71BMVEUAAAAGBgY3NTVeW1uCgYGMi4tnZWVycXH%2Fu0TIZQBaV1dDQUKVk5Nsa2wNCgqGhYVvbG0BAQECAgICAQBNTVlsbXckIyIAAAAdGxt4d3eKiYpwbWVwbGSIhoeHhoY%2FPDxRT0%2Bko6PIyMjFxcWvrq5aWFhQTU2zsrLt7e22tbVeXV0AAACSk5To6Onl5uaLjY4AAABeMiJ5UkN%2BRCtrLxWTRyuZOhGEQiqENhVVMSaAPyd%2FSjeEQSSFNBJULBsBAABzbWybmpuMi4xcVVUJAABNSkqUk5N7e3wxLi9jYWGbmprBwsOjpKWKiotSUFFFQ0OCgYKdnZ6Li4xrams5NzgAAAA8OTo%2BPDwAAABycnRrXzluYTZkY2XS0tNYVEl5Zyx7aCxVT0LFxMXw8PDu7u%2FDwsJtZUx%2Bayl%2BaypuZEjBwL%2Fv8PGrmI%2BOhm5sYDxbUS9dUS5dUS5bUDBtYT2TinOqjYCoTCmJRChLPSpGQz5PTk9AP0JAP0JPTk9GQz5LPCiUOxO3QA2pTiuNRChlVlKOjo%2Bqqqt2dnd5eHqoqKmOj5BrV0%2BSORKzQA6IgoCcm5uioaGbm5ycnJyioaGamZmKgX7AwMCsq6u9vr68vL2vrq2%2Bvb20tLS3t7exsbKxsrK2tratra2pqaqbm5z%2F%2F%2F%2FE7yMkAAAAWXRSTlMAAAAAAAAAAAAAAAAAAAAAAAAAAwoMBQAJRZK8w6VbEB2Z8PirIxiw%2F7kdAHv%2B%2F3oAgt3ejvn2vrQ0sPDvrC8Kbfn3ZgkTtKwPKrn8%2B7MlF3G7uWwVAQsKAQ1Y5csAAACJSURBVAjXTckxS0IBAEbR74hIiAg5NDm4OznIwyJQaGsU9R8q4uwSoYNYvyJoaGkRIZoiaHiJ3u1wJYkWfCaRVG6UffxGqm0KvPL%2Bo9ahX94Xb7rlgx16bp3a0B85t1Y8XnBlwPgfC1wVTE7aS%2F3OFMztviWNe2aWbL8iaRqC52MiSa4f8HRI8ge%2FnBov5mGOrQAAAABJRU5ErkJggg%3D%3D)][4]
[![Build status](https://img.shields.io/travis/ldc-developers/ldc/master.svg?logo=travis&label=Travis%20CI)][5]
[![Bountysource](https://www.bountysource.com/badge/tracker?tracker_id=283332)][6]

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

Portable stand-alone binary builds for common platforms (incl. Linux,
macOS and Windows) are available at the
[GitHub release page](https://github.com/ldc-developers/ldc/releases).

For bleeding-edge users, we also provide the [latest successful
Continuous Integration builds](https://github.com/ldc-developers/ldc/releases/tag/CI)
with enabled LLVM & LDC assertions (increasing compile times by roughly 50%).

The [dlang.org install script](https://dlang.org/install.html)
can also be used to install LDC:

    curl -fsS https://dlang.org/install.sh | bash -s ldc

In addition, LDC is available from various package managers
(but note that these packages **might be outdated** as they are
not currently integrated into the project release process):

|              | Command                                      |
| ------------ | -------------------------------------------- |
| **Android**  | in [Termux app](https://play.google.com/store/apps/details?id=com.termux&hl=en): `pkg install ldc` |
| Arch Linux   | `pacman -S ldc`                              |
| Debian       | `apt install ldc`                            |
| Fedora       | `dnf install ldc`                            |
| Gentoo       | `layman -a ldc`                              |
| Homebrew     | `brew install ldc`                           |
| Ubuntu       | `apt install ldc`                            |
| Snap         | `snap install --classic --channel=edge ldc2` |
| Nix/NixOS    | `nix-env -i ldc`                             |
| Chocolatey   | `choco install ldc`                          |
| Docker       | `docker pull dlang2/ldc-ubuntu`              |

#### Targeting Android

You can find full instructions on cross-compiling or natively compiling
for Android [on the wiki](https://wiki.dlang.org/Build_D_for_Android).

### Building from source

In-depth material on building and installing LDC and the standard
libraries is available on the project wiki for
[Linux, macOS, BSD, and Android](http://wiki.dlang.org/Building_LDC_from_source) and
[Windows](http://wiki.dlang.org/Building_and_hacking_LDC_on_Windows_using_MSVC).

If you have a working C++/D build environment, CMake, and a recent LLVM
version (≥ 3.9) available, there should be no big surprises. Do not
forget to make sure all the submodules (druntime, phobos, dmd-testsuite)
are up to date:

    $ cd ldc
    $ git submodule update --init

(DMD, GDC and LDC are supported as host compilers. For bootstrapping
purposes, we recommend GDC via its `gdmd` wrapper. Alternatively, LDC 0.17,
the last version not to require a D compiler, is maintained to some degree in
the `ltsmaster` branch.)

### Cross-compiling

Similar to other LLVM-based compilers, cross-compiling with LDC is simple.
Full instructions and example invocations are provided on the dedicated
[Wiki page](https://wiki.dlang.org/Cross-compiling_with_LDC).

Contact
-------

The best way to get in touch with the developers is either via the
[digitalmars.D.ldc forum/newsgroup/mailing list](https://forum.dlang.org)
or our [Gitter chat](http://gitter.im/ldc-developers/main).
There is also the #ldc IRC channel on FreeNode.

For further documentation, contributor information, etc. please see
[the D wiki](https://wiki.dlang.org/LDC).

Feedback of any kind is very much appreciated!


[0]: https://github.com/ldc-developers/ldc/releases/latest
[1]: https://dev.azure.com/ldc-developers/ldc/_build/latest?definitionId=1&branchName=master
[2]: https://circleci.com/gh/ldc-developers/ldc/tree/master
[3]: https://cirrus-ci.com/github/ldc-developers/ldc/master
[4]: https://app.shippable.com/github/ldc-developers/ldc/runs?branchName=master
[5]: https://travis-ci.org/ldc-developers/ldc/branches
[6]: https://www.bountysource.com/teams/ldc-developers/issues
