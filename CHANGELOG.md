# LDC master

#### Big news

#### Platform support

#### Bug fixes

# LDC 1.35.0 (2023-10-15)

#### Big news
- Frontend, druntime and Phobos are at version [2.105.2+](https://dlang.org/changelog/2.105.0.html). (#4476, #4498, #4513)
- The Windows installer now supports non-admin installs *without* an explicit `/CURRENTUSER` switch. (#4495)

#### Platform support
- Initial compiler support for LoongArch64. druntime support is pending. (#4500)

#### Bug fixes
- ImportC:
  - Fix `static` linkage. (#4484, #4487)
  - Make gcc builtins available. (#4483)
  - Apple: Support weird `asm("_" "<name>")` mangling stuff. (#4485, #4486)
- AArch64: Fix an ABI-related ICE. (#4489, #4490)
- Fix GC2Stack optimization regression introduced in v1.24. (#4510, #4511)
- Fix druntime ABI divergence when compiling with sanitizers support. (#4508, #4509)
- Windows: Fix an instance of missed backslash-escaping in `-ftime-trace` JSON. (#4506, #4507)

# LDC 1.34.0 (2023-08-26)

#### Big news
- Frontend, druntime and Phobos are at version [2.104.2](https://dlang.org/changelog/2.104.0.html). (#4440)
- Support for [LLVM 16](https://releases.llvm.org/16.0.0/docs/ReleaseNotes.html). The prebuilt packages use v16.0.6. (#4411, #4423)
  - We have come across miscompiles with LLVM 16's newly-enabled-by-default function specializations (on Win64 and macOS). To be on the safe side, LDC disables them by default for all targets via `-func-specialization-size-threshold=1000000000` in `etc/ldc2.conf` (and separately for LTO on Posix platforms). To enable the function specializations, explicitly override it with e.g. `-func-specialization-size-threshold=100` (the LLVM 16 default) and, for LTO on Posix, a similar LTO plugin option in the linker cmdline (see linker cmdline with `-v`).

#### Platform support
- Supports LLVM 11.0 - 16.0. Support for LLVM 9 and 10 was dropped.
- 64-bit RISC-V: Now defaults to `-mattr=+m,+a,+f,+d,+c` ('rv64gc' ABI) for non-bare-metal targets, i.e., if the target triple includes a valid operating system. (#4390)

#### Bug fixes
- Fix function pointers/delegates on Harvard architectures (e.g., AVR). (#4432, #4465)

# LDC 1.33.0 (2023-07-23)

#### Big news
- Frontend, druntime and Phobos are at version [2.103.1](https://dlang.org/changelog/2.103.0.html), incl. new command-line option `-verror-supplements`. (#4345)
- The `--plugin` commandline option now also accepts semantic analysis plugins. Semantic analysis plugins are recognized by exporting the symbol: `extern(C) void runSemanticAnalysis(Module m)`. The plugin's `runSemanticAnalysis` function is called for each module, after all other semantic analysis steps (also after DCompute SemA), just before object codegen. (#4430)
- New tool `ldc-build-plugin` that helps compiling user plugins. It downloads the correct LDC source version (if it's not already available), and calls LDC with the correct commandline flags to build a plugin. (#4430)
- New commandline option `-femit-local-var-lifetime` that enables variable lifetime (scope) annotation to LLVM IR codegen. Lifetime annotation enables stack memory reuse for local variables with non-overlapping scope. (#4395)
- C files are now automatically preprocessed using the external C compiler (configurable via `-gcc` or the `CC` environment variable, and `-Xcc` for extra flags). Extra preprocessor flags (e.g., include dirs and manual defines) can be added via new command-line option `-P`. (#4417)
  - Windows: If `clang-cl.exe` is on `PATH`, it is preferred over Microsoft's `cl.exe` by default (e.g., to avoid printing the C source file name to stderr during preprocessing).
- Less pedantic checks for conflicting C(++) function declarations when compiling multiple modules to a single object file ('Error: Function type does not match previously declared function with the same mangled name'). The error now only appears if an object file actually references multiple conflicting functions. (#4420)
- New command-line option `--fcf-protection`, which enables Intel's Control-Flow Enforcement Technology (CET). (#4437)

#### Platform support
- Supports LLVM 9.0 - 15.0.

#### Bug fixes
- Handle potential lambda mangle collisions across separately compiled object files (and the linker then silently picking an arbitrary implementation). Lambdas (and their nested global variables) are now internal to each referencing object file (`static` linkage in C). (#4415)

# LDC 1.32.2 (2023-05-12)

#### Big news
- New command-line option `--fwarn-stack-size=<threshold>` with LLVM 13+. (#4378)
- New command-line option `--fsplit-stack` for incremental stack allocations, see https://llvm.org/docs/SegmentedStacks.html. (#4379)
  - New UDA `ldc.attributes.noSplitStack` disables it on a per-function basis. (#4382)
- New command-line option `--indent` for the `timetrace2txt` tool. (#4391)

#### Bug fixes
- Fix potentially huge compile slowdowns with `-g` and LLVM 15+. (#4354, #4393)
- Treat *all* LLVM warnings as regular warnings (e.g., errors with `-w`). Requires LLVM 13+. (#4384)

# LDC 1.32.1 (2023-04-17)

#### Big news
- The prebuilt Linux packages are now generated on a Ubuntu 20.04 box, so the min required `glibc` version has been raised from 2.26 to 2.31. (#4367)

#### Bug fixes
- Fix empty `ldc.gccbuiltins_*` modules with LLVM 15+. (#4347, #4350)
- Fix v1.31 regression wrt. potentially wrong constant pointer offsets. (#4362, #4365)
- Windows: Fix v1.32 regression wrt. leaking `Throwable.info` backtraces. (#4369)
- Fix C assert calls for newlib targets. (#4351)

# LDC 1.32.0 (2023-03-12)

#### Big news
- Frontend, druntime and Phobos are at version [2.102.2](https://dlang.org/changelog/2.102.0.html). (#4323, #4341)
- LLVM for prebuilt packages bumped to v15.0.7. (#4311)
- Linker-level dead code elimination is enabled by default for Apple, wasm and *all* ELF targets too now. (#4320)
- Vector comparisons (==, !=, <, <=, >, >=) now yield a vector mask. Identity comparisons (`is`, `!is`) still yield a scalar `bool`. (3a59ee81)
- New `timetrace2txt` tool for easier inspection of `--ftime-trace` output. (#4335)
- `--ftime-trace` now also traces CTFE execution: the start expression of CTFE and function calls during CTFE. (#4339)

#### Platform support
- Supports LLVM 9.0 - 15.0.
- Now supports `-mabi` for RISC-V targets. (#4322)

#### Bug fixes
- GC closures including variables with alignment > 16 bytes are now properly aligned. (ef8ba481)
- Fix regression with LLVM 13+: some errors in inline assembly don't stop compilation. (#4293, #4331)

# LDC 1.31.0 (2022-02-11)

#### Big news
- Frontend, druntime and Phobos are at version [2.101.2](https://dlang.org/changelog/2.101.0.html). (#4141, #4279)
- Bit fields support. (#4015)
- macOS on Apple M1: linking with `-g` is working again without unaligned pointer warnings/errors. This fixes file:line debug information in exception backtraces (requiring `atos`, a macOS development tool installed with Xcode), without the need to set MACOSX_DEPLOYMENT_TARGET=11 and using a modified LLVM. (#4291)
- *Preliminary* support for LLVM 15, incl. adding support for the 'new' pass manager (`-passmanager`) and opaque IR pointers (`-opaque-pointers`). (way too many PRs to list!)
- New command-line option `-fno-delete-null-pointer-checks`, mimicking the same option of GCC and Clang. (#4297)
- New UDA `ldc.attributes.callingConvention("...")`, which overrides the default calling convention. For expert use only! (#4299)
- New command-line option `-fno-discard-value-names` to keep value names in LLVM IR. (#4012)
- dcompute: Support for OpenCL image I/O. (#3835)

#### Platform support
- Initial ABI support for 64-bit RISC-V. (#4007)

#### Bug fixes
- dcompute: Fix v1.29 regression when trying to use intrinsics. (#4266, #4267)
- Fix 64-bit symbol offsets. (#4264, #4283)
- Add missing 32-bit LTO versions of druntime & Phobos to Linux multilib package. (#4234, #4235)
- Fix compiler crash. (#4130, #4135)

#### Internals
- The former druntime and dmd-testsuite git submodules are now part of the LDC repo directly, leaving Phobos as single remaining submodule. We are now using a subset of the DMD repo (which includes druntime since v2.101), rewritten via `git filter-repo` and exposed as `dmd-rewrite-*` branches/tags in the LDC repo, to merge newer frontend+druntime+tests from upstream DMD. The `tests/d2/dmd-testsuite` dir was moved to `tests/dmd`. (#4274, #4276)

# LDC 1.30.0 (2022-07-20)

#### Big news
- Frontend, druntime and Phobos are at version [2.100.1](https://dlang.org/changelog/2.100.0.html). (#3970, #4008, #4009)
- LLVM for prebuilt packages bumped to v14.0.3. (#3952, #3979)
  - All LLVM targets are enabled now (=> more targets for cross-compilation).
  - For the Mac package, the minimum supported macOS version has been raised to v10.12.
- The minimum D version for bootstrapping has been raised to v2.079 (for GDC: v9.4), in line with DMD. (#3956)
- The minimum LLVM version has been raised to v9.0. (#3960)
- New LeakSanitizer support via `-fsanitize=leak` (not (yet?) supported on Windows). (#4005)
- New prebuilt *universal* macOS package, runnable on both x86_64 and arm64, and enabling x86_64/arm64 macOS/iOS cross-compilation targets out of the box (`-mtriple={x86_64,arm64}-apple-{macos,ios12.0}`). The x86_64 package doesn't bundle any arm64 libs anymore; the arm64 package newly bundles iOS libs (arm64). (#3958)
  - Avoid an external x86_64-only dub, use the bundled universal dub executable instead.

#### Platform support
- Supports LLVM 9.0 - 14.0.

#### Bug fixes
- Enable output of variable names in ASan and MSan error reporting. (#4004)
- Report unexpected type repaints as fatal ICEs instead of crashing. (#3990, #3991)

#### Internals
- Main CI was moved from Azure Pipelines to GitHub Actions. Any fork on GitHub can trivially reuse the fully automated prebuilt packages generation & upload to a GitHub release. (#3978)

# LDC 1.29.0 (2022-04-08)

#### Big news
- Frontend, druntime and Phobos are at version [2.099.1](https://dlang.org/changelog/2.099.0.html). (#3917, #3893, #3937, #3953)
- Support for LLVM 13 and 14. The prebuilt packages use v13.0.1. (#3842, #3951)
- On Linux, LDC doesn't default to the `ld.gold` linker anymore. The combination of LLVM 13+ and older gold linkers can apparently cause problems. We recommend using LLD, e.g., via `-linker=lld` or by setting your default `/usr/bin/ld` symlink; it's significantly faster too.
- `-linkonce-templates` is less aggressive by default now and IMHO production-ready. (#3924)
- When linking manually (not via LDC) against *shared* druntime, it is now required to link the bundled `lib/ldc_rt.dso.o[bj]` object file into each binary. It replaces the previously Windows-specific `dso_windows.obj`. (#3850)
- Breaking `extern(D)` ABI change for all targets: formal parameters of non-variadic functions aren't reversed anymore, in line with the spec. For 32-bit x86, the *first* parameter is accordingly now potentially passed in EAX, not the last one. So non-variadic `extern(D)` functions with multiple explicit parameters will break if expecting parameters in specific registers or stack slots, e.g., naked DMD-style inline assembly. (#3873, ldc-developers/phobos@3d725fce8f0acb78bf6cb984a8462e81e8e1b715)

#### Platform support
- Supports LLVM 6.0 - 14.0.
- Basic compiler support for Newlib targets, i.e., triples like `arm-none-newlibeabi`. (#3946)

#### Bug fixes
- Linux: Make LTO work with LLD. (#3786, #3850)
- Windows: Fix most undefined symbols with `-dllimport=all` without `-linkonce-templates`. (#3916, #3923, #3926, #3927, #3928, #3931, #3932)
- Capture NRVO variable by ref for stack closures. (#3883, #3902)
- `-ftime-trace`: JSON-escape filenames. (#3947, #3948)
- RISC-V: Use 128-bit quadruple `real`. (#3892)

# LDC 1.28.1 (2022-01-13)

#### Big news
- Frontend, druntime and Phobos are at version [2.098.1+](https://dlang.org/changelog/2.098.0.html). (#3886, #3896)
- New `@hidden` UDA (as counterpart of `export`). (#3855)
- Support 'raw mangles' via leading `\1` in `pragma(mangle)` strings, e.g., to access magic linker symbols on Mac. (#3854)
- New `@noSanitize` UDA to selectively disable sanitizer instrumentation of functions. (#3889)
- WebAssembly: Larger default stack size (1 MB) and protection against stack overflow overwriting global memory. (#3882)

#### Bug fixes
- Linux x86/x64: Fix TLS range with static druntime and bfd/lld linkers. (#3849, https://github.com/ldc-developers/druntime/commit/ec3c0aafbf4b6f3345e276e21a26ffee077470cf)
- Support `rdtscp` in DMD-style inline assembly. (#3895)

# LDC 1.28.0 (2021-10-20)

#### Big news
- Frontend, druntime and Phobos are at version [2.098.0+](https://dlang.org/changelog/2.098.0.html). (#3821, #3839, #3844, #3852)
- Windows: `-dllimport=defaultLibsOnly` (e.g., implied by `-link-defaultlib-shared -fvisibility=hidden`) doesn't require `-linkonce-templates` anymore. (#3816)
- dcompute: Add support for OpenCL image I/O. (#3835)

#### Platform support
- Supports LLVM 6.0 - 12.0.

#### Bug fixes
- Fix dynamic casts across binary boundaries (DLLs etc.). (dlang/druntime#3543)
- Windows: Fix potentially wrongly caught exceptions due to non-unique `TypeInfo_Class` names. (#3520)
- Don't silently ignore invalid external tool specifications. (#3841)
- LLVM v11.1: Add missing PGO `ldc-profdata` tool.

# LDC 1.27.1 (2021-08-14)

#### Big news
- Frontend, druntime and Phobos are at version [2.097.2](https://dlang.org/changelog/2.097.0.html). (#3811)
- Revamped and improved `-ftime-trace` implementation for compiler profiling/tracing, now excluding LLVM-internal traces, adding frontend memory tracing, source file location infos etc. (#3797)
- An official prebuilt package for Linux AArch64 is available again after migrating from Shippable to Travis. (#3733)

#### Bug fixes
- ICE for 64-bit targets with 32-bit pointer size. (#3802, #3808)
- Implement `core.atomic.pause()` for some architectures. (#3806, #3807)

# LDC 1.27.0 (2021-07-31)

#### Big news
- Frontend, druntime and Phobos are at version [2.097.1+](https://dlang.org/changelog/2.097.0.html). (#3741, #3770, #3771, #3790, #3794, #3796, #3799) **(new)**
- LLVM for prebuilt packages bumped to **v12.0.1**, and Linux base image to Ubuntu 18.04. Unfortunately, the dynamic-compile (JIT) functionality is lost this way - it needs some [more work](https://github.com/ldc-developers/ldc/pull/3184) to adapt to a newer LLVM API. (#3701, #3789)
- Prebuilt packages now bundle [reggae](https://github.com/atilaneves/reggae), a meta build tool to generate [ninja](https://github.com/ninja-build/ninja/releases)/make build files for dub projects (and more). Building large projects with many dependencies can be significantly sped-up via parallelization and dependency tracking for incremental builds. (#3739)
  Basic [usage](https://github.com/atilaneves/reggae#d-projects-and-dub-integration), in a dub project dir (containing a `dub.{sdl,json}` file):
  ```
  reggae -b ninja|make --dc=ldc2   # only needed the first time or when adding source files
  ninja|make [-j<N>]
  ```
- Greatly improved **DLL support on Windows**, making it almost as easy as on Posix:
  - `-fvisibility=public` now also affects Windows, exporting all defined symbols as on Posix, without explicit `export` visibility. Compiling a DLL with `-shared` now defaults to `-fvisibility=public` for consistency with Posix. (#3703)
  - This paved the way for druntime and Phobos DLLs, now bundled with prebuilt Windows packages and linkable via `-link-defaultlib-shared` (default with `-shared`, consistent with Posix targets). Previous hacks to partially accomodate for multiple, statically linked druntimes and Phobos in a single process (GC proxy etc.) aren't required any longer. With `-link-defaultlib-shared`, LDC now defaults to `-mscrtlib=msvcrt`, linking against the shared MSVC runtime. (ldc-developers/druntime#197, #3704, ldc-developers/druntime#198)
  - Limitation: TLS variables cannot be accessed directly across DLL boundaries. This can be worked around with an accessor function, e.g., ldc-developers/druntime@5d3e21a35d.
  - Non-TLS `extern(D)` global variables *not* defined in a root module are `dllimport`ed (with `-fvisibility=public`, or - restricted to druntime/Phobos symbols - with `-link-defaultlib-shared`). Compiling all modules of a DLL at once thus avoids linker warnings about 'importing locally defined symbol'. When linking a DLL against a static library, the static library may likely need to be compiled with `-fvisibility=public` to make its globals importable from the DLL. There's a new `-dllimport` option for explicit control. (#3763)
  - Caveat: symbols aren't uniqued across the whole process, so can be defined in multiple DLLs/executables, each with their own address, so you cannot rely on TypeInfos, instantiated symbols and functions to have the same address for the whole process.
  - When linking manually (not via LDC), binaries linked against druntime DLL need to include new `lib\dso_windows.obj`.
  - To restore the previous behavior of `-shared`, add `-fvisibility=hidden -link-defaultlib-shared=false`.
- Windows: ANSI color codes can now be enforced for redirected stderr via `-enable-color`. (#3744)
- Prebuilt Linux and Mac packages now use the [mimalloc](https://github.com/microsoft/mimalloc) allocator, significantly increasing compiler performance in some cases. (#3758, #3759)
- The prebuilt macOS x64 package now bundles *shared* druntime/Phobos libs for iOS too. (#3764)
- Possibly more performant *shared* Phobos library by compiling to a single object file with implicit cross-module inlining. (#3757)
- New `-cov-increment` option for more performant coverage count execution. (#3724)
- `-fsanitize=memory`: Bundle according LLVM compiler-rt library and add new `-fsanitize-memory-track-origins` option. (#3751)
- New LDC-specific language addition: `__traits(initSymbol, <aggregate type>)` with semantics equivalent to `TypeInfo.initializer()`, but circumventing the `TypeInfo` indirection and thus e.g. also usable for `-betterC` code. (#3774, ldc-developers/druntime#201)

#### Platform support
- Supports LLVM 6.0 - 12.0.

#### Bug fixes
- Fix debuginfo source file paths, e.g., including directories in exception stack traces. (#3687)
- Fix potentially corrupt context pointers for nested functions with `-linkonce-templates`. (#3690, #3766)
- Predefine version `CppRuntime_Gcc` for musl targets. (#3769)
- RVO: In-place construct `<temporary>.__ctor(<args>)`. (#3778, #3779)
- `-linkonce-templates`: Make sure special struct TypeInfo members are semantically analyzed before emitting the `TypeInfo`. (#3783)

# LDC 1.26.0 (2021-04-28)

#### Big news
- Frontend, druntime and Phobos are at version [2.096.1+](https://dlang.org/changelog/2.096.0.html), incl. new `ldmd2` command-line option `-gdwarf=<version>` (use `-dwarf-version` for `ldc2`). (#3678, #3706)

#### Platform support
- Supports LLVM 6.0 - 12.0.

#### Bug fixes
- v1.25 regression: TypeInfo for interface gives invalid string for name. (#3693)
- Make enums show up correctly as members of a struct when debugging. (#3688, #3694)
- Some new GCC builtins are available in `ldc.gccbuiltins_*`, by not rejecting LLVM `i1` anymore (mapping to D `bool` instead). Thanks Bruce! (#3682)
- dcompute: Don't reject CUDA versions 7.x - 8.0.0. (#3683)
- Don't enforce the frame pointer for functions with GCC-style inline asm. (#3685)
- `-i`: Exclude `ldc.*` modules by default. (#3679)
- Fix some cases of insufficient alignment for arguments and parameters. (#3692, #3698)
- Fix a few issues with LLVM 12. (#3697, #3708)

# LDC 1.25.0 (2021-02-21)

#### Big news
- Frontend, druntime and Phobos are at version [2.095.1](https://dlang.org/changelog/2.095.0.html), incl. new command-line option `-makedeps`. (#3620, #3658, #3668)
- Support for **LLVM 12** and LLVM 11.1. (#3663, ldc-developers/druntime#195)
- LLVM for prebuilt packages bumped to v11.0.1. (#3639)
- New prebuilt package for **native macOS/arm64** ('Apple silicon'). (#3666)
- LDC invocations can now be nicely profiled via `--ftime-trace`. (#3624)
- Struct TypeInfos are emitted into *referencing* object files only, and special TypeInfo member functions into the owning object file only. (#3491)
- Windows:
  - New CI-automated [Windows installer](https://github.com/ldc-developers/ldc/releases/download/v1.25.0/ldc2-1.25.0-windows-multilib.exe) corresponding to the multilib package. (#3601)
  - Bundled MinGW-based libs bumped to MinGW-w64 v8.0.0. (#3605)
  - Bundled libcurl upgraded to v7.74.0. (#3638)
  - Breaking ABI changes:
    - `extern(D)`: Pass non-PODs by ref to temporary. (#3612)
    - Win64: Pass/return delegates like slices - in (up to) 2 GP registers. (#3609)
    - Win64 `extern(D)`: Pass/return Homogeneous Vector Aggregates in SIMD registers. (#3610)
- `-linkonce-templates` comes with a new experimental template emission scheme and is now suited for projects consisting of multiple object files too. It's similar to C++, emitting templated symbols into *each* referencing compilation unit with optimizer-discardable `linkonce_odr` linkage. The consequences are manifold - each object file is self-sufficient wrt. templated symbols, naturally working around any template-culling bugs and also meaning increased opportunity for inlining and less need for LTO.
  The probably biggest advantage is that the optimizer can discard unused `linkonce_odr` symbols early instead of optimizing and forwarding to the assembler. So this is especially useful to **decrease compilation times with `-O`** and can at least in some scenarios greatly outweigh the (potentially very much) higher number of symbols defined by the glue layer - on my box, building optimized dub (all-at-once) is 28% faster with `-linkonce-templates`, and building the optimized Phobos unittests (per module) 56% faster.
  Libraries compiled with `-linkonce-templates` can generally *not* be linked against dependent code compiled without `-linkonce-templates`; the other way around works. (#3600)
- Emit function/delegate literals as `linkonce_odr`, as they are emitted into each referencing compilation unit too. (#3650)
- Exploit ABI specifics with `-preview=in`. (#3578)
- Musl: Switch to cherry-picked libunwind-based backtrace alternative. (#3641, ldc-developers/druntime#192)

#### Platform support
- Supports LLVM 6.0 - 12.0.

#### Bug fixes
- Fix LTO with `-link-internally`. The prebuilt Windows packages don't bundle an external `lld-link.exe` LLD linker anymore. (#2657, #3604)
- Add source location information for `TypeInfo` diagnostics with `-betterC`. (#3631, #3632)
- Keep init symbols of built-in `TypeInfo` classes mutable just like any other TypeInfo, so that e.g. `synchronized()` can be used on the implicit monitor. (#3599)
- Windows: Fix colliding EH TypeDescriptors for exceptions with the same `TypeInfo_Class` name. (#3501, #3614)
- Predefine version `FreeStanding` when targeting bare-metal. (#3607, #3608)
- druntime: Define `rt.aaA.AA` as naked pointer, no struct wrapper. (#3613)
- Misc. fixes and improvements for the CMake scripts, incl. new defaults for `LDC_INSTALL_{LTOPLUGIN,LLVM_RUNTIME_LIBS}`. (#3647, #3655, #3654, #3673)
- `-cleanup-obj`: Put object files into unique temporary directory by default. (#3643, #3660)
- druntime: Add missing `core.atomic.atomicFetch{Add,Sub}`. (#3646, ldc-developers/druntime#193)
- Fix regression wrt. non-deleted temporary `-run` executable. (#3636)

#### Internals
- Ignore `-enable-cross-module-inlining` if inlining is generally disabled. (#3664)
- Travis CI ported to GitHub Actions (excl. Linux/AArch64). (#3661, #3662)

# LDC 1.24.0 (2020-10-24)

#### Big news
- Frontend, druntime and Phobos are at version [2.094.1+](https://dlang.org/changelog/2.094.0.html), incl. new command-line options `-cov=ctfe`,  `-vtemplates=list-instances` and `-HC=<silent|verbose>` . (#3560, #3582, #3588, #3593)
- Support for **LLVM 11**. The prebuilt packages use v11.0.0; x86 packages newly include the LLVM backend for AMD GPUs. (#3546, #3586)
- Experimental support for **macOS on 64-bit ARM**, thanks Guillaume! All druntime/Phobos unit tests pass. The macOS package includes prebuilt druntime/Phobos; adapt the SDK path in `etc/ldc2.conf` and then use `-mtriple=arm64-apple-macos` to cross-compile. (dlang/druntime#3226, #3583)

#### Platform support
- Supports LLVM 6.0 - 11.0.

#### Bug fixes
- Fix potentially wrong context pointers when calling delegate literals. (#3553, #3554)
- Fix alignment issue when casting vector rvalue to static array. (c8889a9219)
- Make sure lambdas in `pragma(inline, true)` functions are emitted into each referencing compilation unit. (#3570)
- Fix `-Xcc=-Wl,...` by dropping support for comma-separated list of `cc` options. (c61b1357ed)
- Fix ThreadSanitizer support by not detaching main thread upon program termination. (#3572)
- Traverse full chain of nested aggregates when resolving a nested variable. (#3556, #3558)

#### Internals
- CI: Linux AArch64 is now also tested by a Travis job, because Shippable has sadly become unreliable. (#3469)
- Building LDC with an LDC host compiler might be somewhat faster now (requires `-DLDC_LINK_MANUALLY=OFF` in the CMake command-line on non-Windows hosts). (#3575)

# LDC 1.23.0 (2020-08-19)

#### Big news
- Frontend, druntime and Phobos are at version [2.093.1+](https://dlang.org/changelog/2.093.0.html), incl. new command-line option `-vtemplates`. (#3476, #3538, #3541)
- Min required LLVM version raised to v6.0, dropping support for v3.9-5.0. (#3493)
- LLVM for prebuilt packages bumped to v10.0.1. (#3513)
- The prebuilt Mac package now also includes prebuilt druntime/Phobos for the iOS/x86_64 simulator, making cross-compilation work out of the box with `-mtriple=x86_64-apple-ios12.0`. (#3478)
- Windows: New `-gdwarf` CLI option to emit DWARF debuginfos for MSVC targets, e.g., for debugging with gdb/lldb. (#3533)
- New `-platformlib` CLI option to override the default linked-with platform libraries, e.g., when targeting bare-metal. (#3374, #3475)

#### Platform support
- Supports LLVM 6.0 - 10.0.

#### Bug fixes
- Fix regression since v1.22: shared druntime potentially overriding libstdc++ symbols and breaking exceptions in C++ libraries. (#3530, #3537)
- Fix naked DMD-style asm emission for non-Mac x86 Darwin targets (e.g., iOS simulators). (#3478)
- `-betterC`: Don't use unsupported EH for handling clean-ups. (#3479, #3482)
- dcompute: Fix wrong address space loads and stores. Thx Rob! (#3428)
- Fix ICE wrt. missing IR declarations for some forward-declared functions. (#3496, #3503)
- Fix ICE wrt. inline IR and empty parameter types tuple. (#3509)
- Fix PGO issues. (#3375, #3511, #3512, #3524)
- Improve support for LLVM's ThreadSanitizer. (#3522)
- Fix linker cmdline length limitation via response files. (#3535, #3536)

#### Internals
- Compiler performance wrt. string literals emission has been improved. Thx @looked-at-me! (#3490, #3492)
- Link libstdc++ statically for `libldc-jit.so` of prebuilt Linux packages, to increase portability. (#3473, #3474)
- Set up Visual D when using the Visual Studio CMake generator, making LDC compiler development on Windows a smooth out-of-the-box experience. (#3494)

# LDC 1.22.0 (2020-06-16)

#### Big news
- Frontend, druntime and Phobos are at version [2.092.1+](https://dlang.org/changelog/2.092.0.html). (#3413, #3416, #3429, #3434, #3452, #3467)
- **AArch64**: All known ABI issues have been fixed. C(++) interop should now be on par with x86_64, and variadics usable with `core.{vararg,stdc.stdarg}`. (#3421)
- Windows hosts: DMD's Visual C++ toolchain detection has been adopted. As that's orders of magnitude faster than the previous method involving the MS batch file, auto-detection has been enabled by default, so if you have a non-ancient Visual C++ installation, it will now be used automatically for linking. The environment setup has been reduced to the bare minimum (`LIB` and `PATH`). (#3415)
- **FreeBSD** x64: CI with CirrusCI is now fully green and includes automated prebuilt package generation. The package depends on the `llvm` ports package and should currently work on FreeBSD 11-13. (#3453, #3464)
- Link-time overridable `@weak` functions are now emulated for Windows targets and work properly for ELF platforms. For ELF, LDC doesn't emit any COMDATs anymore. (#3424)
- New `ldc.gccbuiltins_{amdgcn,nvvm}` for AMD GCN and NVIDIA PTX targets. (#3411)
- druntime: Significant speed-up for `core.math.ldexp`. (#3440, #3446)

#### Platform support
- Supports LLVM 3.9 - 10.0.

#### Bug fixes
- Cross-module inlining (incl. `pragma(inline, true)`): Enable emission into multiple object files. This may have a significant impact on performance (incl. druntime/Phobos) when not using LTO. (#3126, #3442)
- Android: Fix TLS initialization regression (introduced in v1.21) and potential alignment issues. Unfortunately, the `ld.bfd` linker is required for our custom TLS emulation scheme, unless you're willing to use a custom linker script. So `-linker=bfd` is the new default for Android targets. (#3462)
- Casting (static and dynamic) arrays to vectors now loads the data instead of splatting the first element. (#3418, #3419)
- Fix return statements potentially accessing memory from destructed temporaries. (#3426)
- Add proper support for `-checkaction=halt`. (#3430, #3431)
- druntime: Include `core.stdcpp.*` modules. (#3103, #3158)
- GCC-style asm: Add support for indirect input operands (`"m"`). (#3438)
- FreeBSD: Fix backtraces for optimized code by switching to external `libexecinfo`. (#3108, #3453)
- FreeBSD: Fix C math related issues (incl. CTFE math issues) by bringing `core.stdc.{math,tgmath}` up to speed. (dlang/druntime#3119)
- Fix ICE for captured parameters not passed on the LLVM level. (#3441)
- Convenience fixes for RISC-V and other exotic targets. (#3457, #3460)

#### Internals
- When printing compile-time reals to hex strings (mangling, .di headers), LDC now uses LLVM instead of the host C runtime, for proper and consistent results. (#3410)
- One limitation for exotic hosts wrt. C `long double` precision has been lifted. (#3414)
- For AVR targets, the compiler now predefines `AVR` and emits all TLS globals as regular `__gshared` ones. (#3420)
- WebAssembly: New memory grow/size intrinsics. (ldc-developers/druntime#187)
- New `-fno-plt` option to avoid PLT external calls. (#3443)
- iOS/arm64 CI, running the debug druntime & Phobos unittests on an iPhone 6S. Thx Jacob for this tedious work! (#3379, #3450)

# LDC 1.21.0 (2020-04-23)

#### Big news
- Frontend, druntime and Phobos are at version [2.091.1+](https://dlang.org/changelog/2.091.1.html), incl. new CLI switches `-verror-style` and `-HC`, `-HCd`, `-HCf`. (#3333, #3399)
- **iOS** (incl. watchOS and tvOS) support has landed in druntime and Phobos (thanks Jacob!). All unittests are green on iOS/arm64. The prebuilt macOS package includes prebuilt druntime & Phobos libraries for iOS/arm64, for first `-mtriple=arm64-apple-ios12.0` cross-compilation experiments. (#3373)
- LLVM for prebuilt packages upgraded to v10.0.0. Android NDK version bumped to r21. (#3307, #3387, #3398)
- Initial support for **GCC/GDC-style inline assembly** syntax, besides DMD-style inline asm and LDC-specific `__asm`, enabling to write inline asm that is portable across GDC/LDC and corresponds to the GCC syntax in C. See ldc-developers/druntime#171 for examples wrt. how to transition from `__asm` to similar GCC-style asm.  (#3304)
- Inline assembly diagnostics have been extended by the D source location. (#3339)
- **Android**:
  - Revamped druntime initialization, fixing related issues for i686/x86_64 targets, enabling the usage of the `ld.gold` linker (bfd isn't required anymore) as well as getting rid of the D `main()` requirement. (#3350, #3357, ldc-developers/druntime#178)
  - Reduced size for shared libraries by compiling druntime and Phobos with hidden visibility. (#3377)

#### Platform support
- Supports LLVM 3.9 - 10.0.

#### Bug fixes
- Fixed tail calls in thunks, affecting **AArch64** (the debug libraries now work) and possibly other architectures. (#3329, #3332)
- Windows: Do not emit any column infos for CodeView by default (like clang) & add `-gcolumn-info`. (#3102, #3388)
- Windows: Do not leak MSVC-environment-setup into `-run` child processes. A new `LDC_VSDIR_FORCE` environment variable can be used to enforce MSVC toolchain setup. (#3340, #3341)
- Windows: Fix memory leak when throwing exceptions in threads. (#3369, ldc-developers/druntime#181)
- Try to use `memcmp` for (in)equality of non-mutable static arrays and mutable slices. (#3400, #3401)
- `ldc.gccbuiltins_*`: Lift 256-bit vector limit, adding 174 AVX512 builtins for x86; 512-bit vector aliases have been added to `core.simd`. (#3405, #3406)

#### Internals
- `core.bitop.{bts,btr,btc}` are now CTFE-able. (ldc-developers/druntime#182)
- Do not fallback to host for critical section size of unknown targets. (#3389)
- Linux: Possibility to avoid passing `-fuse-ld` to `cc` via `-linker=`. (#3382)
- WebAssembly: Switch from legacy linked-list ModuleInfo registry to `__minfo` section. (#3348)
- Windows: Bundled libcurl upgraded to v7.69.1, incl. the option to link it statically. (#3378)
- Windows: Switch to wide `wmain` C entry point in druntime. (#3351)
- druntime unittests are now compiled with `-checkaction=context`.

#### Known issues
- When building LDC, old LDC 0.17.*/ltsmaster host compilers miscompile LDC ≥ 1.21, leading to potential segfaults of the built LDC. Ltsmaster can still be used to bootstrap a first compiler and then let that compiler compile itself. (#3354)

# LDC 1.20.1 (2020-03-07)

#### Bug fixes
- Non-Windows: Revert to strong `ModuleInfo.importedModules` references for correct module constructors execution order. (#3346, #3347)

# LDC 1.20.0 (2020-02-14)

#### Big news
- Frontend, druntime and Phobos are at version [2.090.1+](https://dlang.org/changelog/2.090.1.html). (#3262, #3296, #3306, #3317, #3326)
- Codegen preparations for:
  - iOS/tvOS/watchOS on AArch64. Thanks Jacob! (#3288)
  - WASI (WebAssembly System Interface) (#3295)
- The config file for multilib builds has been restructured by adding a separate section for the multilib target. This avoids `--no-warn-search-mismatch` for the linker and enables support for LLD. (#3276)
- Support for embedding `pragma({lib,linkerDirective}, ...)` in Mach-O object files. (#3259)
  E.g., `pragma(linkerDirective, "-framework", "CoreFoundation");` makes Apple's linker pull in that framework when pulling in the compiled object file.
  ELF object files newly embed `pragma(lib, ...)` library names in a special `.deplibs` section, but that only works with LLD 9+ for now.
- The `ldc-build-runtime` tool has been slightly revised; `--dFlags` now extends the base D flags instead of overriding them. (1200601d44280d5f948a577b444ffa2dd4f9e433)
- `ModuleInfo.importedModules` are now emitted as weak references (except on Windows, for LLD compatibility), following DMD. (#3262)
- Windows: Bundled MinGW-based libs now support wide `wmain` and `wWinMain` C entry points. (#3311)

#### Platform support
- Supports LLVM 3.9 - 10.0.

#### Bug fixes
- Potential stack overflows on Linux in GC worker threads. (#3127, dlang/druntime#2904)
- Support 2 leading dashes (not just 1) in command-line pre-parsing, thus fixing config file section lookup when using `--mtriple` and not ignoring `--conf` and `--lowmem` any longer. (#3268, #3275)
- Support for data directives in DMD-style inline asm. (#3299, #3301)
- Cherry-picked fixes for soft-float targets. (#3292, dlang/phobos#7362, dlang/phobos#7366, dlang/phobos#7377)
- ICE during debuginfo generation for function literals inside enum declarations. (#3272, #3274)

#### Internals
- Misc. tweaks for `dmd-testsuite`: (#3287, #3306)
  - Significantly accelerated by skipping uninteresting permutations.
  - Switch from Makefile to `run.d`, incl. moving LDC-specific exceptions from Makefile to individual test files and support for extended `DISABLED` directives.
- Addition of (recommendable!) Cirrus CI service (incl. FreeBSD) and removal of Semaphore CI. (#3298)
- Some improvements for `gdmd` host compilers, incl. CI tests. (#3286)

# LDC 1.19.0 (2019-12-20)

#### Big news
- Frontend, druntime and Phobos are at version [2.089.1+](https://dlang.org/changelog/2.089.1.html). (#3192, #3210, #3215, #3232, #3242, #3255, #3261)
- LLVM for prebuilt packages upgraded to v9.0.1; our fork has moved to [ldc-developers/llvm-project](https://github.com/ldc-developers/llvm-project). The x86[_64] packages newly include the experimental **AVR** backend. (#3244)
- **Android**: A prebuilt AArch64 package has been added. It also includes prebuilt druntime/Phobos libraries for x86_64; the armv7a package includes the i686 libraries. So all 4 Android targets are covered with prebuilt druntime/Phobos. (#3244)
- Breaking `extern(D)` ABI change for Posix x86[_64]: non-POD arguments are now passed by ref under the hood, just like they already were for `extern(C++)`. Some superfluous implicit blits have been optimized away as well, for all targets. (#3204)
- Posix: Defaults to `cc` now for linking, not `gcc` (or `clang` for FreeBSD 10+) - if the `CC` environment variable isn't set. Override with `-gcc=<gcc|clang>`. (#3202)
- Codegen elision of dead branches for `if` statements with constant condition (not depending on enabled LLVM optimizations). (#3134)
- druntime: New `llvm_sideeffect` intrinsic, new `@cold` function UDA and extended CAS functionality in `core.atomic` (incl. support for weak CAS and separate failure ordering). (https://github.com/ldc-developers/druntime/pull/166, https://github.com/ldc-developers/druntime/pull/167, #3220)
- Windows: Bundled MinGW-based libs have been upgraded to use the .def files from MinGW-w64 v7.0.0. They now also contain a default `DllMain` entry point as well as `_[v]snprintf`. ([libs](https://github.com/ldc-developers/mingw-w64-libs/releases/tag/v7.0.0-rc.1), #3142)

#### Platform support
- Supports LLVM 3.9 - 9.0.

#### Bug fixes
- Misc. CMake issues with some LLVM 9 configurations. (#3079, #3198)
- Equality/identity comparisons of vectors with length ≥ 32. (#3208, #3209)
- `ldc.gccbuiltins_*` druntime modules now available to non-installed compiler too. (#3194, #3201)
- Potential ICE when applying `@assumeUsed` on global union. (#3221, #3222)
- `Context from outer function, but no outer function?` regression introduced in v1.11 (inability to access outer context from `extern(C++)` methods). (#3234, #3235)
- Lvalue expressions with nested temporaries to be destructed yielding a wrong lvalue. (#3233)
- druntime: Cherry-picked fix wrt. GC potentially collecting objects still referenced in other threads' TLS area. (dlang/druntime#2558)

# LDC 1.18.0 (2019-10-16)

#### Big news
- Frontend, druntime and Phobos are at version [2.088.1](https://dlang.org/changelog/2.088.1.html). (#3143, #3161, #3176, #3190)
- Support for **LLVM 9.0**. The prebuilt packages have been upgraded to [LLVM 9.0.0](http://releases.llvm.org/9.0.0/docs/ReleaseNotes.html). (#3166)
- Preliminary **Android** CI, incl. experimental prebuilt armv7a package generation (API level 21, i.e., Android 5+). (#3164)
- Bundled dub upgraded to v1.17.0+ with improved LDC support, incl. cross-compilation (e.g., `--arch=x86_64-pc-windows-msvc`). (https://github.com/dlang/dub/pull/1755, [Wiki](https://wiki.dlang.org/Cross-compiling_with_LDC))
- Init symbols of zero-initialized structs are no longer emitted. (#3131)
- druntime: DMD-compatible `{load,store}Unaligned` and `prefetch` added to `core.simd`. (https://github.com/ldc-developers/druntime/pull/163)
- JIT improvements, incl. multi-threaded compilation. (#2758, #3154, #3174)

#### Platform support
- Supports LLVM 3.9 - 9.0.

#### Bug fixes
- Don't error out when initializing a `void` vector. (#3130, #3139)
- druntime: Fix exception chaining for latest MSVC runtime v14.23, shipping with Visual Studio 2019 v16.3. (https://github.com/ldc-developers/druntime/pull/164)
- Keep lvalue-ness when casting associative array to another AA. (#3162, #3179)

# LDC 1.17.0 (2019-08-25)

#### Big news
- Frontend, druntime and Phobos are at version [2.087.1+](https://dlang.org/changelog/2.087.1.html). (#3093, #3124)
  - The upstream fix wrt. [local templates can now receive local symbols](https://issues.dlang.org/show_bug.cgi?id=5710) hasn't been ported yet. (#3125)
- LLVM for prebuilt packages upgraded to v8.0.1. (#3113)
- Breaking change: Init symbols, TypeInfos and vtables of non-`export`ed aggregates are now hidden with `-fvisibility=hidden`. (#3129)
- LLVM 8+: New intrinsics `llvm_*_sat` (saturation arithmetic) and `llvm_{min,max}imum`. Thanks Stefanos! (https://github.com/ldc-developers/druntime/pull/161, https://github.com/ldc-developers/druntime/pull/162)

#### Platform support
- Supports LLVM 3.9 - 8.0.

#### Bug fixes
- Fix for v1.16.0 regression when returning `void` expressions. (#3094, #3095)
- `-lowmem` (and on Windows, `--DRT-*` options) in response files (e.g., used by dub) aren't ignored anymore. (#3086)
- Windows: LDC and LDMD now internally use UTF-8 strings only, incl. command-line options and environment variables. The LDC install dir, source file names etc. can now contain non-ASCII chars. For proper console output, especially to stderr, you'll need Windows 10 v1809+ and may need to set a Unicode console font (e.g., Consolas). (#611, #3086)
- Android: Linker errors when building LDC/LDMD should be fixed. (#3128)
- Support for recent `gdmd` as D host compiler. Thanks Moritz! (#3087)
- Do not require gold plugin when linking with LLD. (#3105)
- Enable linker stripping on FreeBSD (with non-`bfd` linkers). (#3106)
- Some JIT bind fixes. (#3099, #3100)

#### Known issues
- If you encounter segfaults in GC worker threads with shared druntime on Linux that are fixed by disabling new parallel GC marking (e.g., via `--DRT-gcopt=parallel:0` in executable cmdline), please let us know about it: #3127

# LDC 1.16.0 (2019-06-20)

#### Big news
- Frontend, druntime and Phobos are at version [2.086.1](https://dlang.org/changelog/2.086.1.html), incl. a DIP1008 fix. (#3062, #3076, #3091)
- Non-Windows x86: Faster `real` versions of `std.math.{tan,expi}`. (#2855)
- dcompute: New `__traits(getTargetInfo, "dcomputeTargets")`. (#3090)

#### Platform support
- Supports LLVM 3.9 - 8.0 (incl. 7.1).

#### Bug fixes
- Make `pragma(LDC_no_typeinfo)` actually elide TypeInfo emission for structs, classes and interfaces. (#3068)
- Windows: Fix DLL entry point in MinGW-based libs. (https://github.com/ldc-developers/mingw-w64-libs/commit/8d930c129daa798379b3d563617847f8e895f43e)
- WebAssembly: Use `--export-dynamic` when linking with LLD 8+. (#3023, #3072)
- Fix corrupt `this` in functions nested in in/out contracts. (45460a1)
- Fix identity comparisons of integral vectors. (a44c78f)
- Improved handling of unsupported vector ops. (a44c78f)
- uClibc: Fix C assert calls. (#3078, #3082)
- Improved error message on global variable collision. (#3080, #3081)

# LDC 1.15.0 (2019-04-06)

#### Big news
- Frontend, druntime and Phobos are at version **2.085.1**, incl. new command-line options `-preview`, `-revert`, `-checkaction=context`, `-verrors-context` and `-extern-std`. (#3003, #3039, #3053)
  - The Objective-C improvements from DMD 2.085 are not implemented. (#3007)
- Support for **LLVM 8.0**. The prebuilt packages have been upgraded to LLVM 8.0.0 and include the Khronos SPIRV-LLVM-Translator, so that dcompute can now emit **OpenCL** too. (#3005)
- Compiler memory requirements can now be reduced via the new `-lowmem` switch, which enables the garbage collector for the front-end and sacrifices compile times for less required memory. In some cases, the overall max process memory can be reduced by more than 60%; see https://github.com/ldc-developers/ldc/pull/2916#issuecomment-443433594 for some numbers. (#2916)
  - Note for package maintainers: this feature requires a recent D host compiler (most notably, it doesn't work with ltsmaster), ideally LDC 1.15 itself due to important GC memory overhead improvements in 2.085 druntime.
- Support for generic `@llvmAttr("name")` parameter UDAs, incl. new `@restrict` with C-like semantics. (#3043)
- macOS: 32-bit support was dropped in the sense of not being CI-tested anymore and the prebuilt macOS package now containing x86_64 libraries only. `MACOSX_DEPLOYMENT_TARGET` for the prebuilt package has been raised from 10.8 to 10.9.
- Prebuilt packages don't depend on libtinfo and libedit anymore. (#1827, #3019)
- x86: SSSE3 isn't required for the prebuilt packages and generated optimized binaries anymore. (#3045)

#### Platform support
- Supports LLVM 3.9 - 8.0.

#### Bug fixes
- Implicit cross-module-inlining of functions annotated with `pragma(inline, true)` without explicit `-enable-cross-module-inlining` has been restored. (#2552, #3014)
- Propagate well-known length of newly allocated dynamic arrays for better optimizability. (#3041, #3042)
- JIT: Support implicit `__chkstk` calls for Windows targets, e.g., for large stack allocations. (#3051)

#### Internals
- Addition of **Azure Pipelines** as CI service. It is the new main CI service and responsible for creating all prebuilt x86(_64) packages. AppVeyor has been dropped completely and CircleCI rededicated. (#2998)

# LDC 1.14.0 (2019-02-17)

#### Big news
- Frontend, druntime and Phobos are at version **2.084.1**, incl. new command-line options `-mixin`, `-{enable,disable}-switch-errors` and `-checkaction`. (#2946, #2977, #2999)
  - Options `-release`, `-d-debug` and `-unittest` don't override preceding, more specific options (`-{enable,disable}-{asserts,invariants,preconditions,postconditions,contracts}`) anymore.
- Linking WebAssembly doesn't require `-link-internally` (integrated LLD) anymore; an external linker (default: `wasm-ld`, override with `-linker`) can be used as well. (#2951)
- Prebuilt Windows packages include LTO-able 32-bit druntime/Phobos too (previously: Win64 only).
- AddressSanitizer support for fibers (requires [rebuilding the runtime libraries](https://wiki.dlang.org/Building_LDC_runtime_libraries) with CMake option `RT_SUPPORT_SANITIZERS=ON`).  (#2975, https://github.com/ldc-developers/druntime/pull/152)
- Support `pragma(LDC_extern_weak)` for function declarations - if the function isn't available when linking, its address is null. (#2984)

#### Platform support
- Supports LLVM 3.9 - 7.0.

#### Bug fixes
- Fix C++ mangling regression for functions with multiple `real` parameters introduced with v1.13, preventing to build DMD. (#2954, https://github.com/dlang/dmd/pull/9129)
- Fix context of some nested aggregates. (#2960, #2969)
- Support templated LLVM intrinsics with vector arguments. (#2962, #2971)
- Avoid crashes with `-allinst` (fix emission of only speculatively nested functions). (#2932, #2940)
- Fix XRay support for LLVM 7+. (#2965)
- AArch64: Fix DMD-style profile measurements. (#2950)
- Be less picky about placement of pragmas (allow intermediate `extern(C)` etc.). (#2599)
- MSVC: Fix `real` C++ mangling to match Visual C++ `long double`. (#2974)
- Fix bad ICE noticed when building protobuf-d. (#2990, #2992)
- Fix ICE when directly indexing vector return value. (#2988, #2991)
- Fix identity comparisons of complex numbers. (#2918, #2993)
- MIPS32 fix for `core.stdc.stdarg`. (#2989, https://github.com/ldc-developers/druntime/pull/153)
- Fix `core.atomic.cas()` for 64-bit floating-point values. (#3000, #3001)

#### Known issues
- Buggy older `ld.bfd` linker versions may wrongly strip out required symbols, e.g., ModuleInfos (so that e.g. no module ctors/dtors are run). LDC defaults to `ld.gold` on Linux.

# LDC 1.13.0 (2018-12-16)

#### Big news
- Frontend, druntime and Phobos are at version **2.083.1**. (#2878, #2893, #2920, #2933)
- The **Windows packages are now fully self-sufficient**, i.e., a Visual Studio/C++ Build Tools installation isn't required anymore, as we now ship with MinGW-w64-based libraries, similar to DMD. Check out the included [README.txt](https://github.com/ldc-developers/ldc/blob/master/packaging/README.txt) for all relevant details. (https://github.com/dlang/installer/pull/346, https://github.com/ldc-developers/ldc/pull/2886, [Wiki: Cross-compiling with LDC](https://wiki.dlang.org/Cross-compiling_with_LDC))
- Debug info improvements:
  * For GDB: printing global and imported symbols, non-member and member function calls. (#2826)
  * For Visual Studio and mago: names, by-value params, nested variables. (#2895, #2908, #2909, #2912)
  * Associative arrays now showing up properly (at least with mago), not as opaque `void*` anymore. (#2869)
  * `-gc` now translates D names to C++ ones, e.g., to use the regular Visual Studio debugger (bypassing mago) and as preparation for VS Code debugging with Microsoft's C/C++ plug-in ([screenshots](https://github.com/ldc-developers/ldc/pull/2869#issuecomment-427862154)). Thanks to Oleksandr for this contribution and the AA fix! (#2869)
- New command-line option `-fvisibility=hidden` to hide functions/globals not marked as `export` (for non-Windows targets), primarily to reduce the size of shared libraries. Thanks to Andrey for stepping up! (#2894, #2923)
- Dropped support for LLVM 3.7 and 3.8. (#2872)
- LLVM for prebuilt packages upgraded to [v7.0.1](https://github.com/ldc-developers/llvm/releases/tag/ldc-v7.0.1).
- Linux: now defaulting to `ld.gold` linker in general, not just with `-flto=thin`, as buggy older `ld.bfd` versions may wrongly strip out required symbols (change with `-linker`). (#2870)
- Improved support for Android/x86[_64], musl libc and FreeBSD/AArch64. (#2917, https://github.com/ldc-developers/druntime/pull/146)
- LDC-specific druntime: `ldc.simd.inlineIR` moved/renamed to `ldc.llvmasm.__ir` (with deprecated legacy alias). (#2931)
- New CMake option `COMPILE_D_MODULES_SEPARATELY` builds D files in the DDMD frontend separately to reduce the time required to build LDC with many CPU cores and/or for iterative development. (#2914)

#### Platform support
- Supports LLVM 3.9 - 7.0.
- Alpine linux/x64: built against Musl libc to support Docker images based on the Alpine distro, requires the `llvm5-libs`, `musl-dev`, `binutils-gold` and `gcc` packages to build and link D apps and the `tzdata` and `curl-dev` packages for certain stdlib modules.

#### Bug fixes
- 32-bit Android/ARM regression introduced in v1.12. (#2892)
- Non-Windows x86_64 ABI fixes wrt. what's passed in registers, relevant for C[++] interop. (#2864)
- Alignment of `scope` allocated class instances. (#2919)


# LDC 1.12.0 (2018-10-13)

#### Big news
- Frontend, druntime and Phobos are at version **2.082.1**. (#2818, #2837, #2858, #2873)
  - Significant performance improvements for some transcendental `std.math` functions in single and double precision, at least for x86. (https://github.com/dlang/phobos/pull/6272#issuecomment-373967109)
- Support for **LLVM 7**, which is used for the prebuilt packages. Due to an LLVM 7.0.0 [regression](https://bugs.llvm.org/show_bug.cgi?id=38289), the prebuilt x86[_64] LDC binaries require a **CPU with SSSE3**, and so will your optimized binaries (unless compiling with `-mattr=-ssse3`). (#2850)
- **JIT compilation**: new `ldc.dynamic_compile.bind` function with interface similar to C++ `std::bind`, allowing to generate efficient specialized versions of functions (much like [Easy::jit](https://github.com/jmmartinez/easy-just-in-time) for C++). (#2726)
- LTO now working for Win64 too; the prebuilt package includes the required external LLD linker and the optional LTO default libs. Enable as usual with `-flto=<thin|full> [-defaultlib=druntime-ldc-lto,phobos2-ldc-lto]`. (#2774)
- Config file: new `lib-dirs` array for directories to be searched for libraries, incl. LLVM compiler-rt libraries. (#2790)

#### Platform support
- Supports LLVM 3.7 - 7.0.
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017.
- Alpine linux/x64: built against Musl libc to support Docker images based on the Alpine distro, requires the `llvm5-libs`, `musl-dev`, and `gcc` packages to build and link D apps and the `tzdata` and `libcurl` packages for certain stdlib modules.
- Android/ARM: This release slightly changes the way emulated TLS is interfaced, but is missing a patch for 32-bit ARM. [See the wiki for instructions on patching that file manually before cross-compiling the runtime libraries for 32-bit Android/ARM](https://wiki.dlang.org/Build_D_for_Android).

#### Bug fixes
- Fix IR-based PGO on Windows (requires our LLVM fork). (#2539)
- Fix C++ class construction with D `new` on Posix. (#2801)
- Android: No more text relocations in Phobos zlib, required for API level 23+. (#2822, #2835)
- Declare extern const/immutable globals as IR constants. (#2849, #2852)
- Fix issue when emitting both object and textual assembly files at once (`-output-o -output-s`). (#2847)
- Support address of struct member as key/value in AA literal. (#2859, #2860)
- Fix ICE when computing addresses relative to functions/labels. (#2865, #2867)


# LDC 1.11.0 (2018-08-18)

#### Big news
- Frontend, druntime and Phobos are at version **2.081.2**. (#2752, #2772, #2776, #2791, #2815)
  - Add some support for classes without TypeInfos, for `-betterC` and/or a minimal (d)runtime. (#2765)
- LLVM for prebuilt packages upgraded to v6.0.1. The x86_64 packages feature some more LLVM targets for cross-compilation (experiments): MIPS, MSP430, RISC-V and WebAssembly. (#2760)
- Rudimentary support for compiling & linking directly to **WebAssembly**. See the [dedicated Wiki page](https://wiki.dlang.org/Generating_WebAssembly_with_LDC) for how to get started. (#2766, #2779, #2785)
- **AArch64** (64-bit ARM) now mostly working on Linux/glibc and Android. Current `ltsmaster`/0.17.6 is able to bootstrap v1.11, which can also bootstrap itself; most tests pass. (Preliminary) [CI](https://app.shippable.com/github/ldc-developers/ldc/dashboard) has been set up. (#2802, #2817, #2813)
- LDC on Windows now uses 80-bit **compile-time** `real`s. This allows for seamless cross-compilation to other x86(_64) targets, e.g., without `real.min` underflowing to 0 and `real.max` overflowing to infinity. (#2752)
- New `@naked` UDA in `ldc.attributes` & enhanced functionality for `@llvmAttr("<name>")`. (#2773)

#### Platform support
- Supports LLVM 3.7 - 6.0.
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017.

#### Bug fixes
- `extern(C++)` on Posix: Pass non-PODs indirectly by value. (#2728)
- `extern(C++)` on Windows/MSVC: Methods return *all* structs via hidden sret pointer. (#2720, #1935)
- Make GC2Stack IR optimization pass work as intended. (#2750)
- Work around inline assembly regression with LLVM 6 on Win32. The prebuilt Win32 package is now using LLVM 6.0.1 too. (#2629, #2770)
- Fix overzealous check for multiple `main()` functions. (#2778)
- Fix corrupt prefix in integrated LLD's console output. (#2781)
- No context ptr for nested non-`extern(D)` functions. (#2808, #2809)

# LDC 1.10.0 (2018-06-19)

#### Big news
- Frontend, druntime and Phobos are at version **2.080.1**. (#2665, #2719, #2737)
  - No support for Objective-C class/static methods yet. (#2670)
- Breaking Win64 `extern(D)` ABI change: Pass vectors directly in registers, analogous to the MS vector calling convention. (#2714)
- Config file: For cross-compilation, support additional sections named as regex for specific target triples, e.g., `"86(_64)?-.*-linux": { … };`; see the comment in `etc/ldc2.conf`. (#2718)

#### Platform support
- Supports LLVM 3.7 - 6.0.
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017.

#### Bug fixes
- CMake and druntime fixes for DragonFlyBSD, thanks Diederik! (#2690, #2691, #2692, https://github.com/ldc-developers/druntime/pull/138, https://github.com/ldc-developers/druntime/pull/139, https://github.com/ldc-developers/phobos/pull/64)
- DMD-style inline asm label naming issue in overloaded functions. (#2667, #2694)
- Linux: misc. exception stack trace fixes & extensions, incl. default DWARF v4 debuginfo emission with LLVM 6. (#2677)
- Predefine version `D_HardFloat` instead of `D_SoftFloat` for `-float-abi=softfp`. (#2678)
- Bash completion installed to the wrong place with custom `CMAKE_INSTALL_PREFIX`. (#2679, #2179, #2693)
- Default to `ld.gold` linker for ThinLTO on Linux. (#2696)
- Fix compilation issues on 64-bit macOS with DMD host compiler ≥ 2.079. (#2703, #2704)
- druntime: Fix `core.stdc.stdint.(u)int64_t` on 64-bit macOS etc. (#2700)
- Define `D_AVX` and `D_AVX2` if the target supports them. (#2711)
- Fix sporadic front-end segfaults. (#2713)
- Win64: Fix `extern(C++)` ABI wrt. passing small non-POD structs by value. (#2706)
- Misc. druntime/Phobos fixes and upstream cherry-picks for ARM, AArch64, MIPS etc.
- Fix potential LDC crashes when returning static array results from inline IR. (#2729)
- Win64: Fix terminate handler for VC runtime DLL version 14.14.x.y. (#2739)

# LDC 1.9.0 (2018-04-30)

#### Big news
- Frontend, druntime and Phobos are at version **2.079.1**, incl. new switches `-i[=<pattern>]` (include imports in compilation) and `-Xi`. (#2587)
  - Support a **minimal (d)runtime**. (#2641)
  - Win32 breaking ABI change: add extra underscore for mangled names of D symbols. (#2598)
  - *No* breaking ABI change for 64-bit macOS wrt. C++ mangling of D `(u)long`. It's still mangled as C++ `(unsigned) long` in order not to break `size_t` and `ptrdiff_t` interop, whereas DMD 2.079 mangles it as `(unsigned) long long` (which, in combination with missing `core.stdc.config.cpp_(u)long`, makes it impossible to represent a C++ size_t/ptrdiff_t with DMD 2.079 on 64-bit macOS).
- Support for **LLVM 6**. It's used for the prebuilt packages, except for the 32-bit Windows package (due to #2629). (#2608)
- Integrated LLD (enable with `-link-internally`) now also able to **(cross-)link ELF and Mach-O binaries**, in addition to the existing Windows COFF support. (#2203)
- Prebuilt Linux and macOS packages now ship with **LTO default libs** (druntime & Phobos). Keep on using `-flto=<thin|full>` to restrict LTO to your code, or opt for `-flto=<thin|full> -defaultlib=phobos2-ldc-lto,druntime-ldc-lto` to include the default libs. (#2640)
- When linking against shared default libs, LDC now sets a default rpath (absolute path to the LDC lib dir(s); configurable in the `etc/ldc2.conf` file). (#2659)
- New convenience mixin for fuzzing: `ldc.libfuzzer.DefineTestOneInput`. (#2510)

#### Platform support
- Supports LLVM 3.7 - 6.0.
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017.

#### Bug fixes
- DMD-style inline asm:
  - Fix semantics of `extended ptr` for MSVC targets. (#2653)
  - Add missing EIP register. (#2654)
- macOS: Fix install_name and symlinks of shared fat druntime/Phobos libs. (#2659, #2615)
- Make `-static` override `-link-defaultlib-shared`. (#2646)
- Make interface thunks forward variadic args. (#2613)
- Fix `va_arg()` for PowerPC. (https://github.com/ldc-developers/druntime/pull/121)
- MSVC: Support exporting naked functions. (#2648)
- Only emit interface vtables in the declaring module. (#2647)
- Call `_Unwind_Resume()` directly. (#2642)

# LDC 1.8.0 (2018-03-04)

#### Big news
- Frontend, druntime and Phobos are at version **2.078.3**, incl. new switches `-dip1008` and `-transition=<intpromote|16997>` as well as `pragma(crt_{con,de}structor)`. (#2486)
- New switch `-link-defaultlib-shared` to link against shared druntime/Phobos. It defaults to true for shared libraries (`-shared`), so it's primarily useful for executables. (#2443)
- Support for plugins via `-plugin=...` (see [this example](https://github.com/ldc-developers/ldc/tree/master/tests/plugins/addFuncEntryCall)). The mechanism is identical to Clang's LLVM-IR pass plugins and thus supports those as well, e.g., the [AFLfuzz LLVM-mode plugin](https://github.com/mirrorer/afl/blob/master/llvm_mode/afl-llvm-pass.so.cc), [Easy::Jit](https://github.com/jmmartinez/easy-just-in-time). (#2554)
- Support for LLVM IR-based Profile-Guided Optimization via `-fprofile-{generate,use}` (not working on Windows yet). (#2474)
- Basic support for [LLVM XRay instrumentation](https://llvm.org/docs/XRay.html) via `-fxray-{instrument,instruction-threshold}`. (#2465)
- DMD-style function trace profiling via `-profile` (LDMD) / `-fdmd-trace-functions` (LDC). (#2477)
- New UDA `ldc.attributes.assumeUsed` to prevent a symbol from being optimized away. (#2457)
- The PGO helper library `ldc-profile-rt` was replaced by LLVM's vanilla profiling library. Our subset of [LLVM compiler-rt](https://compiler-rt.llvm.org/) libraries is now also shipped on Windows (excl. fuzzer). (#2527, #2544)
- Cherry-picked upstream Musl C runtime support for Docker images based on Alpine and added a native Alpine/x64 compiler, which requires the `llvm5`, `musl-dev`, and `gcc` packages to run and link D apps and the `tzdata` and `libcurl` packages for certain stdlib modules.

#### Platform support
- Supports LLVM 3.7 - 5.0.
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017.

#### Bug fixes
- Strict left-to-right evaluation/load order of function arguments. (#2450, #2502)
- Inline asm silently ignores opcodes db, ds, di, dl, df, dd, de. (#2548)
- Missed optimization for `scope` allocated classes. (#2515, #2516)
- Don't eliminate frame pointer by default at `-O0`. (#2480, #2483)
- LLVM complaining about invalid IR pointer arithmetics. (#2537)
- `llvm_expect()` doesn't work with CTFE. (#2458, #2506)
- `.{so,dylib}` file command line arguments should be forwarded to linker. (#2445, #2485)
- macOS: Set shared stdlib install_name to `@rpath/<filename>`. (#2442, #2581)
- `array ~= element` issue if rhs affects the lhs length. (#2588, #2589)
- EH segfaults when checking D class catch handlers against thrown C++ exception. (#2590)

# LDC 1.7.0 (2018-01-06)

#### Big news
- Frontend, druntime and Phobos are at version **2.077.1**. (#2401, #2430)
- **C++ exceptions** can now be caught in D code, for Linux and MSVC targets (and possibly more). A logical step after consolidating LDC's exception handling for non-MSVC targets with DMD's DWARF implementation. (#2405)
- Automated building of release and [CI packages](https://github.com/ldc-developers/ldc/releases/tag/CI). (#2438)

#### Platform support
- Supports LLVM 3.7 - 5.0. (binary packages on GitHub are built with LLVM 5.0.1)
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017.

#### Bug fixes
- ICE on chained ref-returning opIndex op-assign. (#2415)
- Windows: `export` visibility ignored for globals. (#2437)
- Print error message when trying to use shared libraries with static runtime. (#2454)
- ldc-1.7.0-beta1 regression: ICE with implicit cast. (#2471)
- CMake: use llvm-config to determine LLVM's cmake directory, if possible. (#2482)

# LDC 1.6.0 (2017-11-26)

#### Big news
- Frontend, druntime and Phobos are at version **2.076.1** (#2362), including `-betterC` semantics (#2365).
- Experimental support for **dynamic codegen at runtime** (JIT-style) to tune performance-critical parts for the used CPU and/or treat special runtime variables as constants. See UDAs `@dynamicCompile`, `@dynamicCompileConst` in `ldc.attributes`; compile with command-line option `-enable-dynamic-compile` and use the `ldc.dynamic_compile` module to generate the code at runtime before invoking it. Congratulations to Ivan Butygin for implementing this non-trivial feature! (#2293)
- Many `std.math` functions are now CTFE-able. (#2259)

#### Platform support
- Supports LLVM 3.7 - 5.0.
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017.

#### Bug fixes
- Can't link against wsock32 and ws2_32 on Windows. (#468)
- PGO incompatible with MSVC EH. (#1943)
- Regression: ModuleInfos not emitted as COMDATs. (#2409)
- Incorrect C assert function signature for Android. (#2417)
- Overzealous error check when attempting to evaluate object as constant. (#2422)

# LDC 1.5.0 (2017-10-29)

#### Big news
- Frontend, druntime and Phobos are at version **2.075.1**. (#2252)
- New command-line option `-fp-contract` to control fused floating-point math, as well as about 25 new hidden options influencing codegen, see `-help-hidden` (`-enable-unsafe-fp-math`, `-debugger-tune` etc.). (#2148)
- New command-line option `-linker`. Use `-linker=lld-link` to use an external LLD executable for MSVC targets (with experimental LTO support) or `-linker=<gold|bfd|lld>` for other targets. (#2386)

#### Breaking changes
- Win32: the mangled names of D symbols now start with `_D`, not with `__D`, compatible with DMD. (#2353)

#### Platform support
- Supports LLVM 3.7 - 5.0.
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017.

#### Changes to the prebuilt packages
- LLVM upgraded to [5.0.0](https://github.com/ldc-developers/llvm/releases/tag/ldc-v5.0.0-2).

#### Bug fixes
- Cyclic dependencies with `-cov`. (#2177)
- ICE when capturing `this` in constructors. (#1728)
- Objective-C bugs. (#2387, #2388)
- LLVM/LLD 5.0: `-link-internally` broken. ([LLD patch](https://github.com/ldc-developers/llvm/releases/tag/ldc-v5.0.0-2))
- LLVM 5.0: need to build LDC with CMake option `-DLDC_WITH_LLD=OFF` to avoid conflicting command-line options. (#2148)
- LLVM 5.0 & non-Windows targets: names of members in static libraries generated by LDC's internal archiver contain path information. (#2349)
- ~~Workaround for Xcode 9 ranlib bug: don't use internal (LLVM) archiver by default for OSX targets. (#2350)~~ Xcode 9.0.1 fixes that bug, please upgrade.
- Captured lazy parameters may be garbage. (#2302, #2330)
- Packed struct layout regression (#2346) and `T.alignof` not respecting explicit type alignment via `align(N)`. (#2347)
- OSX and Win32: mangling issue for druntime's `rt_options`. (#1970, #2354)
- MinGW Win64: ABI regression wrt. functions returning x87 reals. (#2358)
- Potential file permission problem when copying over LLVM libraries during LDC build. (#2337)
- PPC64: Forward reference error with 1.3 release. (#2200)

#### Known issues
- LLVM 5.0: potentially failing LLVM assertion when emitting debuginfos and using inlining at the same time. (#2361)

# LDC 1.4.0 (2017-09-11)

#### Big news
- Frontend, druntime and Phobos are at version **2.074.1**. (#2076)
- **ldc-build-runtime**: a small D tool that makes it easy to compile the LDC runtime and standard library yourself, for example, to enable LTO-ability/sanitizers or cross-compiling executables and shared libraries for other platforms, like Android/ARM. ([Wiki page](https://wiki.dlang.org/Building_LDC_runtime_libraries))
- @joakim-noah's Android fixes have finally been fully incorporated, enabling every host to (cross-)compile to Android. (https://github.com/ldc-developers/llvm/commit/8655f3208cce28bb7f903cadf5f58a3911392bdc)  [Instructions on using this ldc release to cross-compile D apps for Android are on the wiki](https://wiki.dlang.org/Build_D_for_Android), including how to try out the native Android/arm package, ie a D compiler that you can run _on_ your Android smartphone or tablet.
- Improved support for [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer). LDC will automatically link with the AddressSanitizer runtime library when `-fsanitize=address` is passed (when LDC can find the AddressSanitizer library).
- [libFuzzer](https://llvm.org/docs/LibFuzzer.html) sanitizer support using `-fsanitize=fuzzer` (same as Clang). This flag implies `-fsanitize-coverage=trace-pc-guard,indirect-calls,trace-cmp` and automatically links-in the runtime libFuzzer library if LDC can locate the runtime library. (With LLVM 4.0, there is a dependency on sanitizer runtime, so manually link the ASan library or use `-fsanitize=fuzzer,address`.)
- New `-fsanitize-blacklist=<file>` command-line option to exclude functions from sanitizer instrumentation (identical to Clang). The file must adhere to the textual [Sanitizer Special Case List format](https://clang.llvm.org/docs/SanitizerSpecialCaseList.html).
- New `-fsanitize-coverage=...` command-line option with the same [functionality as Clang](https://clang.llvm.org/docs/SanitizerCoverage.html).
- The config file sections now feature an additional `post-switches` list for switches to be appended to the command line (the existing `switches` list is prepended). E.g., this now allows the user to override the directory containing the runtime libraries via `-L-L/my/runtime/libs` in the command line. (#2281)

#### Breaking changes
- The `-sanitize` command-line option has been renamed to `-fsanitize*`, for clang conformance.
- The semantics of an empty `-conf=` command-line option have been changed from 'use default config file' to 'use no config file'.
- The binary representations of the init values for `float/double/real` have been unified to a special quiet NaN, with both most significant mantissa bits set, on all hosts and for all targets. (#2207)

#### Platform support
- Supports LLVM 3.7 - 5.0. Support for 3.5 and 3.6 has been dropped.
- Windows: Supports Visual Studio/C++ Build Tools 2015 and 2017, incl. the latest Visual Studio 2017 Update 15.3.

#### Changes to the prebuilt packages
- Consistent usage of a [minimally tailored](https://github.com/ldc-developers/llvm/releases/tag/ldc-v4.0.1) LLVM 4.0.1.
- Newly enabled LLVM target `NVPTX` in order to target [CUDA via DCompute](http://forum.dlang.org/thread/smrnykcwpllukwtlfzxg@forum.dlang.org).
- Linux x86_64:
  - Shipping with the LLVM **LTO plugin** for the `gold` linker. On Ubuntu 14.04 and later, `-flto=full|thin -Xcc=-fuse-ld=gold` should work out of the box.
  - Build environment upgraded from Ubuntu 12.04 and gcc 4.9 to Ubuntu 14.04 and gcc 6.3.
- Windows/MSVC: Build environment upgraded from Visual Studio 2015 Update 3 to Visual Studio 2017 15.3.3 (WinSDK 10.0.15063).

#### Bug fixes
- Misc. debuginfo issues, incl. adaptations to internal LLVM 5.0 changes: (#2315)
  - `ref` parameters and closure parameters declared with wrong address and hence potentially showing garbage.
  - Win64: parameters > 64 bit passed by value showing garbage.
  - Win64: debuginfos for closure and nested variables now finally available starting with LLVM 5.0. 
- LLVM error `Global variable initializer type does not match global variable type!` for `T.init` with explicit initializers for dominated members in nested unions. (#2108)
- Inconsistent handling of lvalue slicees wrt. visible side-effects of slice lower/upper bound expressions. (#1433)
- Misc. dcompute issues. (#2195, #2215)
- Potential LDC crashes due to dangling pointers after replacing IR globals (required in some cases if the type contains unions) almost fully mitigated. (#1829)
- Multiple arrayop function emissions. (#2216)
- Potentially incorrect memory layout for unnaturally aligned aggregates. (#2235)
- Wrong `-m32/64` in command-line for external ARM assembler used via `-no-integrated-as`.

#### Internals
- Misc. CI improvements:
  - Addition of high-performant SemaphoreCI (incl. enabled LLVM/LDC assertions).
  - CircleCI upgraded to 2.0, testing with latest gcc.
- Compile all D files for (non-unittest) druntime/Phobos at once. May be disabled via CMake option `COMPILE_ALL_D_FILES_AT_ONCE=OFF`. (#2231)

#### Known issues
- ThinLTO may not work well with the `ld.bfd` linker, use `ld.gold` instead (`-Xcc=-fuse-ld=gold`).
- When building with LLVM 5.0, you may need `-DLDC_WITH_LLD=OFF` in the CMake command line. Otherwise, if the LLD headers are available and LDC is built with LLD integration, the produced LDC binary will refuse to work due to conflicting command-line options.

# LDC 1.3.0 (2017-07-07)

#### Big news
- Frontend, druntime and Phobos are at version **2.073.2**.
- A first experimental version of DCompute for **OpenCL/CUDA** targets has landed. See [announcement](http://forum.dlang.org/thread/zcfqujlgnultnqfksbjh@forum.dlang.org).
- LLVM 3.9+: Experimental integration of **LLD**, the LLVM cross-linker, for MSVC targets. Check out these hassle-free [instructions](https://github.com/ldc-developers/ldc/pull/2142#issuecomment-304472412) to make LDC emit Windows executables and DLLs on any host! (#2142)
- libconfig was replaced by an ad-hoc parser (in D), getting rid of the build and runtime dependency and shrinking the license file by roughly 50%. Thanks again, Remi! (#2016)
- LDC now ships with static and shared runtime libs on supported platforms. (#1960)
- LLVM 3.9+: Static libraries are now generated by LDC (LLVM) by default, not by system `ar` or `lib.exe`. This means that LDC can cross-compile and -archive static libs for all supported targets. Command-line option `-ar` allows specifying an external archiver to be used. (#2030)
- New command-line options `-dip1000`, `-mv` and `-mscrtlib` (#2041).
- Ships with dlang tools rdmd, ddemangle and dustmite.

#### New features
- LLVM 4.0+: Output LLVM optimization records via `-fsave-optimization-record`. (#2089)
- New `-Xcc` command-line option for C compiler flags when linking via gcc/clang. Thanks Adrian! (#2104)
- New function UDA `@ldc.attributes.llvmFastMathFlag("contract")` that specifically enables floating point operation fusing (fused multiply-add), previously only achievable with `@fastmath`. (#2060)

#### Platform support
- Supports LLVM 3.5 - 4.0.
- Additional LLVM targets have been enabled for the prebuilt x86_64 packages: ARM, AArch64 and PowerPC.
- Windows: Supports Visual Studio/Build Tools 2015 and **2017**. (#2065)
- NetBSD: The 2.074 druntime patches have been cherry-picked.

#### Bug fixes
- LTO flags leaking into standard libraries when building LDC with LTO. (#2077)
- Debug info fixes for class types - thanks Elie! (#2130)
- OSX: Incomplete backtrace. (#2097)
- Phobos on ARM: alignment and 64-bit `real` issues. (#2024)
- Windows: EH-related crashes when linking against shared MS runtimes. (#2080)
- ICE when initializing vector with `TVector.init`. (#2101)
- Weird object file type autodetection. (#2105)
- Typesafe variadics emitted as LLVM variadics. (#2121)
- Superfluous masking of `bool` values. (#2131)
- Output for `-mcpu=help` or `-mattr=help` printed multiple times. (#2073)
- LDMD refuses some duplicate command-line options. (#2110)
- Change format of predefined versions output for DMD compatibility. (#1962)
- Fix potential segfault when formatting error msg (#2160)
- Fix ICE when using `-main -cov` (#2164)
- Make inlining threshold customizable via (existing) `-inline-threshold`, fix performance decrease with `-boundscheck=off` (#2161, #2180)
- Switch Android onto the sectionELF style of module registry (#2172)
- Check fiber migration on Android/ARM too (https://github.com/ldc-developers/druntime/pull/97)
- Android moduleinfo section druntime fix (https://github.com/ldc-developers/druntime/pull/98)

#### Building LDC
- Building LDC requires a preinstalled D compiler.

#### Internals
- LDC now features D unittests itself. Just add some to LDC's D modules and they'll be compiled and executed by CI. (#2016)

# LDC 1.2.0 (2017-04-21)

#### Big news
- Frontend, druntime and Phobos are at version **2.072.2**.

#### New features
- New function attribute `@ldc.attributes.allocSize` (#1610), see https://wiki.dlang.org/LDC-specific_language_changes#.40.28ldc.attributes.allocSize.29

#### Platform support
- Supports LLVM 3.5 - 4.0.
- Exception backtrace robustness has been significantly improved.
- Emission of `reals` with differing precision to the host platform's `real` has been fixed. <br />(The compiler still uses the host platform's D real type to represent compile-time floating-point values, so parsing of literals and CTFE is restricted to the host real precision. For instance, LDC on AArch64 with its quad-precision reals would now make for a universal cross-compiler. On the other hand, cross-compiling from ARM to x86 with 80 bit reals still does not work – for example, `real.max` would silently overflow at compile-time and be emitted as 80-bit infinity.)

#### Bug fixes
- Compilation error with DMD 2.074 host compiler.
- LLVM error when accessing `typeid(null)` (#2062).
- Some LLVM intrinsics not available for LLVM ≥ 4.0 (#2037).
- Spurious crashes on OS X user program shutdown when linking against static druntime lib.
- Lexing floating-point literals may fail on PowerPC (#2046).
- LDC crashes when trying to repaint static arrays (#2033).
- No stack trace on Linux (#2004) and Windows (#1976, https://github.com/ldc-developers/druntime/pull/85).
- Generated documentation file is immediately deleted when compiling at the same time.
- LDMD doesn't append default file extension if `-of` option doesn't contain any (#2001, #2002).

#### Building LDC
- Building LDC requires a preinstalled D compiler.

# LDC 1.1.1 (2017-02-23)

#### Bug fixes
- Linux: Always build C parts of standard library as PIC (#2009). This makes the binary packages usable on Ubuntu 16.10 (where executables are linked as position-independent code by default, in contrast to the older system used for preparing the packages).

# LDC 1.1.0 (2017-01-26)

#### Big news
- Frontend, druntime and Phobos are at version **2.071.2**.
- **[Link-Time Optimization (LTO)](https://johanengelen.github.io/ldc/2016/11/10/Link-Time-Optimization-LDC.html)** with `-flto={thin|full}` (LLVM ≥ 3.9). LTO requires linker support and is therefore currently only supported on Linux (`ld.gold` with LLVM plugin) and OS X. For more details, please refer to LLVM's and Clang's documentation, for example [Clang's ThinLTO documentation](http://clang.llvm.org/docs/ThinLTO.html).  (#1840)
- **Experimental cross-module inlining** (#1577, enable with `-enable-cross-module-inlining`)
- **[Profile-guided optimization (PGO)](https://johanengelen.github.io/ldc/2016/07/15/Profile-Guided-Optimization-with-LDC.html)** (#1219)
- Windows: enable C-style DLL exports/imports via `export` (functions only) (#1856)
- Experimental IR-to-obj caching with `-cache=<cache dir>` (#1572, #1753, #1812, #1893)
- Accept bitcode files on commandline (#1539)
- `@ldc.attributes.fastmath` for [aggressive math optimization](https://johanengelen.github.io/ldc/2016/10/11/Math-performance-LDC.html) (#1472, #1438)
- Binary distribution now bundles DUB (v1.2.0) (#1573)
- **Breaking changes to command-line semantics** (see http://forum.dlang.org/post/ubobkfmsspbsmjunosna@forum.dlang.org).

#### New features
- New traits `__traits(targetCPU)` and `__traits(targetHasFeature, )` (#1434)
- Drastic reduction of large symbol name lengths with optional `-hash-threshold` (#1445)
- `@ldc.attributes.optStrategy(...)` for per-function optimization setting (#1637)
- Extend intrinsic `llvm_memory_fence` for single-thread fences (#1837)
- Add function instrumentation and profiling options via `-finstrument-functions` (#1845)
- Add line-tables-only debuginfo via `-gline-tables-only` (#1861)
- Implement DMD-compatible `-betterC` (#1872)

#### Platform support
- Supports LLVM 3.5 - 3.9 and current 4.0 release candidate.
- ABI fixes, mainly for PowerPC targets. For bootstrapping, make sure to use source branch `ltsmaster` or the latest 0.17.x release, as all existing LDC releases ≥ 1.0 for PowerPC are unusable. (#1905)
- Added ARM assembly code for Phobos `std.bigint`. (https://github.com/ldc-developers/phobos/pull/31)
- Added some definitions for OpenBSD. (https://github.com/ldc-developers/druntime/commit/1ef83229673f5ae23f6a2a97f8e6b039647fbf87)
- Updates for Solaris (https://github.com/ldc-developers/druntime/pull/71, https://github.com/ldc-developers/druntime/pull/72, https://github.com/ldc-developers/druntime/pull/73, https://github.com/ldc-developers/druntime/pull/74, https://github.com/ldc-developers/druntime/pull/75, https://github.com/ldc-developers/druntime/pull/79)
- Linux: changed default to fully relocatable, position independent code (PIC). Change back to non-relocatable with `-relocation-model=static`. (#1664)

#### Bug fixes
- Potential crash when generating debuginfos for nested variables AND optimizing (#1933, #1963, #1984)
- Alignment and size of critical sections, causing crashes on ARM (#1955, #1956)
- `-finstrument-functions` using wrong return address (#1961)
- Response files expanded too late, preventing cross-compilation on Windows when using dub (#1941, #1942)
- Non-Windows x86_64 ABI bug wrt. returning static arrays (#1925, #1938)
- Some array literals wrongly promoted to constants (#1924, #1927)
- Misc. DUB regressions introduced by beta 3 (#1819)
  - Don't output static libs (with relative target filename) in `-od` objects directory (for LDC, but continue to do so for LDMD, for DMD compatibility).
  - LDMD: avoid object file collisions (due to multiple D source files with identical name in different dirs) when creating a static lib and remove the object files on success, mimicking DMD.
  - Create output directories recursively.
- Potential ICE when building vibe.d projects (#1741)
- ICE when calling an abstract function. (#1822)
- ICE for invalid `__asm` constraints. (#802)
- Wrong code for LLVM inline assembly returning a tuple (`__asmtuple`). (#1823)
- Potential ICE wrt. captured variables. (#1864)
- ARM: ICE when using LTO. (#1860)
- Union layout and initialization, fixing the compilation of DMD (#1846, fixing most cases of #1829)
- Allow custom file extension for .ll/.bc./.s output files. (#1843)
- Windows: produced binaries with debuginfos are now large-address-aware too. (#442, #1876)
- Fix debuginfos for parameters. (#1816)
- Allow alignment of global variables < pointer size. (#1825)
- Promote more immutable array literals to LLVM constants. (#506, #1821, #1838)
- ICE when incrementing a complex variable. (#1806)
- `llvm.va_start` not matched with `llvm.va_end` (#1744)
- ldmd2 ignores -od option for libraries. (#1724)
- ICE: toConstElem(CastExp) doesn't support NewExp as cast source. (#1723)
- Mark runtime intrinsic shims as pragma(inline, true). (#1715)
- pragma(inline, false) is incompatible with store/loadUnaligned. (#1711)
- ICE: function not fully analyzed; previous unreported errors compiling std.variant.VariantN!(16LU, int, string).VariantN.__xopEquals? (#1698)
- Segfault at at ldc/ldc-1.1.0/driver/main.cpp:1351. (#1696)
- Make sure MSVC Build Tools are automatically detected by LDC. (#1690)
- Update Windows README.txt. (#1689)
- [ldc2-1.1.0-beta2] Missing symbol with inlining enabled. (#1678)
- [REG ldc-1.1.0-beta2] ICE with templated classes. (#1677)
- FreeBSD: Fix shared library build, working Hello World. (#1673)
- Strange compile time error. (#1638)
- LDC+DUB on Windows: folder separator is ignored. (#1621)
- Fix evaluation order issues. (#1620, #1623)
- Ubuntu 16.10 linker failures due to PIE by default (relocation R_X86_64_32S … can not be used). (#1618)
- ICE on returning struct with zero-length static array. (#1611)
- Debug info generation fixes for LLVM >= 3.8. (#1598)
- ICE after return in the middle of a function on Win64/MSVC. (#1582)
- Enums with referenced struct members result in floating point error. (#1581)
- `pragma(inline, {true|false})` is no longer ignored (#1577)
- Static array initialization with single element misdetected as direct construction via sret. (#1548)
- ICE on static typeid. (#1540)
- super doesn't work. (#1450)
- Sub-expression evaluation order fixes. (#1327)
- Add suffix to LTO linker plugin name to disambiguate with LLVM installation. (#1898)

#### Building LDC
- LDC now requires a preinstalled D compiler. (Versions `0.17.*` and the `ltsmaster` branch can be used to 'bootstrap' a build when only a C++ compiler is available.)
- On Unix-like systems we now use gcc for linking. (#1594)

#### Internals
- optimizer: Skip adding verifier function pass if `-disable-verify` is given. (#1591)
- DValue refactoring. (#1562)
- Several improvements to generated IR. (#1528, #1630)
- The vtable's of inherited interfaces are now put between the class's _monitor field and the user data fields. (https://issues.dlang.org/show_bug.cgi?id=15644)

# LDC 1.0.0 (2016-06-03)

#### Big news
- Frontend, druntime and Phobos are at version **2.070.2**.

#### Platform support
- Support for LLVM 3.5 - 3.8 and preliminary support for LLVM 3.9.
- Objective-C Support. (#1419)
- ARM platform is now fully supported. (#1283, #489)
- Better support for Android. (#1447)
- Preliminary support for AArch64.

#### Bug fixes
-  Outdated Copyright notice in LICENSE file. (#1322)
-  libconfig.so.8 not found (ubuntu 14.04) (#1460)
- Wrong template filter on atomicOp. (#1454)
- Runtime error on synchronized(typeid(SomeInterface)) { }. (#1377)
- TypeInfo is stored read-only, but mutable from D. (#1337)
- Inline assembly regression with local variable references. (#1292)
- Compile error on Linux/PPC and Linux/PPC64 due to missing import in Phobos.

#### Building LDC
- LDC now requires a preinstalled D compiler.
- Building on OS X requires ld64-264 or above (shipping with Xcode 7.3). This avoid spurious crashes during exception handling. XCode 7.3.1 should be used to avoid linker errors. (#1444, #1512)

#### Internals
- Linking against LLVM shared library is now supported.

# LDC 0.17.6 (2018-08-24)

#### News
- Added support for **LLVM 6.0 and 7.0**. (https://github.com/ldc-developers/ldc/pull/2600, https://github.com/ldc-developers/ldc/pull/2825)
- Backported **AArch64** fixes from master; most tests passing on Linux/glibc and Android. (https://github.com/ldc-developers/ldc/pull/2575, https://github.com/ldc-developers/ldc/pull/2811, https://github.com/ldc-developers/phobos/pull/49, https://github.com/ldc-developers/phobos/pull/50, https://github.com/ldc-developers/phobos/pull/51, https://github.com/ldc-developers/phobos/pull/52, https://github.com/ldc-developers/phobos/pull/53, https://github.com/ldc-developers/phobos/pull/54, https://github.com/ldc-developers/phobos/pull/55, https://github.com/ldc-developers/phobos/pull/56)
- Fix generation of debug info. (https://github.com/ldc-developers/ldc/pull/2594)
- Added support for bootstrapping on **DragonFly BSD**. (https://github.com/ldc-developers/ldc/pull/2580, https://github.com/ldc-developers/ldc/pull/2593, https://github.com/ldc-developers/ldc/pull/2689, https://github.com/ldc-developers/druntime/pull/110, https://github.com/ldc-developers/phobos/pull/45)
- Fixed missing definition in `std.datetime` on Solaris. (https://github.com/ldc-developers/phobos/pull/46)
- Fixed `std.datetime` unittest failure. (https://github.com/ldc-developers/phobos/pull/59)
- Fixed tests for PowerPC. (https://github.com/ldc-developers/ldc/pull/2634, https://github.com/ldc-developers/ldc/pull/2635)
- Improvements for **MIPS**.
- Make `core.stdc.stdarg.va_*` functions `nothrow` to enable compiling the **2.082** frontend. (https://github.com/ldc-developers/ldc/pull/2821)
- CI updates.

# LDC 0.17.5 (2017-09-12)

#### News
- Added LLVM 5.0 support.
- druntime: fixes for Android and addition of `core.math.yl2x[p1]()` for x86(_64) targets.
- dmd-testsuite: backported `runnable/cppa.d` fix for GCC > 5.
- CI updates.

# LDC 0.17.4 (2017-03-23)

#### News
- Added LLVM 4.0 support.

# LDC 0.17.3 (2017-02-01)

#### Big news
- Full stdlib and dmd testsuite passes on Android/ARM.

#### Bug fixes
- Fixes for PPC64-LE, MIPS64 and ARM/AArch64 ABIs (#1905)

# LDC 0.17.2 (2016-10-09)

#### Platform support
- Support for LLVM 3.5 - 3.9.

#### Bug fixes
- Fixed soft float and hard float issues on ARM.
- Fixed ABI error on Linux/PPC and Linux/PPC64.
- Fixed error in `core.stdc.stdarg` on Linux/PPC and Linux/PPC64.
- Fixed issue with `__tls_get_addr` on Linux/PPC and Linux/PPC64.

# LDC 0.17.1 (2016-03-22)

#### Big news
- ARM platform is now a first class target for LDC. It passes most test cases (except 2 failures) and can successfully compile the D version of the compiler.

#### Platform support
- ARM platform is now fully supported. (#1283, #489)
- Preliminary support for AArch64.
- Preliminary support for LLVM 3.9.

#### Bug fixes
- Inline assembly regression with local variable references. (#1292)
- Compile error on Linux/PPC and Linux/PPC64 due to missing import in Phobos.

#### Building LDC
- Linking against LLVM shared library is now supported.

# LDC 0.17.0 (2016-02-13)

#### Big news:
- Frontend, druntime and Phobos are at version **2.068.2**.
- The **exception handling** runtime now **no** longer allocates **GC** memory (although it still uses C `malloc` if there are more than 8 concurrent exceptions or nested `finally` blocks per thread). _Note:_ Creating the `Throwable`s in user code (e.g. `new Exception("…")`) and the `Runtime.traceHandler` may GC-allocate still. (Thanks for this goes to our newest contributor, @Philpax).
- The `@ldc.attributes.section("…")` attribute can now be used to explicitly specify the object file section a variable or function is emitted to.
- The `@ldc.attributes.target("…")` attribute can now be used to explicitly specify CPU features or architecture for a function.
- The `-static` option can be used to create fully static binaries on Linux (akin to the GCC option of the same name).
- `core.atomic.atomicOp()` now exploits LLVM read-modify-write intrinsics instead of using a compare-and-swap loop. As side-effect, the atomic intrinsics in module `ldc.intrinsics` have been renamed:
  - `llvm_atomic_cmp_swap` => `llvm_atomic_cmp_xchg`
  - `llvm_atomic_swap` => `llvm_atomic_rmw_xchg`
  - `llvm_atomic_load_*` => `llvm_atomic_rmw_*`

#### Platform support:
- Improved ARM support. (#1280)
- The compiler now supports NetBSD. (#1247) (Thanks for this goes to @nrTQgc.)
- The float ABI can now be derived from the second field of the triple. E.g. the hardfloat ABI is used if triple `armv7a-hardfloat-linux-gnueabi` is given. (#1253)
- Support for fibers on AArch64.
- Support for LLVM 3.8 and preliminary support for LLVM 3.9

#### Bug fixes:
- make install problem. (#1289)
- When a class contains a union, other fields are not statically initialized. (#1286)
- Compiling DCD with -singleobj causes segmentation fault. (#1275)
- 0.17.0-beta2: Cannot build DCD. (#1266)
- Invalid bitcast error. (#1211)
- 0.16.0-beta1: Trivial program fails on FreeBSD. (#1119)
- Can't build gtk-d 3.1.4. (#1112)
-  x86 ABI: Fix Solaris regression and work around MSVC byval alignment issue. (#1230)
- Atomic RMW operations emit subpar x86 assembly. (#1195)
- align() not respected for local variable declarations. (#1154)
- Codegen optimizations are no longer disabled when `-g` is given. (75b3270a)
- Debug information is now generated for `ref` and `out` parameters. (#1177)
- `core.internal.convert` tests do not depend on `real` padding bytes any longer. (#788)

#### Building LDC:
- LDC now requires LLVM 3.5–3.8 and thus also a C++11-capable compiler to build.

#### Internals:
- The LDC-specific parts of the source code have received a big overhaul to make use of some C++11 features and to unify the style (the LLVM style as per `clang-format` is now used).
- The groundwork for a code generation test suite working on the LLVM IR level has been laid, together with some first test cases for alignment issues.
- LDC now emits more optional LLVM IR attributes for more optimization opportunities. (#1232)

#### Known issues:
- LDC does not zero the padding area of a real variable. This may lead to wrong results if the padding area is also considered. See #770. Does not apply to real members inside structs etc.
- Phobos does not compile on MinGW platform.
