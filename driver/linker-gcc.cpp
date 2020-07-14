//===-- linker-gcc.cpp ----------------------------------------------------===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/errors.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/configfile.h"
#include "driver/exe_path.h"
#include "driver/ldc-version.h"
#include "driver/linker.h"
#include "driver/tool.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#if LDC_WITH_LLD && LDC_LLVM_VER >= 600
#include "lld/Common/Driver.h"
#endif

//////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<std::string>
    ltoLibrary("flto-binary", llvm::cl::ZeroOrMore,
               llvm::cl::desc("Set the linker LTO plugin library file (e.g. "
                              "LLVMgold.so (Unixes) or libLTO.dylib (Darwin))"),
               llvm::cl::value_desc("file"));

static llvm::cl::opt<bool> linkNoCpp(
    "link-no-cpp", llvm::cl::ZeroOrMore, llvm::cl::Hidden,
    llvm::cl::desc("Disable automatic linking with the C++ standard library."));

//////////////////////////////////////////////////////////////////////////////

namespace {

class ArgsBuilder {
public:
  std::vector<std::string> args;

  virtual ~ArgsBuilder() = default;

  void build(llvm::StringRef outputPath,
             const std::vector<std::string> &defaultLibNames);

private:
  virtual void addSanitizers(const llvm::Triple &triple);
  virtual void addASanLinkFlags(const llvm::Triple &triple);
  virtual void addFuzzLinkFlags(const llvm::Triple &triple);
  virtual void addCppStdlibLinkFlags(const llvm::Triple &triple);
  virtual void addProfileRuntimeLinkFlags(const llvm::Triple &triple);
  virtual void addXRayLinkFlags(const llvm::Triple &triple);
  virtual bool addCompilerRTArchiveLinkFlags(llvm::StringRef baseName,
                                             const llvm::Triple &triple);

  virtual void addLinker();
  virtual void addUserSwitches();
  void addDefaultPlatformLibs();
  virtual void addTargetFlags();

  void addLTOGoldPluginFlags(bool requirePlugin);
  void addDarwinLTOFlags();
  void addLTOLinkFlags();
  bool isLldDefaultLinker();

  virtual void addLdFlag(const llvm::Twine &flag) {
    args.push_back(("-Wl," + flag).str());
  }

  virtual void addLdFlag(const llvm::Twine &flag1, const llvm::Twine &flag2) {
    args.push_back(("-Wl," + flag1 + "," + flag2).str());
  }
};

//////////////////////////////////////////////////////////////////////////////
// LTO functionality

std::string getLTOGoldPluginPath() {
  if (!ltoLibrary.empty()) {
    if (llvm::sys::fs::exists(ltoLibrary))
      return ltoLibrary;

    error(Loc(), "-flto-binary: file '%s' not found", ltoLibrary.c_str());
    fatal();
  } else {
    std::string searchPaths[] = {
      // The plugin packaged with LDC has a "-ldc" suffix.
      exe_path::prependLibDir("LLVMgold-ldc.so"),
      // Perhaps the user copied the plugin to LDC's lib dir.
      exe_path::prependLibDir("LLVMgold.so"),
#if __LP64__
      "/usr/local/lib64/LLVMgold.so",
#endif
      "/usr/local/lib/LLVMgold.so",
#if __LP64__
      "/usr/lib64/LLVMgold.so",
#endif
      "/usr/lib/LLVMgold.so",
      "/usr/lib/bfd-plugins/LLVMgold.so",
    };

    // Try all searchPaths and early return upon the first path found.
    for (const auto &p : searchPaths) {
      if (llvm::sys::fs::exists(p))
        return p;
    }

    error(Loc(), "The LLVMgold.so plugin (needed for LTO) was not found. You "
                 "can specify its path with -flto-binary=<file>.");
    fatal();
  }
}

void ArgsBuilder::addLTOGoldPluginFlags(bool requirePlugin) {
  if (requirePlugin)
    addLdFlag("-plugin", getLTOGoldPluginPath());

  if (opts::isUsingThinLTO())
    addLdFlag("-plugin-opt=thinlto");

  const auto cpu = gTargetMachine->getTargetCPU();
  if (!cpu.empty())
    addLdFlag(llvm::Twine("-plugin-opt=mcpu=") + cpu);

  // Use the O-level passed to LDC as the O-level for LTO, but restrict it to
  // the [0, 3] range that can be passed to the linker plugin.
  static char optChars[15] = "-plugin-opt=O0";
  optChars[13] = '0' + std::min<char>(optLevel(), 3);
  addLdFlag(optChars);

#if LDC_LLVM_VER >= 400
  const llvm::TargetOptions &TO = gTargetMachine->Options;
  if (TO.FunctionSections)
    addLdFlag("-plugin-opt=-function-sections");
  if (TO.DataSections)
    addLdFlag("-plugin-opt=-data-sections");
#endif
}

// Returns an empty string when libLTO.dylib was not specified nor found.
std::string getLTOdylibPath() {
  if (!ltoLibrary.empty()) {
    if (llvm::sys::fs::exists(ltoLibrary))
      return ltoLibrary;

    error(Loc(), "-flto-binary: '%s' not found", ltoLibrary.c_str());
    fatal();
  } else {
    // The plugin packaged with LDC has a "-ldc" suffix.
    std::string searchPath = exe_path::prependLibDir("libLTO-ldc.dylib");
    if (llvm::sys::fs::exists(searchPath))
      return searchPath;

    return "";
  }
}

void ArgsBuilder::addDarwinLTOFlags() {
  std::string dylibPath = getLTOdylibPath();
  if (!dylibPath.empty()) {
    addLdFlag("-lto_library", dylibPath);
  }
}

/// Adds the required linker flags for LTO builds to args.
void ArgsBuilder::addLTOLinkFlags() {
  if (global.params.targetTriple->isOSLinux() ||
      global.params.targetTriple->isOSFreeBSD() ||
      global.params.targetTriple->isOSNetBSD() ||
      global.params.targetTriple->isOSOpenBSD() ||
      global.params.targetTriple->isOSDragonFly()) {
    // LLD supports LLVM LTO natively, do not add the plugin itself.
    // Otherwise, assume that ld.gold or ld.bfd is used with plugin support.
    bool isLld = opts::linker == "lld" || useInternalLLDForLinking() ||
                 (opts::linker.empty() && isLldDefaultLinker());
    addLTOGoldPluginFlags(!isLld);
  } else if (global.params.targetTriple->isOSDarwin()) {
    addDarwinLTOFlags();
  }
}

bool ArgsBuilder::isLldDefaultLinker() {
  auto triple = global.params.targetTriple;
  if (triple->isOSFreeBSD()) {
    if (triple->getOSMajorVersion() >= 12 &&
        triple->getArch() == llvm::Triple::ArchType::x86_64)
      return true;
    if (triple->getArch() == llvm::Triple::ArchType::aarch64)
      return true;
  }
  return false;
}

//////////////////////////////////////////////////////////////////////////////

// Returns the arch name as used in the compiler_rt libs.
// FIXME: implement correctly for non-x86 platforms (e.g. ARM)
// See clang/lib/Driver/Toolchain.cpp.
llvm::StringRef getCompilerRTArchName(const llvm::Triple &triple) {
  return triple.getArchName();
}

// Appends arch suffix and extension.
// E.g., for name="libclang_rt.fuzzer" and sharedLibrary=false, returns
// "libclang_rt.fuzzer_osx.a" on Darwin.
std::string getCompilerRTLibFilename(const llvm::Twine &name,
                                     const llvm::Triple &triple,
                                     bool sharedLibrary) {
  return (triple.isOSDarwin()
              ? name + (sharedLibrary ? "_osx_dynamic.dylib" : "_osx.a")
              : name + "-" + getCompilerRTArchName(triple) +
                    (sharedLibrary ? ".so" : ".a"))
      .str();
}

// Clang's RT libs are in a subdir of the lib dir.
// E.g., for name="libclang_rt.asan" and sharedLibrary=true, returns
// "clang/6.0.0/lib/darwin/libclang_rt.asan_osx_dynamic.dylib" on
// Darwin.
// This function is "best effort", the path may not be what Clang does...
// See clang/lib/Driver/Toolchain.cpp.
std::string getRelativeClangCompilerRTLibPath(const llvm::Twine &name,
                                              const llvm::Triple &triple,
                                              bool sharedLibrary) {
  llvm::StringRef OSName =
      triple.isOSDarwin()
          ? "darwin"
          : triple.isOSFreeBSD() ? "freebsd" : triple.getOSName();

  std::string relPath = (llvm::Twine("clang/") + ldc::llvm_version_base +
                         "/lib/" + OSName + "/" + name)
                            .str();

  return getCompilerRTLibFilename(relPath, triple, sharedLibrary);
}

void appendFullLibPathCandidates(std::vector<std::string> &paths,
                                 const llvm::Twine &filename) {
  for (const char *dir : ConfigFile::instance.libDirs()) {
    llvm::SmallString<128> candidate(dir);
    llvm::sys::path::append(candidate, filename);
    paths.emplace_back(candidate.data(), candidate.size());
  }

  // for backwards compatibility
  paths.push_back(exe_path::prependLibDir(filename));
}

// Returns candidates of full paths to a compiler-rt lib.
// E.g., for baseName="asan" and sharedLibrary=false, returns something like
// [ "<libDir>/libldc_rt.asan.a",
//   "<libDir>/libclang_rt.asan_osx.a",
//   "<libDir>/clang/6.0.0/lib/darwin/libclang_rt.asan_osx.a" ].
std::vector<std::string>
getFullCompilerRTLibPathCandidates(llvm::StringRef baseName,
                                   const llvm::Triple &triple,
                                   bool sharedLibrary = false) {
  std::vector<std::string> r;
  const auto ldcRT =
      ("libldc_rt." + baseName +
       (!sharedLibrary ? ".a" : triple.isOSDarwin() ? ".dylib" : ".so"))
          .str();
  appendFullLibPathCandidates(r, ldcRT);
  const auto clangRT = getCompilerRTLibFilename("libclang_rt." + baseName,
                                                triple, sharedLibrary);
  appendFullLibPathCandidates(r, clangRT);
  const auto fullClangRT = getRelativeClangCompilerRTLibPath(
      "libclang_rt." + baseName, triple, sharedLibrary);
  appendFullLibPathCandidates(r, fullClangRT);
  return r;
}

void ArgsBuilder::addASanLinkFlags(const llvm::Triple &triple) {
  // Examples: "libclang_rt.asan-x86_64.a" or "libclang_rt.asan-arm.a" and
  // "libclang_rt.asan-x86_64.so"

  // TODO: let user choose to link with shared lib.
  // In case of shared ASan, I think we also need to statically link with
  // libclang_rt.asan-preinit-<arch>.a on Linux. On Darwin, the only option is
  // to use the shared library.
  bool linkSharedASan = triple.isOSDarwin();
  const auto searchPaths =
      getFullCompilerRTLibPathCandidates("asan", triple, linkSharedASan);

  for (const auto &filepath : searchPaths) {
    IF_LOG Logger::println("Searching ASan lib: %s", filepath.c_str());

    if (llvm::sys::fs::exists(filepath) &&
        !llvm::sys::fs::is_directory(filepath)) {
      IF_LOG Logger::println("Found, linking with %s", filepath.c_str());
      args.push_back(filepath);

      if (linkSharedASan) {
        // Add @executable_path to rpath to support having the shared lib copied
        // with the executable.
        args.push_back("-rpath");
        args.push_back("@executable_path");

        // Add the path to the resource dir to rpath to support using the shared
        // lib from the default location without copying.
        args.push_back("-rpath");
        args.push_back(std::string(llvm::sys::path::parent_path(filepath)));
      }

      return;
    }
  }

  // When we reach here, we did not find the ASan library.
  // Fallback, requires Clang. The asan library contains a versioned symbol
  // name and a linker error will happen when the LDC-LLVM and Clang-LLVM
  // versions don't match.
  args.push_back("-fsanitize=address");
}

// Adds all required link flags for -fsanitize=fuzzer when libFuzzer library is
// found.
void ArgsBuilder::addFuzzLinkFlags(const llvm::Triple &triple) {
#if LDC_LLVM_VER >= 600
  const auto searchPaths = getFullCompilerRTLibPathCandidates("fuzzer", triple);
#else
  std::vector<std::string> searchPaths;
  appendFullLibPathCandidates(searchPaths, "libFuzzer.a");
  appendFullLibPathCandidates(searchPaths, "libLLVMFuzzer.a");
#endif

  for (const auto &filepath : searchPaths) {
    IF_LOG Logger::println("Searching libFuzzer: %s", filepath.c_str());

    if (llvm::sys::fs::exists(filepath) &&
        !llvm::sys::fs::is_directory(filepath)) {
      IF_LOG Logger::println("Found, linking with %s", filepath.c_str());
      args.push_back(filepath);

      // libFuzzer requires the C++ std library, but only add the link flags
      // when libFuzzer was found.
      addCppStdlibLinkFlags(triple);
      return;
    }
  }
}

// Adds all required link flags for -fxray-instrument when the xray library is
// found.
void ArgsBuilder::addXRayLinkFlags(const llvm::Triple &triple) {
  if (!triple.isOSLinux())
    warning(Loc(), "XRay may not be fully supported on non-Linux target OS.");

  bool libraryFoundAndLinked = addCompilerRTArchiveLinkFlags("xray", triple);
#if LDC_LLVM_VER >= 700
  // Since LLVM 7, each XRay mode was split into its own library.
  if (libraryFoundAndLinked) {
    addCompilerRTArchiveLinkFlags("xray-basic", triple);
    addCompilerRTArchiveLinkFlags("xray-fdr", triple);
  }
#else
  // Before LLVM 7, XRay requires the C++ std library (but not on Darwin).
  // Only link with the C++ stdlib when the XRay library was found.
  if (libraryFoundAndLinked && !triple.isOSDarwin())
    addCppStdlibLinkFlags(triple);
#endif
}

// Returns true if library was found and added to link flags.
bool ArgsBuilder::addCompilerRTArchiveLinkFlags(llvm::StringRef baseName,
                                                const llvm::Triple &triple) {
  const bool linkerDarwin = triple.isOSDarwin();
  const auto searchPaths = getFullCompilerRTLibPathCandidates(baseName, triple);
  for (const auto &filepath : searchPaths) {
    IF_LOG Logger::println("Searching runtime library: %s", filepath.c_str());

    if (llvm::sys::fs::exists(filepath) &&
        !llvm::sys::fs::is_directory(filepath)) {
      if (!linkerDarwin)
        addLdFlag("--whole-archive");

      IF_LOG Logger::println("Found, linking with %s", filepath.c_str());
      args.push_back(filepath);

      if (!linkerDarwin)
        addLdFlag("--no-whole-archive");

      return true;
    }
  }

  return false;
}

void ArgsBuilder::addCppStdlibLinkFlags(const llvm::Triple &triple) {
  if (linkNoCpp)
    return;

  switch (triple.getOS()) {
  case llvm::Triple::Linux:
    if (triple.getEnvironment() == llvm::Triple::Android) {
      args.push_back("-lc++");
    } else {
      args.push_back("-lstdc++");
    }
    break;
  case llvm::Triple::Solaris:
  case llvm::Triple::NetBSD:
  case llvm::Triple::OpenBSD:
  case llvm::Triple::DragonFly:
    args.push_back("-lstdc++");
    break;
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
  case llvm::Triple::IOS:
  case llvm::Triple::WatchOS:
  case llvm::Triple::TvOS:
  case llvm::Triple::FreeBSD:
    args.push_back("-lc++");
    break;
  default:
    // Don't know: do nothing so the user can step in
    break;
  }
}

// Adds all required link flags for PGO.
void ArgsBuilder::addProfileRuntimeLinkFlags(const llvm::Triple &triple) {
  const auto searchPaths =
      getFullCompilerRTLibPathCandidates("profile", triple);

  if (global.params.targetTriple->isOSLinux()) {
    // For Linux, explicitly define __llvm_profile_runtime as undefined
    // symbol, so that the initialization part of profile-rt is linked in.
    addLdFlag("-u", llvm::getInstrProfRuntimeHookVarName());
  }

  for (const auto &filepath : searchPaths) {
    IF_LOG Logger::println("Searching profile runtime: %s", filepath.c_str());

    if (llvm::sys::fs::exists(filepath)) {
      IF_LOG Logger::println("Found, linking with %s", filepath.c_str());
      args.push_back(filepath);
      return;
    }
  }
}

void ArgsBuilder::addSanitizers(const llvm::Triple &triple) {
  if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
    addASanLinkFlags(triple);
  }

  if (opts::isSanitizerEnabled(opts::FuzzSanitizer)) {
    addFuzzLinkFlags(triple);
  }

  // TODO: instead of this, we should link with our own sanitizer libraries
  // because LDC's LLVM version could be different from the system clang.
  if (opts::isSanitizerEnabled(opts::MemorySanitizer)) {
    args.push_back("-fsanitize=memory");
  }

  // TODO: instead of this, we should link with our own sanitizer libraries
  // because LDC's LLVM version could be different from the system clang.
  if (opts::isSanitizerEnabled(opts::ThreadSanitizer)) {
    args.push_back("-fsanitize=thread");
  }
}

//////////////////////////////////////////////////////////////////////////////

void ArgsBuilder::build(llvm::StringRef outputPath,
                        const std::vector<std::string> &defaultLibNames) {
  // object files
  for (auto objfile : global.params.objfiles) {
    args.push_back(objfile);
  }

  // Link with profile-rt library when generating an instrumented binary.
  if (opts::isInstrumentingForPGO()) {
    addProfileRuntimeLinkFlags(*global.params.targetTriple);
  }

  if (opts::enableDynamicCompile) {
    args.push_back("-lldc-jit-rt");
    args.push_back("-lldc-jit");
  }

  // user libs
  for (auto libfile : global.params.libfiles) {
    args.push_back(libfile);
  }
  for (auto dllfile : global.params.dllfiles) {
    args.push_back(dllfile);
  }

  if (global.params.dll) {
    args.push_back("-shared");
  }

  if (linkFullyStatic() == llvm::cl::BOU_TRUE) {
    args.push_back("-static");
  }

  args.push_back("-o");
  args.push_back(std::string(outputPath));

  addSanitizers(*global.params.targetTriple);

  if (opts::fXRayInstrument) {
    addXRayLinkFlags(*global.params.targetTriple);
  }

  // Add LTO link flags before adding the user link switches, such that the user
  // can pass additional options to the LTO plugin.
  if (opts::isUsingLTO())
    addLTOLinkFlags();

  addLinker();
  addUserSwitches();

  // lib dirs
  for (const char *dir_c : ConfigFile::instance.libDirs()) {
    const llvm::StringRef dir(dir_c);
    if (!dir.empty())
      args.push_back(("-L" + dir).str());
  }

  // default libs
  for (const auto &name : defaultLibNames) {
    args.push_back("-l" + name);
  }

  // libs added via pragma(lib, libname)
  for (auto ls : global.params.linkswitches) {
    args.push_back(ls);
  }

  // -rpath if linking against shared default libs or ldc-jit
  if (linkAgainstSharedDefaultLibs() || opts::enableDynamicCompile) {
    llvm::StringRef rpath = ConfigFile::instance.rpath();
    if (!rpath.empty())
      addLdFlag("-rpath", rpath);
  }

  if (global.params.targetTriple->getOS() == llvm::Triple::Linux ||
      (global.params.targetTriple->getOS() == llvm::Triple::FreeBSD &&
       (useInternalLLDForLinking() ||
        (!opts::linker.empty() && opts::linker != "bfd") ||
        (opts::linker.empty() && isLldDefaultLinker())))) {
    // Make sure we don't do --gc-sections when generating a profile-
    // instrumented binary. The runtime relies on magic sections, which
    // would be stripped by gc-section on older version of ld, see bug:
    // https://sourceware.org/bugzilla/show_bug.cgi?id=19161
    if (!opts::disableLinkerStripDead && !opts::isInstrumentingForPGO()) {
      addLdFlag("--gc-sections");
    }
  }

  addDefaultPlatformLibs();

  addTargetFlags();
}

//////////////////////////////////////////////////////////////////////////////

void ArgsBuilder::addLinker() {
  llvm::StringRef linker = opts::linker;

  // We have a default linker preference for Linux targets. It can be disabled
  // via `-linker=` (explicitly empty).
  if (global.params.isLinux && opts::linker.getNumOccurrences() == 0) {
    // Default to ld.bfd for Android (placing .tdata and .tbss sections adjacent
    // to each other as required by druntime's rt.sections_android, contrary to
    // gold and lld as of Android NDK r21d).
    if (global.params.targetTriple->getEnvironment() == llvm::Triple::Android) {
      linker = "bfd";
    }
    // Otherwise default to ld.gold for Linux due to ld.bfd issues with ThinLTO
    // (see #2278) and older bfd versions stripping llvm.used symbols (e.g.,
    // ModuleInfo refs) with --gc-sections (see #2870).
    else {
      linker = "gold";
    }
  }

  if (!linker.empty())
    args.push_back(("-fuse-ld=" + linker).str());
}

//////////////////////////////////////////////////////////////////////////////

void ArgsBuilder::addUserSwitches() {
  // additional linker and cc switches (preserve order across both lists)
  for (unsigned ilink = 0, icc = 0;;) {
    unsigned linkpos = ilink < opts::linkerSwitches.size()
                           ? opts::linkerSwitches.getPosition(ilink)
                           : std::numeric_limits<unsigned>::max();
    unsigned ccpos = icc < opts::ccSwitches.size()
                         ? opts::ccSwitches.getPosition(icc)
                         : std::numeric_limits<unsigned>::max();
    if (linkpos < ccpos) {
      const std::string &p = opts::linkerSwitches[ilink++];
      // Don't push -l and -L switches using -Xlinker, but pass them indirectly
      // via GCC. This makes sure user-defined paths take precedence over
      // GCC's builtin LIBRARY_PATHs.
      // Options starting with `-Wl,`, -shared or -static are not handled by
      // the linker and must be passed to the driver.
      auto str = llvm::StringRef(p);
      if (!(str.startswith("-l") || str.startswith("-L") ||
            str.startswith("-Wl,") || str.startswith("-shared") ||
            str.startswith("-static"))) {
        args.push_back("-Xlinker");
      }
      args.push_back(p);
    } else if (ccpos < linkpos) {
      args.push_back(opts::ccSwitches[icc++]);
    } else {
      break;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////

void ArgsBuilder::addDefaultPlatformLibs() {
  bool addSoname = false;

  const auto &triple = *global.params.targetTriple;

  switch (triple.getOS()) {
  case llvm::Triple::Linux:
    addSoname = true;
    if (triple.getEnvironment() == llvm::Triple::Android) {
      args.push_back("-ldl");
      args.push_back("-lm");
      break;
    }
    args.push_back("-lrt");
  // fallthrough
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
    args.push_back("-ldl");
  // fallthrough
  case llvm::Triple::FreeBSD:
  case llvm::Triple::NetBSD:
  case llvm::Triple::OpenBSD:
  case llvm::Triple::DragonFly:
    addSoname = true;
    args.push_back("-lpthread");
    args.push_back("-lm");
    break;

  case llvm::Triple::Solaris:
    args.push_back("-lm");
    args.push_back("-lumem");
    args.push_back("-lsocket");
    args.push_back("-lnsl");
    break;

  default:
    // OS not yet handled, will probably lead to linker errors.
    // FIXME: Win32.
    break;
  }

  if (triple.isWindowsGNUEnvironment()) {
    // This is really more of a kludge, as linking in the Winsock functions
    // should be handled by the pragma(lib, ...) in std.socket, but it
    // makes LDC behave as expected for now.
    args.push_back("-lws2_32");
  }

  if (global.params.dll && addSoname && !opts::soname.empty()) {
    addLdFlag("-soname", opts::soname);
  }
}

//////////////////////////////////////////////////////////////////////////////

void ArgsBuilder::addTargetFlags() { appendTargetArgsForGcc(args); }

//////////////////////////////////////////////////////////////////////////////
// Specialization for plain ld.

class LdArgsBuilder : public ArgsBuilder {
  void addSanitizers(const llvm::Triple &triple) override {}

  void addLinker() override {}

  void addUserSwitches() override {
    if (!opts::ccSwitches.empty()) {
      warning(Loc(), "Ignoring -Xcc options");
    }

    args.insert(args.end(), opts::linkerSwitches.begin(),
                opts::linkerSwitches.end());
  }

  void addTargetFlags() override {}

  void addLdFlag(const llvm::Twine &flag) override {
    args.push_back(flag.str());
  }

  void addLdFlag(const llvm::Twine &flag1, const llvm::Twine &flag2) override {
    args.push_back(flag1.str());
    args.push_back(flag2.str());
  }
};

} // anonymous namespace

//////////////////////////////////////////////////////////////////////////////

int linkObjToBinaryGcc(llvm::StringRef outputPath,
                       const std::vector<std::string> &defaultLibNames) {
#if LDC_WITH_LLD && LDC_LLVM_VER >= 600
  if (useInternalLLDForLinking()) {
    LdArgsBuilder argsBuilder;
    argsBuilder.build(outputPath, defaultLibNames);

    const auto fullArgs =
        getFullArgs("lld", argsBuilder.args, global.params.verbose);

    // CanExitEarly == true means that LLD can and will call `exit()` when
    // errors occur.
    bool CanExitEarly = false;

    bool success = false;
    if (global.params.targetTriple->isOSBinFormatELF()) {
      success = lld::elf::link(fullArgs, CanExitEarly
#if LDC_LLVM_VER >= 1000
                               ,
                               llvm::outs(), llvm::errs()
#endif
      );
    } else if (global.params.targetTriple->isOSBinFormatMachO()) {
      success = lld::mach_o::link(fullArgs
#if LDC_LLVM_VER >= 700
                                  ,
                                  CanExitEarly
#if LDC_LLVM_VER >= 1000
                                  ,
                                  llvm::outs(), llvm::errs()
#endif
#endif
      );
    } else if (global.params.targetTriple->isOSBinFormatCOFF()) {
      success = lld::mingw::link(fullArgs
#if LDC_LLVM_VER >= 1000
                                 ,
                                 CanExitEarly, llvm::outs(), llvm::errs()
#endif
      );
    } else if (global.params.targetTriple->isOSBinFormatWasm()) {
#if __linux__ && LDC_LLVM_VER >= 700
      // FIXME: segfault in cleanup (`freeArena()`) after successful linking,
      //        but only on Linux?
      CanExitEarly = true;
#endif
      success = lld::wasm::link(fullArgs, CanExitEarly
#if LDC_LLVM_VER >= 1000
                                ,
                                llvm::outs(), llvm::errs()
#endif
      );
    } else {
      error(Loc(), "unknown target binary format for internal linking");
    }

    if (!success)
      error(Loc(), "linking with LLD failed");

    return success ? 0 : 1;
  }
#endif

  // build command-line for gcc-compatible linker driver
  // exception: invoke (ld-compatible) linker directly for WebAssembly targets
  std::string tool;
  std::unique_ptr<ArgsBuilder> argsBuilder;
#if LDC_LLVM_VER >= 500
  if (global.params.targetTriple->isOSBinFormatWasm()) {
    tool = getProgram("wasm-ld", &opts::linker);
    argsBuilder = llvm::make_unique<LdArgsBuilder>();
  } else {
#endif
    tool = getGcc();
    argsBuilder = llvm::make_unique<ArgsBuilder>();
#if LDC_LLVM_VER >= 500
  }
#endif

  // build arguments
  argsBuilder->build(outputPath, defaultLibNames);

  Logger::println("Linking with: ");
  Stream logstr = Logger::cout();
  for (const auto &arg : argsBuilder->args) {
    if (!arg.empty()) {
      logstr << "'" << arg << "' ";
    }
  }
  logstr << "\n"; // FIXME where's flush ?

  // try to call linker
  return executeToolAndWait(tool, argsBuilder->args, global.params.verbose);
}
