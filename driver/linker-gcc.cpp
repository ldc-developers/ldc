//===-- linker-gcc.cpp ----------------------------------------------------===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "errors.h"
#include "driver/cl_options.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/exe_path.h"
#include "driver/tool.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

//////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<std::string>
    ltoLibrary("flto-binary", llvm::cl::ZeroOrMore,
               llvm::cl::desc("Set the linker LTO plugin library file (e.g. "
                              "LLVMgold.so (Unixes) or libLTO.dylib (Darwin))"),
               llvm::cl::value_desc("file"));

//////////////////////////////////////////////////////////////////////////////

namespace {

class ArgsBuilder {
public:
  std::vector<std::string> args;

  virtual ~ArgsBuilder() = default;

  void build(llvm::StringRef outputPath,
             llvm::cl::boolOrDefault fullyStaticFlag);

private:
  virtual void addSanitizers();
  virtual void addASanLinkFlags();

  virtual void addUserSwitches();
  void addDefaultLibs();
  virtual void addArch();

#if LDC_LLVM_VER >= 309
  void addLTOGoldPluginFlags();
  void addDarwinLTOFlags();
  void addLTOLinkFlags();
#endif

  virtual void addLdFlag(const llvm::Twine &flag) {
    args.push_back(("-Wl," + flag).str());
  }

  virtual void addLdFlag(const llvm::Twine &flag1, const llvm::Twine &flag2) {
    args.push_back(("-Wl," + flag1 + "," + flag2).str());
  }
};

//////////////////////////////////////////////////////////////////////////////
// LTO functionality

#if LDC_LLVM_VER >= 309

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

void ArgsBuilder::addLTOGoldPluginFlags() {
  addLdFlag("-plugin", getLTOGoldPluginPath());

  if (opts::isUsingThinLTO())
    addLdFlag("-plugin-opt=thinlto");

  if (!opts::mCPU.empty())
    addLdFlag(llvm::Twine("-plugin-opt=mcpu=") + opts::mCPU);

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
    args.push_back("-lto_library");
    args.push_back(std::move(dylibPath));
  }
}

/// Adds the required linker flags for LTO builds to args.
void ArgsBuilder::addLTOLinkFlags() {
  if (global.params.targetTriple->isOSLinux() ||
      global.params.targetTriple->isOSFreeBSD() ||
      global.params.targetTriple->isOSNetBSD() ||
      global.params.targetTriple->isOSOpenBSD() ||
      global.params.targetTriple->isOSDragonFly()) {
    // Assume that ld.gold or ld.bfd is used with plugin support.
    addLTOGoldPluginFlags();
  } else if (global.params.targetTriple->isOSDarwin()) {
    addDarwinLTOFlags();
  }
}

#endif // LDC_LLVM_VER >= 309

//////////////////////////////////////////////////////////////////////////////

// Returns true on success.
bool addDarwinASanLinkFlags(std::vector<std::string> &args) {
  std::string searchPaths[] = {
    exe_path::prependLibDir("libldc.asan_osx_dynamic.dylib"),
    exe_path::prependLibDir("libclang_rt.asan_osx_dynamic.dylib"),
  };

  for (const auto &filepath : searchPaths) {
    if (llvm::sys::fs::exists(filepath)) {
      args.push_back(filepath);

      // Add @executable_path to rpath to support having the dylib copied with
      // the executable.
      args.push_back("-rpath");
      args.push_back("@executable_path");

      // Add the path to the resource dir to rpath to support using the dylib
      // from the default location without copying.
      args.push_back("-rpath");
      args.push_back(llvm::sys::path::parent_path(filepath));

      return true;
    }
  }

  // We did not find the library.
  return false;
}

void ArgsBuilder::addASanLinkFlags() {
  bool success = false;
  if (global.params.targetTriple->isOSDarwin()) {
    success = addDarwinASanLinkFlags(args);
  } else {
    success = false; // TODO: implement
  }

  if (!success) {
    // Fallback, requires Clang. The asan library contains a versioned symbol
    // name and a linker error will happen when the LDC-LLVM and Clang-LLVM
    // versions don't match.
    args.push_back("-fsanitize=address");
  }
}

void ArgsBuilder::addSanitizers() {
  if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
    addASanLinkFlags();
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
                        llvm::cl::boolOrDefault fullyStaticFlag) {
  // object files
  for (auto objfile : *global.params.objfiles) {
    args.push_back(objfile);
  }

  // Link with profile-rt library when generating an instrumented binary.
  // profile-rt uses Phobos (MD5 hashing) and therefore must be passed on the
  // commandline before Phobos.
  if (global.params.genInstrProf) {
#if LDC_LLVM_VER >= 308
    if (global.params.targetTriple->isOSLinux()) {
      // For Linux, explicitly define __llvm_profile_runtime as undefined
      // symbol, so that the initialization part of profile-rt is linked in.
      addLdFlag("-u", llvm::getInstrProfRuntimeHookVarName());
    }
#endif
    args.push_back("-lldc-profile-rt");
  }

  // user libs
  for (auto libfile : *global.params.libfiles) {
    args.push_back(libfile);
  }

  if (global.params.dll) {
    args.push_back("-shared");
  }

  if (fullyStaticFlag == llvm::cl::BOU_TRUE) {
    args.push_back("-static");
  }

  args.push_back("-o");
  args.push_back(outputPath);

  addSanitizers();

#if LDC_LLVM_VER >= 309
  // Add LTO link flags before adding the user link switches, such that the user
  // can pass additional options to the LTO plugin.
  if (opts::isUsingLTO())
    addLTOLinkFlags();
#endif

  addUserSwitches();

  // libs added via pragma(lib, libname)
  for (auto ls : *global.params.linkswitches) {
    args.push_back(ls);
  }

  if (global.params.targetTriple->getOS() == llvm::Triple::Linux) {
    // Make sure we don't do --gc-sections when generating a profile-
    // instrumented binary. The runtime relies on magic sections, which
    // would be stripped by gc-section on older version of ld, see bug:
    // https://sourceware.org/bugzilla/show_bug.cgi?id=19161
    if (!opts::disableLinkerStripDead && !global.params.genInstrProf) {
      addLdFlag("--gc-sections");
    }
  }

  addDefaultLibs();

  addArch();
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

void ArgsBuilder::addDefaultLibs() {
  bool addSoname = false;

  switch (global.params.targetTriple->getOS()) {
  case llvm::Triple::Linux:
    addSoname = true;
    if (global.params.targetTriple->getEnvironment() == llvm::Triple::Android) {
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

  if (global.params.targetTriple->isWindowsGNUEnvironment()) {
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

void ArgsBuilder::addArch() {
  // Only specify -m32/-m64 for architectures where the two variants actually
  // exist (as e.g. the GCC ARM toolchain doesn't recognize the switches).
  // MIPS does not have -m32/-m64 but requires -mabi=.
  if (global.params.targetTriple->get64BitArchVariant().getArch() !=
          llvm::Triple::UnknownArch &&
      global.params.targetTriple->get32BitArchVariant().getArch() !=
          llvm::Triple::UnknownArch) {
    if (global.params.targetTriple->get64BitArchVariant().getArch() ==
            llvm::Triple::mips64 ||
        global.params.targetTriple->get64BitArchVariant().getArch() ==
            llvm::Triple::mips64el) {
      switch (getMipsABI()) {
      case MipsABI::EABI:
        args.push_back("-mabi=eabi");
        break;
      case MipsABI::O32:
        args.push_back("-mabi=32");
        break;
      case MipsABI::N32:
        args.push_back("-mabi=n32");
        break;
      case MipsABI::N64:
        args.push_back("-mabi=64");
        break;
      case MipsABI::Unknown:
        break;
      }
    } else {
      switch (global.params.targetTriple->getArch()) {
      case llvm::Triple::arm:
      case llvm::Triple::armeb:
      case llvm::Triple::aarch64:
      case llvm::Triple::aarch64_be:
#if LDC_LLVM_VER == 305
      case llvm::Triple::arm64:
      case llvm::Triple::arm64_be:
#endif
        break;
      default:
        if (global.params.is64bit) {
          args.push_back("-m64");
        } else {
          args.push_back("-m32");
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// (Yet unused) specialization for plain ld.

class LdArgsBuilder : public ArgsBuilder {
  void addSanitizers() override {}

  void addUserSwitches() override {
    if (!opts::ccSwitches.empty()) {
      warning(Loc(), "Ignoring -Xcc options");
    }

    args.insert(args.end(), opts::linkerSwitches.begin(),
                opts::linkerSwitches.end());
  }

  void addArch() override {}

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

int linkObjToBinaryGcc(llvm::StringRef outputPath, bool useInternalLinker,
                       llvm::cl::boolOrDefault fullyStaticFlag) {
  // find gcc for linking
  const std::string tool = getGcc();

  // build arguments
  ArgsBuilder argsBuilder;
  argsBuilder.build(outputPath, fullyStaticFlag);

  Logger::println("Linking with: ");
  Stream logstr = Logger::cout();
  for (const auto &arg : argsBuilder.args) {
    if (!arg.empty()) {
      logstr << "'" << arg << "' ";
    }
  }
  logstr << "\n"; // FIXME where's flush ?

  // try to call linker
  return executeToolAndWait(tool, argsBuilder.args, global.params.verbose);
}
