//===-- linker.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/linker.h"
#include "mars.h"
#include "module.h"
#include "root.h"
#include "driver/archiver.h"
#include "driver/cl_options.h"
#include "driver/exe_path.h"
#include "driver/tool.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include <algorithm>

//////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<bool>
    staticFlag("static", llvm::cl::ZeroOrMore,
               llvm::cl::desc("Create a statically linked binary, including "
                              "all system dependencies"));

static llvm::cl::opt<std::string>
    ltoLibrary("flto-binary", llvm::cl::ZeroOrMore,
               llvm::cl::desc("Set the linker LTO plugin library file (e.g. "
                              "LLVMgold.so (Unixes) or libLTO.dylib (Darwin))"),
               llvm::cl::value_desc("file"));

//////////////////////////////////////////////////////////////////////////////

static std::string getOutputName() {
  const auto &triple = *global.params.targetTriple;
  const bool sharedLib = global.params.dll;

  const char *extension = nullptr;
  if (sharedLib) {
    extension = global.dll_ext;
  } else if (triple.isOSWindows()) {
    extension = "exe";
  }

  if (global.params.exefile) {
    // DMD adds the default extension if there is none
    return opts::invokedByLDMD && extension
               ? FileName::defaultExt(global.params.exefile, extension)
               : global.params.exefile;
  }

  // Infer output name from first object file.
  std::string result = global.params.objfiles->dim
                           ? FileName::removeExt((*global.params.objfiles)[0])
                           : "a.out";

  if (sharedLib && !triple.isWindowsMSVCEnvironment())
    result = "lib" + result;

  if (global.params.run) {
    // If `-run` is passed, the executable is temporary and is removed
    // after execution. Make sure the name does not collide with other files
    // from other processes by creating a unique filename.
    llvm::SmallString<128> tempFilename;
    auto EC = llvm::sys::fs::createTemporaryFile(FileName::name(result.c_str()),
                                                 extension ? extension : "",
                                                 tempFilename);
    if (!EC)
      result = tempFilename.str();
  } else if (extension) {
    result += '.';
    result += extension;
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////
// LTO functionality

#if LDC_LLVM_VER >= 309

namespace {

void addLinkerFlag(std::vector<std::string> &args, const llvm::Twine &flag) {
  args.push_back("-Xlinker");
  args.push_back(flag.str());
}

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

void addLTOGoldPluginFlags(std::vector<std::string> &args) {
  addLinkerFlag(args, "-plugin");
  addLinkerFlag(args, getLTOGoldPluginPath());

  if (opts::isUsingThinLTO())
    addLinkerFlag(args, "-plugin-opt=thinlto");

  if (!opts::mCPU.empty())
    addLinkerFlag(args, llvm::Twine("-plugin-opt=mcpu=") + opts::mCPU);

  // Use the O-level passed to LDC as the O-level for LTO, but restrict it to
  // the [0, 3] range that can be passed to the linker plugin.
  static char optChars[15] = "-plugin-opt=O0";
  optChars[13] = '0' + std::min<char>(optLevel(), 3);
  addLinkerFlag(args, optChars);

#if LDC_LLVM_VER >= 400
  const llvm::TargetOptions &TO = gTargetMachine->Options;
  if (TO.FunctionSections)
    addLinkerFlag(args, "-plugin-opt=-function-sections");
  if (TO.DataSections)
    addLinkerFlag(args, "-plugin-opt=-data-sections");
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

void addDarwinLTOFlags(std::vector<std::string> &args) {
  std::string dylibPath = getLTOdylibPath();
  if (!dylibPath.empty()) {
    args.push_back("-lto_library");
    args.push_back(std::move(dylibPath));
  }
}

/// Adds the required linker flags for LTO builds to args.
void addLTOLinkFlags(std::vector<std::string> &args) {
  if (global.params.targetTriple->isOSLinux() ||
      global.params.targetTriple->isOSFreeBSD() ||
      global.params.targetTriple->isOSNetBSD() ||
      global.params.targetTriple->isOSOpenBSD() ||
      global.params.targetTriple->isOSDragonFly()) {
    // Assume that ld.gold or ld.bfd is used with plugin support.
    addLTOGoldPluginFlags(args);
  } else if (global.params.targetTriple->isOSDarwin()) {
    addDarwinLTOFlags(args);
  }
}
} // anonymous namespace

#endif // LDC_LLVM_VER >= 309

//////////////////////////////////////////////////////////////////////////////

namespace {

#if LDC_LLVM_VER >= 306
/// Insert an LLVM bitcode file into the module
void insertBitcodeIntoModule(const char *bcFile, llvm::Module &M,
                             llvm::LLVMContext &Context) {
  Logger::println("*** Linking-in bitcode file %s ***", bcFile);

  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> loadedModule(
      getLazyIRFileModule(bcFile, Err, Context));
  if (!loadedModule) {
    error(Loc(), "Error when loading LLVM bitcode file: %s", bcFile);
    fatal();
  }
#if LDC_LLVM_VER >= 308
  llvm::Linker(M).linkInModule(std::move(loadedModule));
#else
  llvm::Linker(&M).linkInModule(loadedModule.release());
#endif
}
#endif
}

/// Insert LLVM bitcode files into the module
void insertBitcodeFiles(llvm::Module &M, llvm::LLVMContext &Ctx,
                        Array<const char *> &bitcodeFiles) {
#if LDC_LLVM_VER >= 306
  for (const char *fname : bitcodeFiles) {
    insertBitcodeIntoModule(fname, M, Ctx);
  }
#else
  if (!bitcodeFiles.empty()) {
    error(Loc(),
          "Passing LLVM bitcode files to LDC is not supported for LLVM < 3.6");
    fatal();
  }
#endif
}

//////////////////////////////////////////////////////////////////////////////

static int linkObjToBinaryGcc(llvm::StringRef outputPath,
                              llvm::cl::boolOrDefault fullyStaticFlag) {
  // find gcc for linking
  const std::string tool = getGcc();

  // build arguments
  std::vector<std::string> args;

  // object files
  for (unsigned i = 0; i < global.params.objfiles->dim; i++)
    args.push_back((*global.params.objfiles)[i]);

  // Link with profile-rt library when generating an instrumented binary.
  // profile-rt uses Phobos (MD5 hashing) and therefore must be passed on the
  // commandline before Phobos.
  if (global.params.genInstrProf) {
#if LDC_LLVM_VER >= 308
    if (global.params.targetTriple->isOSLinux()) {
      // For Linux, explicitly define __llvm_profile_runtime as undefined
      // symbol, so that the initialization part of profile-rt is linked in.
      args.push_back(
          ("-Wl,-u," + llvm::getInstrProfRuntimeHookVarName()).str());
    }
#endif
    args.push_back("-lldc-profile-rt");
  }

  // user libs
  for (unsigned i = 0; i < global.params.libfiles->dim; i++)
    args.push_back((*global.params.libfiles)[i]);

  if (global.params.dll) {
    args.push_back("-shared");
  }

  if (fullyStaticFlag == llvm::cl::BOU_TRUE) {
    args.push_back("-static");
  }

  args.push_back("-o");
  args.push_back(outputPath);

  // Pass sanitizer arguments to linker. Requires clang.
  if (opts::sanitize == opts::AddressSanitizer) {
    args.push_back("-fsanitize=address");
  }

  if (opts::sanitize == opts::MemorySanitizer) {
    args.push_back("-fsanitize=memory");
  }

  if (opts::sanitize == opts::ThreadSanitizer) {
    args.push_back("-fsanitize=thread");
  }

#if LDC_LLVM_VER >= 309
  // Add LTO link flags before adding the user link switches, such that the user
  // can pass additional options to the LTO plugin.
  if (opts::isUsingLTO())
    addLTOLinkFlags(args);
#endif

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

  // libs added via pragma(lib, libname)
  for (unsigned i = 0; i < global.params.linkswitches->dim; i++) {
    args.push_back((*global.params.linkswitches)[i]);
  }

  // default libs
  bool addSoname = false;
  switch (global.params.targetTriple->getOS()) {
  case llvm::Triple::Linux:
    addSoname = true;
    // Make sure we don't do --gc-sections when generating a profile-
    // instrumented binary. The runtime relies on magic sections, which
    // would be stripped by gc-section on older version of ld, see bug:
    // https://sourceware.org/bugzilla/show_bug.cgi?id=19161
    if (!opts::disableLinkerStripDead && !global.params.genInstrProf) {
      args.push_back("-Wl,--gc-sections");
    }
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

  if (global.params.dll && addSoname) {
    std::string soname = opts::soname;
    if (!soname.empty()) {
      args.push_back("-Wl,-soname," + soname);
    }
  }

  Logger::println("Linking with: ");
  Stream logstr = Logger::cout();
  for (const auto &arg : args) {
    if (!arg.empty()) {
      logstr << "'" << arg << "'"
             << " ";
    }
  }
  logstr << "\n"; // FIXME where's flush ?

  // try to call linker
  return executeToolAndWait(tool, args, global.params.verbose);
}

//////////////////////////////////////////////////////////////////////////////

// path to the produced executable/shared library
static std::string gExePath;

//////////////////////////////////////////////////////////////////////////////

// linker-msvc.cpp
int linkObjToBinaryMSVC(llvm::StringRef outputPath,
                        llvm::cl::boolOrDefault fullyStaticFlag);

int linkObjToBinary() {
  Logger::println("*** Linking executable ***");

  // remember output path for later
  gExePath = getOutputName();

  createDirectoryForFileOrFail(gExePath);

  llvm::cl::boolOrDefault fullyStaticFlag = llvm::cl::BOU_UNSET;
  if (staticFlag.getNumOccurrences() != 0) {
    fullyStaticFlag = staticFlag ? llvm::cl::BOU_TRUE : llvm::cl::BOU_FALSE;
  }

  if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
    return linkObjToBinaryMSVC(gExePath, fullyStaticFlag);
  }

  return linkObjToBinaryGcc(gExePath, fullyStaticFlag);
}

//////////////////////////////////////////////////////////////////////////////

void deleteExeFile() {
  if (!gExePath.empty() && !llvm::sys::fs::is_directory(gExePath)) {
    llvm::sys::fs::remove(gExePath);
  }
}

//////////////////////////////////////////////////////////////////////////////

int runProgram() {
  assert(!gExePath.empty());

  // Run executable
  int status =
      executeToolAndWait(gExePath, opts::runargs, global.params.verbose);
  if (status < 0) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    error(Loc(), "program received signal %d", -status);
#else
    error(Loc(), "program received signal %d (%s)", -status,
          strsignal(-status));
#endif
    return -status;
  }
  return status;
}
