//===-- linker.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/linker.h"

#include "dmd/errors.h"
#include "dmd/target.h"
#include "driver/cl_options.h"
#include "driver/timetrace.h"
#include "driver/tool.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include <sstream>

namespace cl = llvm::cl;

//////////////////////////////////////////////////////////////////////////////

#if LDC_WITH_LLD
static cl::opt<bool> linkInternally("link-internally", cl::ZeroOrMore,
                                    cl::desc("Use internal LLD for linking"),
                                    cl::cat(opts::linkingCategory));
#else
constexpr bool linkInternally = false;
#endif

static cl::opt<std::string> platformLib(
    "platformlib", cl::ZeroOrMore, cl::value_desc("lib1,lib2,..."),
    cl::desc("Platform libraries to link with (overrides previous)"),
    cl::cat(opts::linkingCategory));

static cl::opt<bool> noDefaultLib(
    "nodefaultlib", cl::ZeroOrMore, cl::Hidden,
    cl::desc("Don't add a default library for linking implicitly"));

static cl::opt<std::string>
    defaultLib("defaultlib", cl::ZeroOrMore, cl::value_desc("lib1,lib2,..."),
               cl::desc("Default libraries to link with (overrides previous)"),
               cl::cat(opts::linkingCategory));

static cl::opt<std::string> debugLib(
    "debuglib", cl::ZeroOrMore, cl::Hidden, cl::value_desc("lib1,lib2,..."),
    cl::desc("Debug versions of default libraries (overrides previous). If the "
             "option is omitted, LDC will append -debug to the -defaultlib "
             "names when linking with -link-defaultlib-debug"),
    cl::cat(opts::linkingCategory));

static cl::opt<bool> linkDefaultLibDebug(
    "link-defaultlib-debug", cl::ZeroOrMore,
    cl::desc("Link with debug versions of default libraries"),
    cl::cat(opts::linkingCategory));
static cl::alias _linkDebugLib("link-debuglib", cl::Hidden,
                               cl::aliasopt(linkDefaultLibDebug),
                               cl::desc("Alias for -link-defaultlib-debug"),
                               cl::cat(opts::linkingCategory));

static cl::opt<cl::boolOrDefault> linkDefaultLibShared(
    "link-defaultlib-shared", cl::ZeroOrMore,
    cl::desc("Link with shared versions of default libraries. Defaults to true "
             "when generating a shared library (-shared)."),
    cl::cat(opts::linkingCategory));

static cl::opt<cl::boolOrDefault>
    staticFlag("static", cl::ZeroOrMore,
               cl::desc("Create a statically linked binary, including "
                        "all system dependencies"),
               cl::cat(opts::linkingCategory));

static llvm::cl::opt<std::string>
    mscrtlib("mscrtlib", llvm::cl::ZeroOrMore,
             llvm::cl::desc("MS C runtime library to link with"),
             llvm::cl::value_desc("libcmt[d]|msvcrt[d]"),
             llvm::cl::cat(opts::linkingCategory));

//////////////////////////////////////////////////////////////////////////////

// linker-gcc.cpp
int linkObjToBinaryGcc(llvm::StringRef outputPath,
                       const std::vector<std::string> &defaultLibNames);

// linker-msvc.cpp
int linkObjToBinaryMSVC(llvm::StringRef outputPath,
                        const std::vector<std::string> &defaultLibNames);

//////////////////////////////////////////////////////////////////////////////

static std::string getOutputPath() {
  const auto &triple = *global.params.targetTriple;
  const bool sharedLib = global.params.dll;

  const char *extension = nullptr;
  if (sharedLib) {
    extension = target.dll_ext.ptr;
  } else if (triple.isOSWindows()) {
    extension = "exe";
  } else if (triple.getArch() == llvm::Triple::wasm32 ||
             triple.getArch() == llvm::Triple::wasm64) {
    extension = "wasm";
  }

  if (global.params.exefile.length) {
    // DMD adds the default extension if there is none
    return opts::invokedByLDMD && extension
               ? FileName::defaultExt(global.params.exefile.ptr, extension)
               : global.params.exefile.ptr;
  }

  // Infer output name from first object file.
  std::string result =
      global.params.objfiles.length
          ? FileName::removeExt(FileName::name(global.params.objfiles[0]))
          : "a.out";

  if (sharedLib && !triple.isWindowsMSVCEnvironment())
    result = "lib" + result;

  if (global.params.run) {
    // If `-run` is passed, the executable is temporary and is removed
    // after execution. Make sure the name does not collide with other files
    // from other processes by creating a unique filename.
    llvm::SmallString<128> tempFilename;
    auto EC = llvm::sys::fs::createTemporaryFile(
        result, extension ? extension : "", tempFilename);
    if (!EC)
      result = {tempFilename.data(), tempFilename.size()};
  } else if (extension) {
    result += '.';
    result += extension;
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////

static std::vector<std::string>
parseLibNames(llvm::StringRef commaSeparatedList, llvm::StringRef suffix = {}) {
  std::vector<std::string> result;

  std::stringstream list(commaSeparatedList.str());
  while (list.good()) {
    std::string lib;
    std::getline(list, lib, ',');
    if (lib.empty()) {
      continue;
    }

    result.push_back(suffix.empty() ? std::move(lib) : (lib + suffix).str());
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////

static std::vector<std::string> getDefaultLibNames() {
  std::vector<std::string> result;

  if (noDefaultLib) {
    deprecation(Loc(), "-nodefaultlib is deprecated, as -defaultlib now "
                       "overrides the existing list instead of appending to "
                       "it. Please use the latter instead.");
  } else if (!global.params.betterC) {
    llvm::StringRef list = defaultLib;
    std::string suffix;

    if (linkDefaultLibDebug) {
      if (debugLib.getNumOccurrences() == 0)
        suffix = "-debug";
      else
        list = debugLib;
    }
    if (linkAgainstSharedDefaultLibs()) {
      suffix += "-shared";
    }

    result = parseLibNames(list, suffix);
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Optional<std::vector<std::string>> getExplicitPlatformLibs() {
  if (platformLib.getNumOccurrences() > 0)
    return parseLibNames(platformLib);
#if LDC_LLVM_VER < 1600
  return llvm::None;
#else
  return std::nullopt;
#endif
}

//////////////////////////////////////////////////////////////////////////////

bool useInternalLLDForLinking() {
  return linkInternally
#if LDC_WITH_LLD
         ||
         // MSVC: DWARF debuginfos and LTO require LLD
         (linkInternally.getNumOccurrences() == 0 && // not explicitly disabled
          opts::linker.empty() && // no explicitly selected linker
          global.params.targetTriple->isWindowsMSVCEnvironment() &&
          (opts::emitDwarfDebugInfo || opts::isUsingLTO()))
#endif
      ;
}

cl::boolOrDefault linkFullyStatic() { return staticFlag; }

bool linkAgainstSharedDefaultLibs() {
  // -static enforces static default libs.
  // Default to shared default libs for DLLs.
  return staticFlag != cl::BOU_TRUE &&
         (linkDefaultLibShared == cl::BOU_TRUE ||
          (linkDefaultLibShared == cl::BOU_UNSET && global.params.dll));
}

//////////////////////////////////////////////////////////////////////////////

llvm::StringRef getExplicitMscrtLibName() { return mscrtlib; }

llvm::StringRef getMscrtLibName(const bool *useInternalToolchain) {
  llvm::StringRef name = getExplicitMscrtLibName();
  if (!name.empty())
    return name;

  bool useInternal = false;
  if (useInternalToolchain) {
    useInternal = *useInternalToolchain;
  } else {
#ifdef _WIN32
    static bool haveMSVC = windows::isMsvcAvailable();
    useInternal = !haveMSVC;
#else
    useInternal = true;
#endif
  }

  if (useInternal) {
    return "vcruntime140";
  } else {
    return linkAgainstSharedDefaultLibs() ? "msvcrt" : "libcmt";
  }
}

//////////////////////////////////////////////////////////////////////////////

/// Insert an LLVM bitcode file into the module
static void insertBitcodeIntoModule(const char *bcFile, llvm::Module &M,
                                    llvm::LLVMContext &Context) {
  Logger::println("*** Linking-in bitcode file %s ***", bcFile);

  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> loadedModule(
      getLazyIRFileModule(bcFile, Err, Context));
  if (!loadedModule) {
    error(Loc(), "Error when loading LLVM bitcode file: %s", bcFile);
    fatal();
  }
  llvm::Linker(M).linkInModule(std::move(loadedModule));
}

/// Insert LLVM bitcode files into the module
void insertBitcodeFiles(llvm::Module &M, llvm::LLVMContext &Ctx,
                        Array<const char *> &bitcodeFiles) {
  for (const char *fname : bitcodeFiles) {
    insertBitcodeIntoModule(fname, M, Ctx);
  }
}

//////////////////////////////////////////////////////////////////////////////

// path to the produced executable/shared library
static std::string gExePath;

//////////////////////////////////////////////////////////////////////////////

int linkObjToBinary() {
  Logger::println("*** Linking executable ***");
  TimeTraceScope timeScope("Linking executable");

  // remember output path for later
  gExePath = getOutputPath();

  createDirectoryForFileOrFail(gExePath);

  const auto defaultLibNames = getDefaultLibNames();

  if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
    return linkObjToBinaryMSVC(gExePath, defaultLibNames);
  }

  return linkObjToBinaryGcc(gExePath, defaultLibNames);
}

const char *getPathToProducedBinary() {
  assert(!gExePath.empty());
  return gExePath.c_str();
}

//////////////////////////////////////////////////////////////////////////////

void deleteExeFile() {
  if (!gExePath.empty() && !llvm::sys::fs::is_directory(gExePath)) {
    llvm::sys::fs::remove(gExePath);
  }
}

//////////////////////////////////////////////////////////////////////////////

int runProgram() {
  TimeTraceScope timeScope("Run user program");

  assert(!gExePath.empty());

  // Run executable
  int status = executeToolAndWait(Loc(), gExePath, opts::runargs,
                                  global.params.v.verbose);
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
