//===-- linker.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "errors.h"
#include "driver/cl_options.h"
#include "driver/linker.h"
#include "driver/tool.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"

namespace cl = llvm::cl;

//////////////////////////////////////////////////////////////////////////////

#if LDC_WITH_LLD
static cl::opt<bool> useInternalLinker("link-internally", cl::ZeroOrMore,
                                       cl::desc("Use internal LLD for linking"),
                                       cl::cat(opts::linkingCategory));
#else
constexpr bool useInternalLinker = false;
#endif

static cl::opt<cl::boolOrDefault>
    staticFlag("static", cl::ZeroOrMore,
               cl::desc("Create a statically linked binary, including "
                        "all system dependencies"),
               cl::cat(opts::linkingCategory));

static cl::opt<cl::boolOrDefault> linkDefaultLibShared(
    "link-defaultlib-shared", cl::ZeroOrMore,
    cl::desc("Link with shared versions of default libraries"),
    cl::cat(opts::linkingCategory));

//////////////////////////////////////////////////////////////////////////////

// linker-gcc.cpp
int linkObjToBinaryGcc(llvm::StringRef outputPath, bool useInternalLinker,
                       cl::boolOrDefault fullyStaticFlag);

// linker-msvc.cpp
int linkObjToBinaryMSVC(llvm::StringRef outputPath, bool useInternalLinker,
                        cl::boolOrDefault fullyStaticFlag);

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
  std::string result =
      global.params.objfiles.dim
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
      result = tempFilename.str();
  } else if (extension) {
    result += '.';
    result += extension;
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////

bool willLinkAgainstSharedDefaultLibs() {
  // -static enforces static default libs.
  // Default to shared default libs for DLLs.
  return staticFlag != cl::BOU_TRUE &&
         (linkDefaultLibShared == cl::BOU_TRUE ||
          (linkDefaultLibShared == cl::BOU_UNSET && global.params.dll));
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
#if LDC_LLVM_VER >= 308
  llvm::Linker(M).linkInModule(std::move(loadedModule));
#else
  llvm::Linker(&M).linkInModule(loadedModule.release());
#endif
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

  // remember output path for later
  gExePath = getOutputName();

  createDirectoryForFileOrFail(gExePath);

  if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
    return linkObjToBinaryMSVC(gExePath, useInternalLinker, staticFlag);
  }

  return linkObjToBinaryGcc(gExePath, useInternalLinker, staticFlag);
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
