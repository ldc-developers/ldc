//===-- linker-msvc.cpp ---------------------------------------------------===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "errors.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/exe_path.h"
#include "driver/linker.h"
#include "driver/tool.h"
#include "gen/logger.h"

#include "llvm/Support/FileSystem.h"

#if LDC_WITH_LLD
#if LDC_LLVM_VER >= 600
#include "lld/Common/Driver.h"
#else
#include "lld/Driver/Driver.h"
#endif
#endif

//////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<std::string>
    mscrtlib("mscrtlib", llvm::cl::ZeroOrMore,
             llvm::cl::desc("MS C runtime library to link with"),
             llvm::cl::value_desc("libcmt[d]|msvcrt[d]"),
             llvm::cl::cat(opts::linkingCategory));

//////////////////////////////////////////////////////////////////////////////

namespace {

void addMscrtLibs(std::vector<std::string> &args) {
  llvm::StringRef mscrtlibName = mscrtlib;
  if (mscrtlibName.empty()) {
    // default to static release variant
    mscrtlibName =
        linkFullyStatic() != llvm::cl::BOU_FALSE ? "libcmt" : "msvcrt";
  }

  args.push_back(("/DEFAULTLIB:" + mscrtlibName).str());

  const bool isStatic = mscrtlibName.startswith_lower("libcmt");
  const bool isDebug =
      mscrtlibName.endswith_lower("d") || mscrtlibName.endswith_lower("d.lib");

  const llvm::StringRef prefix = isStatic ? "lib" : "";
  const llvm::StringRef suffix = isDebug ? "d" : "";

  args.push_back(("/DEFAULTLIB:" + prefix + "vcruntime" + suffix).str());
}

void addLibIfFound(std::vector<std::string> &args, const llvm::Twine &name) {
  if (llvm::sys::fs::exists(exe_path::prependLibDir(name)))
    args.push_back(name.str());
}

void addSanitizerLibs(std::vector<std::string> &args) {
  if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
    args.push_back("ldc_rt.asan.lib");
  }

  // TODO: remaining sanitizers
}

} // anonymous namespace

//////////////////////////////////////////////////////////////////////////////

int linkObjToBinaryMSVC(llvm::StringRef outputPath,
                        const std::vector<std::string> &defaultLibNames) {
  if (!opts::ccSwitches.empty()) {
    error(Loc(), "-Xcc is not supported for MSVC");
    fatal();
  }

#ifdef _WIN32
  windows::setupMsvcEnvironment();
#endif

  // build arguments
  std::vector<std::string> args;

  args.push_back("/NOLOGO");

  // specify that the image will contain a table of safe exception handlers
  // and can handle addresses >2GB (32bit only)
  if (!global.params.is64bit) {
    args.push_back("/SAFESEH");
    args.push_back("/LARGEADDRESSAWARE");
  }

  // output debug information
  if (global.params.symdebug) {
    args.push_back("/DEBUG");
  }

  // remove dead code and fold identical COMDATs
  if (opts::disableLinkerStripDead) {
    args.push_back("/OPT:NOREF");
  } else {
    args.push_back("/OPT:REF");
    args.push_back("/OPT:ICF");
  }

  // add C runtime libs
  addMscrtLibs(args);

  // specify creation of DLL
  if (global.params.dll) {
    args.push_back("/DLL");
  }

  args.push_back(("/OUT:" + outputPath).str());

  // object files
  for (auto objfile : global.params.objfiles) {
    args.push_back(objfile);
  }

  // .res/.def files
  if (global.params.resfile)
    args.push_back(global.params.resfile);
  if (global.params.deffile)
    args.push_back(std::string("/DEF:") + global.params.deffile);

  if (opts::enableDynamicCompile) {
    args.push_back("ldc-jit-rt.lib");
    args.push_back("ldc-jit.lib");
  }

  // user libs
  for (auto libfile : global.params.libfiles) {
    args.push_back(libfile);
  }

  // LLVM compiler-rt libs
  addLibIfFound(args, "ldc_rt.builtins.lib");
  addSanitizerLibs(args);
  if (opts::isInstrumentingForPGO()) {
    args.push_back("ldc_rt.profile.lib");
    // it depends on ws2_32 for symbol `gethostname`
    args.push_back("ws2_32.lib");
  }

  // additional linker switches
  auto addSwitch = [&](std::string str) {
    if (str.length() > 2) {
      // rewrite common -L and -l switches
      if (str[0] == '-' && str[1] == 'L') {
        str = "/LIBPATH:" + str.substr(2);
      } else if (str[0] == '-' && str[1] == 'l') {
        str = str.substr(2) + ".lib";
      }
    }
    args.push_back(str);
  };

  for (const auto &str : opts::linkerSwitches) {
    addSwitch(str);
  }

  // default libs
  for (const auto &name : defaultLibNames) {
    args.push_back(name + ".lib");
  }

  // libs added via pragma(lib, libname) - should be empty due to embedded
  // references in object file
  for (auto ls : global.params.linkswitches) {
    addSwitch(ls);
  }

  // default platform libs
  // TODO check which libaries are necessary
  args.push_back("kernel32.lib");
  args.push_back("user32.lib");
  args.push_back("gdi32.lib");
  args.push_back("winspool.lib");
  args.push_back("shell32.lib"); // required for dmain2.d
  args.push_back("ole32.lib");
  args.push_back("oleaut32.lib");
  args.push_back("uuid.lib");
  args.push_back("comdlg32.lib");
  args.push_back("advapi32.lib");

  Logger::println("Linking with: ");
  Stream logstr = Logger::cout();
  for (const auto &arg : args) {
    if (!arg.empty()) {
      logstr << "'" << arg << "' ";
    }
  }
  logstr << "\n"; // FIXME where's flush ?

#if LDC_WITH_LLD
  if (useInternalLLDForLinking()) {
    const auto fullArgs =
        getFullArgs("lld-link.exe", args, global.params.verbose);

#if LDC_LLVM_VER >= 600
    const bool success = lld::coff::link(fullArgs, /*CanExitEarly=*/false);
#else
    const bool success = lld::coff::link(fullArgs);
#endif
    if (!success)
      error(Loc(), "linking with LLD failed");

    return success ? 0 : 1;
  }
#endif

  // try to call linker
  std::string linker = opts::linker;
  if (linker.empty())
    linker = "link.exe";

  return executeToolAndWait(linker, args, global.params.verbose);
}
