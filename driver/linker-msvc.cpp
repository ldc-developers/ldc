//===-- linker-msvc.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "errors.h"
#include "driver/cl_options.h"
#include "driver/tool.h"
#include "gen/logger.h"

//////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<std::string> mscrtlib(
    "mscrtlib", llvm::cl::ZeroOrMore, llvm::cl::value_desc("name"),
    llvm::cl::desc(
        "MS C runtime library to link against (libcmt[d] / msvcrt[d])"));

//////////////////////////////////////////////////////////////////////////////

namespace {

void addMscrtLibs(std::vector<std::string> &args,
                  llvm::cl::boolOrDefault fullyStaticFlag) {
  llvm::StringRef mscrtlibName = mscrtlib;
  if (mscrtlibName.empty()) {
    // default to static release variant
    mscrtlibName = fullyStaticFlag != llvm::cl::BOU_FALSE ? "libcmt" : "msvcrt";
  }

  args.push_back(("/DEFAULTLIB:" + mscrtlibName).str());

  const bool isStatic = mscrtlibName.startswith_lower("libcmt");
  const bool isDebug =
      mscrtlibName.endswith_lower("d") || mscrtlibName.endswith_lower("d.lib");

  const llvm::StringRef prefix = isStatic ? "lib" : "";
  const llvm::StringRef suffix = isDebug ? "d" : "";

  args.push_back(("/DEFAULTLIB:" + prefix + "vcruntime" + suffix).str());
}

} // anonymous namespace

//////////////////////////////////////////////////////////////////////////////

int linkObjToBinaryMSVC(llvm::StringRef outputPath,
                        llvm::cl::boolOrDefault fullyStaticFlag) {
  if (!opts::ccSwitches.empty()) {
    error(Loc(), "-Xcc is not supported for MSVC");
    fatal();
  }

#ifdef _WIN32
  windows::setupMsvcEnvironment();
#endif

  const std::string tool = "link.exe";

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
  addMscrtLibs(args, fullyStaticFlag);

  // specify creation of DLL
  if (global.params.dll) {
    args.push_back("/DLL");
  }

  args.push_back(("/OUT:" + outputPath).str());

  // object files
  for (unsigned i = 0; i < global.params.objfiles->dim; i++)
    args.push_back((*global.params.objfiles)[i]);

  // .res/.def files
  if (global.params.resfile)
    args.push_back(global.params.resfile);
  if (global.params.deffile)
    args.push_back(std::string("/DEF:") + global.params.deffile);

  // Link with profile-rt library when generating an instrumented binary
  // profile-rt depends on Phobos (MD5 hashing).
  if (global.params.genInstrProf) {
    args.push_back("ldc-profile-rt.lib");
    // profile-rt depends on ws2_32 for symbol `gethostname`
    args.push_back("ws2_32.lib");
  }

  // user libs
  for (unsigned i = 0; i < global.params.libfiles->dim; i++)
    args.push_back((*global.params.libfiles)[i]);

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

  for (unsigned i = 0; i < global.params.linkswitches->dim; i++) {
    addSwitch(global.params.linkswitches->data[i]);
  }

  // default libs
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

  // try to call linker
  return executeToolAndWait(tool, args, global.params.verbose);
}
