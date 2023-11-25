//===-- linker-msvc.cpp ---------------------------------------------------===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/errors.h"
#include "driver/args.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/configfile.h"
#include "driver/exe_path.h"
#include "driver/linker.h"
#include "driver/tool.h"
#include "gen/logger.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#if LDC_WITH_LLD
#include "lld/Common/Driver.h"
#if LDC_LLVM_VER >= 1700
LLD_HAS_DRIVER(coff)
#endif
#endif

//////////////////////////////////////////////////////////////////////////////

namespace {

void addMscrtLibs(bool useInternalToolchain, std::vector<std::string> &args) {
  const auto mscrtlibName = getMscrtLibName(&useInternalToolchain);

  args.push_back(("/DEFAULTLIB:" + mscrtlibName).str());

  // We need the vcruntime lib for druntime's exception handling (ldc.eh_msvc).
  // Pick one of the 4 variants matching the selected main UCRT lib.

#if LDC_LLVM_VER >= 1700
#define contains_lower contains_insensitive
#define endswith_lower ends_with_insensitive
#elif LDC_LLVM_VER >= 1300
#define contains_lower contains_insensitive
#define endswith_lower endswith_insensitive
#endif
  if (useInternalToolchain) {
    assert(mscrtlibName.contains_lower("vcruntime"));
    return;
  }

  const bool isStatic = mscrtlibName.contains_lower("libcmt");

  const bool isDebug =
      mscrtlibName.endswith_lower("d") || mscrtlibName.endswith_lower("d.lib");

  const llvm::StringRef prefix = isStatic ? "lib" : "";
  const llvm::StringRef suffix = isDebug ? "d" : "";

  args.push_back(("/DEFAULTLIB:" + prefix + "vcruntime" + suffix).str());
}

void addLibIfFound(std::vector<std::string> &args, const llvm::Twine &name) {
  for (const char *dir : ConfigFile::instance.libDirs()) {
    llvm::SmallString<128> candidate(dir);
    llvm::sys::path::append(candidate, name);
    if (llvm::sys::fs::exists(candidate)) {
      args.emplace_back(candidate.data(), candidate.size());
      return;
    }
  }
}

void addSanitizerLibs(std::vector<std::string> &args) {
  if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
    args.push_back("ldc_rt.asan.lib");
  } else if (opts::isSanitizerEnabled(opts::LeakSanitizer)) {
    // If ASan is enabled, it includes LSan. So only add LSan link flags if ASan is _not_ enabled already.
    args.push_back("ldc_rt.lsan.lib");
  }
  if (opts::isSanitizerEnabled(opts::FuzzSanitizer)) {
    args.push_back("ldc_rt.fuzzer.lib");
    args.push_back("/SUBSYSTEM:CONSOLE"); // pull main() from fuzzer lib
  }

  // TODO: remaining sanitizers
}

} // anonymous namespace

//////////////////////////////////////////////////////////////////////////////

int linkObjToBinaryMSVC(llvm::StringRef outputPath,
                        const std::vector<std::string> &defaultLibNames) {
#ifdef _WIN32
  windows::MsvcEnvironmentScope msvcEnv;

  const bool forceMSVC = env::has(L"LDC_VSDIR_FORCE");
  const bool useInternalToolchain =
      (!forceMSVC && getExplicitMscrtLibName().contains_lower("vcruntime")) ||
      !msvcEnv.setup();

  if (forceMSVC && useInternalToolchain) {
    warning(Loc(), "no Visual C++ installation found for linking, falling back "
                   "to MinGW-based libraries");
  }
#else
  const bool useInternalToolchain = true;
#endif

  // build arguments
  std::vector<std::string> args;

  args.push_back("/NOLOGO");

  // specify that the image will contain a table of safe exception handlers
  // and can handle addresses >2GB (32bit only)
  if (global.params.targetTriple->isArch32Bit()) {
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
    // don't fold identical COMDATs (e.g., functions) if debuginfos are enabled,
    // otherwise breakpoints may not be hit
    args.push_back(global.params.symdebug ? "/OPT:NOICF" : "/OPT:ICF");
  }

  const bool willLinkAgainstSharedDefaultLibs =
      !defaultLibNames.empty() && linkAgainstSharedDefaultLibs();
  if (willLinkAgainstSharedDefaultLibs) {
    // Suppress linker warning LNK4217 wrt. 'importing locally defined symbol'
    // (dllimport of symbol dllexported from the same binary), because there
    // might be *many* of those (=> instantiated globals) if compiled with
    // -dllimport=all (and without -linkonce-templates).
    args.push_back("/IGNORE:4217");
  }

  // add C runtime libs
  addMscrtLibs(useInternalToolchain, args);

  // specify creation of DLL
  if (global.params.dll) {
    args.push_back("/DLL");
  }

  args.push_back(("/OUT:" + outputPath).str());

  // object files
  for (auto objfile : global.params.objfiles) {
    args.push_back(objfile);
  }

  // add precompiled rt.dso object file (in lib directory) when linking
  // against shared druntime
  const auto &libDirs = ConfigFile::instance.libDirs();
  if (willLinkAgainstSharedDefaultLibs && !libDirs.empty()) {
    args.push_back((llvm::Twine(libDirs[0]) + "/ldc_rt.dso.obj").str());
  }

  // .res/.def files
  if (global.params.resfile.length)
    args.push_back(global.params.resfile.ptr);
  if (global.params.deffile.length)
    args.push_back(std::string("/DEF:") + global.params.deffile.ptr);

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

  // lib dirs
  for (const char *dir_c : libDirs) {
    const llvm::StringRef dir(dir_c);
    if (!dir.empty())
      args.push_back(("/LIBPATH:" + dir).str());
  }

  if (useInternalToolchain && !libDirs.empty()) {
    args.push_back((llvm::Twine("/LIBPATH:") + libDirs[0] + "/mingw").str());
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

  auto explicitPlatformLibs = getExplicitPlatformLibs();
#if LDC_LLVM_VER >= 1600
  if (explicitPlatformLibs.has_value()) {
    for (auto &lib : explicitPlatformLibs.value()) {
      args.push_back(llvm::sys::path::has_extension(lib) ? std::move(lib)
                                                         : lib + ".lib");
    }
  }
#else
  if (explicitPlatformLibs.hasValue()) {
    for (auto &lib : explicitPlatformLibs.getValue()) {
      args.push_back(llvm::sys::path::has_extension(lib) ? std::move(lib)
                                                         : lib + ".lib");
    }
  }
#endif
  else {
    // default platform libs
    // TODO check which libaries are necessary
    args.insert(args.end(),
                {"kernel32.lib", "user32.lib", "gdi32.lib", "winspool.lib",
                 "shell32.lib", // required for dmain2.d
                 "ole32.lib", "oleaut32.lib", "uuid.lib", "comdlg32.lib",
                 "advapi32.lib",
                 // these get pulled in by druntime (rt/msvc.c); include
                 // explicitly for -betterC convenience (issue #3035)
                 "oldnames.lib", "legacy_stdio_definitions.lib"});
  }

  Logger::println("Linking with: ");
  Stream logstr = Logger::cout();
  for (const auto &arg : args) {
    if (!arg.empty()) {
      logstr << "'" << arg << "' ";
    }
  }
  logstr << "\n"; // FIXME where's flush ?

#if LDC_WITH_LLD
  if (useInternalLLDForLinking() ||
      (useInternalToolchain && opts::linker.empty())) {
    const auto fullArgs =
        getFullArgs("lld-link", args, global.params.v.verbose);

    const bool canExitEarly = false;
    const bool success = lld::coff::link(fullArgs
#if LDC_LLVM_VER < 1400
                                         ,
                                         canExitEarly
#endif
                                         ,
                                         llvm::outs(), llvm::errs()
#if LDC_LLVM_VER >= 1400
                                                           ,
                                         canExitEarly, false
#endif
    );

    if (!success)
      error(Loc(), "linking with LLD failed");

    return success ? 0 : 1;
  }
#endif

  // try to call linker
  std::string linker = opts::linker;
  if (linker.empty()) {
#ifdef _WIN32
    // default to lld-link.exe for LTO
    linker = opts::isUsingLTO() ? "lld-link.exe" : "link.exe";
#else
    linker = "lld-link";
#endif
  }

  return executeToolAndWait(Loc(), linker, args, global.params.v.verbose);
}
