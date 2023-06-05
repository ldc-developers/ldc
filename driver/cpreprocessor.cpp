#include "driver/cpreprocessor.h"

#include "dmd/errors.h"
#include "driver/cl_options.h"
#include "driver/tool.h"
#include "llvm/Support/Program.h"

namespace {
const char *getPathToImportc_h(const Loc &loc) {
  // importc.h should be next to object.d
  static const char *cached = nullptr;
  if (!cached) {
    cached = FileName::searchPath(global.path, "importc.h", false);
    if (!cached) {
      error(loc, "cannot find \"importc.h\" along import path");
      fatal();
    }
  }
  return cached;
}

const std::string &getCC(bool isMSVC) {
  static std::string cached;
  if (cached.empty()) {
    std::string fallback = "cc";
    if (isMSVC) {
#ifdef _WIN32
      // by default, prefer clang-cl.exe (if in PATH) over cl.exe
      // (e.g., no echoing of source filename being preprocessed to stderr)
      auto found = llvm::sys::findProgramByName("clang-cl.exe");
      if (found) {
        fallback = found.get();
      } else {
        fallback = "cl.exe";
      }
#else
      fallback = "clang-cl";
#endif
    }
    cached = getGcc(fallback.c_str());
  }
  return cached;
}
} // anonymous namespace

FileName runCPreprocessor(FileName csrcfile, const Loc &loc, bool &ifile,
                          OutBuffer &defines) {
  const char *importc_h = getPathToImportc_h(loc);
  const char *ifilename = FileName::forceExt(csrcfile.toChars(), i_ext.ptr);

  const auto &triple = *global.params.targetTriple;
  const bool isMSVC = triple.isWindowsMSVCEnvironment();

#if 0 //ifdef _WIN32
  // TODO: INCLUDE env var etc.?
  windows::MsvcEnvironmentScope msvcEnv;
  if (isMSVC)
    msvcEnv.setup();
#endif

  const std::string &cc = getCC(isMSVC);
  std::vector<std::string> args;

  if (!isMSVC)
    appendTargetArgsForGcc(args);

  if (triple.isOSDarwin())
    args.push_back("-fno-blocks"); // disable blocks extension

  for (const auto &ccSwitch : opts::ccSwitches) {
    args.push_back(ccSwitch);
  }
  for (const auto &cppSwitch : opts::cppSwitches) {
    args.push_back(cppSwitch);
  }

  if (isMSVC) {
    args.push_back("/nologo");
    args.push_back("/P"); // preprocess only

    const bool isClangCl = llvm::StringRef(cc)
#if LDC_LLVM_VER >= 1300
                               .contains_insensitive("clang-cl");
#else
                               .contains_lower("clang-cl");
#endif

    if (!isClangCl) {
      args.push_back("/PD");              // print all macro definitions
      args.push_back("/Zc:preprocessor"); // use the new conforming preprocessor
    } else {
      // print macro definitions (clang-cl doesn't support /PD - use clang's
      // -dD)
      args.push_back("-Xclang");
      args.push_back("-dD");

      // need to redefine some macros in importc.h
      args.push_back("-Wno-builtin-macro-redefined");
    }

    args.push_back(csrcfile.toChars());
    args.push_back((llvm::Twine("/FI") + importc_h).str());
    // preprocessed output file
    args.push_back((llvm::Twine("/Fi") + ifilename).str());
  } else { // Posix
    // merge #define's with output:
    // https://gcc.gnu.org/onlinedocs/cpp/Invocation.html#index-dD
    args.push_back("-dD");

    // need to redefine some macros in importc.h
    args.push_back("-Wno-builtin-macro-redefined");

    args.push_back("-E"); // run preprocessor only
    args.push_back("-include");
    args.push_back(importc_h);
    args.push_back(csrcfile.toChars());
    args.push_back("-o");
    args.push_back(ifilename);
  }

  const int status = executeToolAndWait(loc, cc, args, global.params.verbose);
  if (status) {
    errorSupplemental(loc, "C preprocessor failed for file '%s'", csrcfile.toChars());
    fatal();
  }

  ifile = true;
  return FileName::create(ifilename);
}
