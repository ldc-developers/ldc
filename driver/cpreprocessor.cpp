#include "driver/cpreprocessor.h"

#include "dmd/errors.h"
#include "driver/cl_options.h"
#include "driver/timetrace.h"
#include "driver/tool.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
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

const std::string &getCC(bool isMSVC,
                         std::vector<std::string> &additional_args) {
  static std::string cached_cc;
  static std::vector<std::string> cached_args;
  if (cached_cc.empty()) {
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
    cached_cc = getGcc(cached_args, fallback.c_str());
  }

  additional_args.insert(additional_args.end(), cached_args.cbegin(), cached_args.cend());
  return cached_cc;
}

FileName getOutputPath(const Loc &loc, const char *csrcfile) {
  llvm::SmallString<64> buffer;

  // 1) create a new temporary directory (e.g., `/tmp/itmp-ldc-10ecec`)
  auto ec = llvm::sys::fs::createUniqueDirectory("itmp-ldc", buffer);
  if (ec) {
    error(loc,
          "failed to create temporary directory for preprocessed .i file: %s\n%s",
          buffer.c_str(), ec.message().c_str());
    fatal();
  }

  // 2) append the .c file name, replacing the extension with .i
  llvm::sys::path::append(buffer, FileName::name(csrcfile));
  llvm::sys::path::replace_extension(buffer, i_ext.ptr);

  // the directory is removed (after the file) in Module.read()

  return FileName::create(buffer.c_str()); // allocates a copy
}
} // anonymous namespace

FileName runCPreprocessor(FileName csrcfile, const Loc &loc, bool &ifile,
                          OutBuffer &defines) {
  TimeTraceScope timeScope("Preprocess C file", csrcfile.toChars());

  const char *importc_h = getPathToImportc_h(loc);

  const auto &triple = *global.params.targetTriple;
  const bool isMSVC = triple.isWindowsMSVCEnvironment();

#ifdef _WIN32
  windows::MsvcEnvironmentScope msvcEnv;
  if (isMSVC)
    msvcEnv.setup(/*forPreprocessingOnly=*/true);
#endif

  FileName ipath = getOutputPath(loc, csrcfile.toChars());

  std::vector<std::string> args;
  const std::string &cc = getCC(isMSVC, args);

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
    args.push_back((llvm::Twine("/Fi") + ipath.toChars()).str());
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
    args.push_back(ipath.toChars());
  }

  const int status = executeToolAndWait(loc, cc, args, global.params.v.verbose);
  if (status) {
    errorSupplemental(loc, "C preprocessor failed for file '%s'", csrcfile.toChars());
    fatal();
  }

  ifile = true;
  return ipath;
}
