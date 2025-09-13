#include "driver/cpreprocessor.h"

#include "dmd/errors.h"
#include "dmd/timetrace.h"
#include "driver/cl_options.h"
#include "driver/tool.h"
#include "gen/irstate.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace {
const char *getPathToImportc_h(Loc loc) {
  // importc.h should be next to object.d
  static const char *cached = nullptr;
  if (!cached) {
    cached = FileName::searchPath(global.importPaths, "importc.h", false);
    if (!cached) {
      error(loc, "cannot find \"importc.h\" along import path");
      fatal();
    }

#ifdef _WIN32
    // if the path to importc.h is relative, cl.exe (but not clang-cl) treats it as relative to the .c file!
    cached = FileName::toAbsolute(cached);
#endif
  }
  return cached;
}

FileName getOutputPath(Loc loc, const char *csrcfile) {
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

FileName runCPreprocessor(FileName csrcfile, Loc loc, OutBuffer &defines) {
  dmd::TimeTraceScope timeScope("Preprocess C file", csrcfile.toChars(), loc);

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
  const std::string &cc = getCC(args);

  args.push_back(isMSVC ? "/std:c11" : "-std=c11");

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

    const bool isClangCl = llvm::StringRef(cc).contains_insensitive("clang-cl");

    if (!isClangCl) {
      args.push_back("/PD");              // print all macro definitions
      args.push_back("/Zc:preprocessor"); // use the new conforming preprocessor
    } else {
      // propagate the target to the preprocessor
      args.push_back("--target=" + triple.getTriple());

#if LDC_LLVM_VER >= 1800 // getAllProcessorFeatures was introduced in this version
      // propagate all enabled/disabled features to the preprocessor
      const auto &subTarget = gTargetMachine->getMCSubtargetInfo();
      const auto &featureBits = subTarget->getFeatureBits();
      llvm::SmallString<64> featureString;
      for (const auto &feature : subTarget->getAllProcessorFeatures()) {
        args.push_back("-Xclang");
        args.push_back("-target-feature");
        args.push_back("-Xclang");

        featureString += featureBits.test(feature.Value) ? '+' : '-';
        featureString += feature.Key;
        args.push_back(featureString.str().str());
        featureString.clear();
      }
#endif

      // print macro definitions (clang-cl doesn't support /PD - use clang's
      // -dD)
      args.push_back("/clang:-dD");

      // need to redefine some macros in importc.h
      args.push_back("-Wno-builtin-macro-redefined");

      // disable the clang resource headers (immintrin.h etc.), using
      // unsupported types like __int128, __bf16 etc. - stick to the MS headers
      args.push_back("-nobuiltininc");
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

  return ipath;
}
