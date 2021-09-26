//===-- tool.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/tool.h"

#include "dmd/errors.h"
#include "dmd/vsoptions.h"
#include "driver/args.h"
#include "driver/cl_options.h"
#include "driver/exe_path.h"
#include "driver/targetmachine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#ifdef _WIN32
#include <Windows.h>
#endif

//////////////////////////////////////////////////////////////////////////////

namespace opts {
llvm::cl::opt<std::string> linker(
    "linker", llvm::cl::ZeroOrMore,
    llvm::cl::value_desc("lld-link|lld|gold|bfd|..."),
    llvm::cl::desc("Set the linker to use. When explicitly set to '' "
                   "(nothing), prevents LDC from passing `-fuse-ld` to `cc`."),
    llvm::cl::cat(opts::linkingCategory));
}

static llvm::cl::opt<std::string>
    gcc("gcc", llvm::cl::ZeroOrMore, llvm::cl::cat(opts::linkingCategory),
        llvm::cl::value_desc("gcc|clang|..."),
        llvm::cl::desc(
            "C compiler to use for linking (and external assembling). Defaults "
            "to the CC environment variable if set, otherwise to `cc`."));

//////////////////////////////////////////////////////////////////////////////

static std::string findProgramByName(llvm::StringRef name) {
  llvm::ErrorOr<std::string> res = llvm::sys::findProgramByName(name);
  return res ? res.get() : std::string();
}

//////////////////////////////////////////////////////////////////////////////

std::string getProgram(const char *fallbackName,
                       const llvm::cl::opt<std::string> *opt,
                       const char *envVar) {
  std::string name;
  if (opt && !opt->empty()) {
    name = *opt;
  } else {
    if (envVar)
      name = env::get(envVar);
    if (name.empty()) // no or empty env var
      name = fallbackName;
  }

  const std::string path = findProgramByName(name);
  if (path.empty()) {
    error(Loc(), "cannot find program `%s`", name.c_str());
    fatal();
  }

  return path;
}

////////////////////////////////////////////////////////////////////////////////

std::string getGcc() { return getProgram("cc", &gcc, "CC"); }

////////////////////////////////////////////////////////////////////////////////

void appendTargetArgsForGcc(std::vector<std::string> &args) {
  using llvm::Triple;

  const auto &triple = *global.params.targetTriple;
  const auto arch64 = triple.get64BitArchVariant().getArch();

  switch (arch64) {
  // Specify -m32/-m64 for architectures where gcc supports those flags.
  case Triple::x86_64:
  case Triple::ppc64:
  case Triple::ppc64le:
  case Triple::sparcv9:
  case Triple::nvptx64:
    args.push_back(triple.isArch64Bit() ? "-m64" : "-m32");
    return;

  // MIPS does not have -m32/-m64 but requires -mabi=.
  case Triple::mips64:
  case Triple::mips64el:
    switch (getMipsABI()) {
    case MipsABI::EABI:
      args.push_back("-mabi=eabi");
      args.push_back("-march=mips32r2");
      break;
    case MipsABI::O32:
      args.push_back("-mabi=32");
      args.push_back("-march=mips32r2");
      break;
    case MipsABI::N32:
      args.push_back("-mabi=n32");
      args.push_back("-march=mips64r2");
      break;
    case MipsABI::N64:
      args.push_back("-mabi=64");
      args.push_back("-march=mips64r2");
      break;
    case MipsABI::Unknown:
      break;
    }
    return;

  default:
    break;
  }
}

//////////////////////////////////////////////////////////////////////////////

void createDirectoryForFileOrFail(llvm::StringRef fileName) {
  auto dir = llvm::sys::path::parent_path(fileName);
  if (!dir.empty() && !llvm::sys::fs::exists(dir)) {
    if (auto ec = llvm::sys::fs::create_directories(dir)) {
      error(Loc(), "failed to create path to file: %s\n%s", dir.data(),
            ec.message().c_str());
      fatal();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

std::vector<const char *> getFullArgs(const char *tool,
                                      const std::vector<std::string> &args,
                                      bool printVerbose) {
  std::vector<const char *> fullArgs;
  fullArgs.reserve(args.size() +
                   2); // args::executeAndWait() may append an additional null

  fullArgs.push_back(tool);
  for (const auto &arg : args)
    fullArgs.push_back(arg.c_str());

  // Print command line if requested
  if (printVerbose) {
    llvm::SmallString<256> singleString;
    for (auto arg : fullArgs) {
      singleString += arg;
      singleString += ' ';
    }
    message("%s", singleString.c_str());
  }

  return fullArgs;
}

////////////////////////////////////////////////////////////////////////////////

int executeToolAndWait(const std::string &tool_,
                       const std::vector<std::string> &args, bool verbose) {
  const auto tool = findProgramByName(tool_);
  if (tool.empty()) {
    error(Loc(), "cannot find program `%s`", tool_.c_str());
    return -1;
  }

  // Construct real argument list; first entry is the tool itself.
  auto fullArgs = getFullArgs(tool.c_str(), args, verbose);

  // We may need a response file to overcome cmdline limits, especially on Windows.
  auto rspEncoding = llvm::sys::WEM_UTF8;
#ifdef _WIN32
  // MSVC tools (link.exe etc.) apparently require UTF-16 encoded response files
  auto triple = global.params.targetTriple;
  if (triple && triple->isWindowsMSVCEnvironment())
    rspEncoding = llvm::sys::WEM_UTF16;
#endif

  // Execute tool.
  std::string errorMsg;
  const int status =
      args::executeAndWait(std::move(fullArgs), rspEncoding, &errorMsg);

  if (status) {
    error(Loc(), "%s failed with status: %d", tool.c_str(), status);
    if (!errorMsg.empty()) {
      errorSupplemental(Loc(), "message: %s", errorMsg.c_str());
    }
  }

  return status;
}

////////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32

namespace windows {

namespace {
bool setupMsvcEnvironmentImpl(
    std::vector<std::pair<std::wstring, wchar_t *>> *rollback) {
  const bool x64 = global.params.targetTriple->isArch64Bit();

  if (env::has(L"VSINSTALLDIR") && !env::has(L"LDC_VSDIR_FORCE")) {
    // Assume a fully set up environment (e.g., VS native tools command prompt).
    // Skip the MSVC setup unless the environment is set up for a different
    // target architecture.
    const auto tgtArch = env::get("VSCMD_ARG_TGT_ARCH"); // VS 2017+
    if (tgtArch.empty() || tgtArch == (x64 ? "x64" : "x86"))
      return true;
  }

  const auto begin = std::chrono::steady_clock::now();

  VSOptions vsOptions;
  vsOptions.initialize();
  if (!vsOptions.VSInstallDir)
    return false;

  llvm::SmallVector<const char *, 3> libPaths;
  if (auto vclibdir = vsOptions.getVCLibDir(x64))
    libPaths.push_back(vclibdir);
  if (auto ucrtlibdir = vsOptions.getUCRTLibPath(x64))
    libPaths.push_back(ucrtlibdir);
  if (auto sdklibdir = vsOptions.getSDKLibPath(x64))
    libPaths.push_back(sdklibdir);

  llvm::SmallVector<const char *, 2> binPaths;
  const char *secondaryBindir = nullptr;
  if (auto bindir = vsOptions.getVCBinDir(x64, secondaryBindir)) {
    binPaths.push_back(bindir);
    if (secondaryBindir)
      binPaths.push_back(secondaryBindir);
  }

  const bool success = libPaths.size() == 3 && !binPaths.empty();
  if (!success)
    return false;

  if (!rollback) // check for availability only
    return true;

  if (global.params.verbose)
    message("Prepending to environment variables:");

  const auto preprendToEnvVar =
      [rollback](const char *key, const wchar_t *wkey,
                 const llvm::SmallVectorImpl<const char *> &entries) {
        wchar_t *originalValue = _wgetenv(wkey);

        llvm::SmallString<256> head;
        for (const char *entry : entries) {
          if (!head.empty())
            head += ';';
          head += entry;
        }

        if (global.params.verbose)
          message("  %s += %.*s", key, (int)head.size(), head.data());

        llvm::SmallVector<wchar_t, 1024> wvalue;
        llvm::sys::windows::UTF8ToUTF16(head, wvalue);
        if (originalValue) {
          wvalue.push_back(L';');
          wvalue.append(originalValue, originalValue + wcslen(originalValue));
        }
        wvalue.push_back(0);

        // copy the original value, if set
        if (originalValue)
          originalValue = wcsdup(originalValue);
        rollback->emplace_back(wkey, originalValue);

        SetEnvironmentVariableW(wkey, wvalue.data());
      };

  rollback->reserve(2);
  preprendToEnvVar("LIB", L"LIB", libPaths);
  preprendToEnvVar("PATH", L"PATH", binPaths);

  if (global.params.verbose) {
    const auto end = std::chrono::steady_clock::now();
    message("MSVC setup took %lld microseconds",
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                .count());
  }

  return true;
}
} // anonymous namespace

bool isMsvcAvailable() { return setupMsvcEnvironmentImpl(nullptr); }

bool MsvcEnvironmentScope::setup() {
  rollback.clear();
  return setupMsvcEnvironmentImpl(&rollback);
}

MsvcEnvironmentScope::~MsvcEnvironmentScope() {
  for (const auto &pair : rollback) {
    SetEnvironmentVariableW(pair.first.c_str(), pair.second);
    free(pair.second);
  }
}

} // namespace windows

#endif // _WIN32
