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
#include "llvm/Support/Program.h"

#ifdef _WIN32
#include <Windows.h>
#include <numeric>
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

std::string getProgram(const char *name, const llvm::cl::opt<std::string> *opt,
                       const char *envVar) {
  std::string path;

  if (opt && !opt->empty()) {
    path = findProgramByName(opt->c_str());
  }

  if (path.empty() && envVar) {
    const std::string prog = env::get(envVar);
    if (!prog.empty())
      path = findProgramByName(prog);
  }

  if (path.empty()) {
    path = findProgramByName(name);
  }

  if (path.empty()) {
    error(Loc(), "failed to locate %s", name);
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
                   2); // executeToolAndWait() appends an additional null

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

#if LDC_LLVM_VER >= 700
namespace {
std::vector<llvm::StringRef> toRefsVector(llvm::ArrayRef<const char *> args) {
  std::vector<llvm::StringRef> refs;
  refs.reserve(args.size());
  for (const char *arg : args)
    refs.emplace_back(arg);
  return refs;
}

#ifdef _WIN32
struct ResponseFile {
  llvm::SmallString<128> path;

  ~ResponseFile() {
    if (!path.empty()) {
      if (llvm::sys::fs::remove(path)) {
        warning(Loc(), "could not remove response file");
      }
    }
  }

  // Creates an appropriate response file if needed (=> path non-empty).
  // Returns false on error.
  bool setup(llvm::ArrayRef<const char *> fullArgs) {
    assert(path.empty());

    const size_t totalLen = std::accumulate(
        fullArgs.begin(), fullArgs.end(),
        fullArgs.size() * 3, // quotes + space
        [](size_t acc, const char *arg) { return acc + strlen(arg); });

    if (totalLen <= 32767)
      return true; // nothing to do

    if (llvm::sys::fs::createTemporaryFile("ldc", "rsp", path))
      return false;

    const std::string content =
        llvm::sys::flattenWindowsCommandLine(toRefsVector(fullArgs.slice(1)));

    // MSVC tools apparently require UTF-16
    if (llvm::sys::writeFileWithEncoding(path, content, llvm::sys::WEM_UTF16))
      return false;

    return true;
  }
};
#endif // _WIN32
}
#endif // LDC_LLVM_VER >= 700

int executeToolAndWait(const std::string &tool_,
                       std::vector<std::string> const &args, bool verbose) {
  const auto tool = findProgramByName(tool_);
  if (tool.empty()) {
    error(Loc(), "failed to locate %s", tool_.c_str());
    return -1;
  }

  // Construct real argument list; first entry is the tool itself.
  auto realargs = getFullArgs(tool.c_str(), args, verbose);

#if LDC_LLVM_VER >= 700
// We may need a response file on Windows hosts to overcome cmdline limits.
#ifdef _WIN32
  ResponseFile rspFile;
  if (!rspFile.setup(realargs)) {
    error(Loc(), "could not write temporary response file");
    return -1;
  }

  std::string rspArg;
  if (!rspFile.path.empty()) {
    rspArg = ("@" + rspFile.path).str();
    realargs.resize(1); // tool only
    realargs.push_back(rspArg.c_str());
  }
#endif // _WIN32

  const std::vector<llvm::StringRef> argv = toRefsVector(realargs);
  auto envVars = llvm::None;
#else // LDC_LLVM_VER < 700
  realargs.push_back(nullptr); // terminate with null
  auto argv = &realargs[0];
  auto envVars = nullptr;
#endif

  // Execute tool.
  std::string errstr;
  const int status =
      llvm::sys::ExecuteAndWait(tool, argv, envVars, {}, 0, 0, &errstr);

  if (status) {
    error(Loc(), "%s failed with status: %d", tool.c_str(), status);
    if (!errstr.empty()) {
      errorSupplemental(Loc(), "message: %s", errstr.c_str());
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
