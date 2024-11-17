//===-- cl_options_sanitizers.cpp -------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Creates and handles the -fsanitize=... and -fsanitize-coverage=...
// commandline options.
//
//===----------------------------------------------------------------------===//

#include "driver/cl_options_sanitizers.h"

#include "dmd/errors.h"
#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
#include "dmd/mangle.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SpecialCaseList.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"

using namespace dmd;

namespace {

using namespace opts;

cl::list<std::string> fSanitize(
    "fsanitize", cl::CommaSeparated,
    cl::desc("Turn on runtime checks for various forms of undefined or "
             "suspicious behavior."),
    cl::value_desc("checks"));

cl::list<std::string> fSanitizeBlacklist(
    "fsanitize-blacklist", cl::CommaSeparated,
    cl::desc("Add <file> to the blacklist files for the sanitizers."),
    cl::value_desc("file"));

std::unique_ptr<llvm::SpecialCaseList> sanitizerBlacklist;

cl::list<std::string> fSanitizeCoverage(
    "fsanitize-coverage", cl::CommaSeparated,
    cl::desc("Specify the type of coverage instrumentation for -fsanitize"),
    cl::value_desc("type"));

llvm::SanitizerCoverageOptions sanitizerCoverageOptions;

cl::list<std::string>
    fSanitizeRecover("fsanitize-recover", cl::CommaSeparated,
                     cl::desc("Enable recovery for specified sanitizers ('all' "
                              "= enable for all supported sanitizers)."),
                     cl::value_desc("name"));

SanitizerBits parseFSanitizeCmdlineParameter() {
  SanitizerBits retval = 0;
  for (const auto &name : fSanitize) {
    SanitizerCheck check = parseSanitizerName(name, [&name] {
      error(Loc(), "Unrecognized -fsanitize value '%s'.", name.c_str());
    });
    retval |= SanitizerBits(check);
  }
  return retval;
}

SanitizerBits parseFSanitizeRecoverCmdlineParameter() {
  SanitizerBits retval = 0;
  for (const auto &name : fSanitizeRecover) {
    if (name == "all") {
      retval |= ~SanitizerBits(0);
      continue;
    }
    SanitizerCheck check = parseSanitizerName(name, [&name] {
      error(Loc(), "fsanitize-recover: Unrecognized sanitizer name '%s'.", name.c_str());
    });
    if (check & supportedSanitizerRecoveries)
      retval |= SanitizerBits(check);
    else
      error(Loc(), "fsanitize-recover: Recovery not supported for '%s'.", name.c_str());
  }
  return retval;
}

void parseFSanitizeCoverageParameter(llvm::StringRef name,
                                     llvm::SanitizerCoverageOptions &opts) {
  if (name == "func") {
    opts.CoverageType = std::max(opts.CoverageType,
                                 llvm::SanitizerCoverageOptions::SCK_Function);
  } else if (name == "bb") {
    opts.CoverageType =
        std::max(opts.CoverageType, llvm::SanitizerCoverageOptions::SCK_BB);
  } else if (name == "edge") {
    opts.CoverageType =
        std::max(opts.CoverageType, llvm::SanitizerCoverageOptions::SCK_Edge);
  } else if (name == "indirect-calls") {
    opts.IndirectCalls = true;
  } else if (name == "trace-bb") {
    opts.TraceBB = true;
  }
  else if (name == "trace-cmp") {
    opts.TraceCmp = true;
  }
  else if (name == "trace-div") {
    opts.TraceDiv = true;
  }
  else if (name == "trace-gep") {
    opts.TraceGep = true;
  }
  else if (name == "trace-loads") {
    opts.TraceLoads = true;
  }
  else if (name == "trace-stores") {
    opts.TraceStores = true;
  }
  else if (name == "8bit-counters") {
    opts.Use8bitCounters = true;
  }
  else if (name == "trace-pc") {
    opts.TracePC = true;
  }
  else if (name == "trace-pc-guard") {
    opts.TracePCGuard = true;
  }
  else if (name == "inline-8bit-counters") {
    opts.Inline8bitCounters = true;
  }
  else if (name == "no-prune") {
    opts.NoPrune = true;
  }
  else if (name == "pc-table") {
    opts.PCTable = true;
  }
  else {
    error(Loc(), "Unrecognized -fsanitize-coverage option '%s'.", name.str().c_str());
  }
}

void parseFSanitizeCoverageCmdlineParameter(llvm::SanitizerCoverageOptions &opts) {
  for (const auto &name : fSanitizeCoverage) {
    // Enable CoverageSanitizer when one or more -fsanitize-coverage parameters are passed
    enabledSanitizers |= CoverageSanitizer;

    parseFSanitizeCoverageParameter(name, opts);
  }
}

} // anonymous namespace

namespace opts {

cl::opt<llvm::AsanDetectStackUseAfterReturnMode> fSanitizeAddressUseAfterReturn(
    "fsanitize-address-use-after-return", cl::ZeroOrMore,
    cl::desc("Select the mode of detecting stack use-after-return (UAR) in "
             "AddressSanitizer: never | runtime (default) | always"),
    cl::init(llvm::AsanDetectStackUseAfterReturnMode::Runtime),
    cl::values(
        clEnumValN(
            llvm::AsanDetectStackUseAfterReturnMode::Never, "never",
            "Completely disables detection of UAR errors (reduces code size)."),
        clEnumValN(llvm::AsanDetectStackUseAfterReturnMode::Runtime, "runtime",
                   "Adds the code for detection, but it can be disabled via the "
                   "runtime environment "
                   "(ASAN_OPTIONS=detect_stack_use_after_return=0). Requires "
                   "druntime support."),
        clEnumValN(
            llvm::AsanDetectStackUseAfterReturnMode::Always, "always",
            "Enables detection of UAR errors in all cases. (reduces code size, "
            "but not as much as never). Requires druntime support.")));

SanitizerBits enabledSanitizers = 0;
SanitizerBits enabledSanitizerRecoveries = 0;
const SanitizerBits supportedSanitizerRecoveries = AddressSanitizer | MemorySanitizer;

// Parse sanitizer name passed on commandline and return the corresponding
// sanitizer bits.
SanitizerCheck parseSanitizerName(llvm::StringRef name,
                                  std::function<void()> actionUponError) {
  SanitizerCheck parsedValue = llvm::StringSwitch<SanitizerCheck>(name)
                                   .Case("address", AddressSanitizer)
                                   .Case("fuzzer", FuzzSanitizer)
                                   .Case("leak", LeakSanitizer)
                                   .Case("memory", MemorySanitizer)
                                   .Case("thread", ThreadSanitizer)
                                   .Default(NoneSanitizer);

  if (parsedValue == NoneSanitizer) {
    actionUponError();
  }

  return parsedValue;
}

void initializeSanitizerOptionsFromCmdline()
{
  enabledSanitizers |= parseFSanitizeCmdlineParameter();
  enabledSanitizerRecoveries |= parseFSanitizeRecoverCmdlineParameter();

  auto &sancovOpts = sanitizerCoverageOptions;

  // The Fuzz sanitizer implies -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table
  if (isSanitizerEnabled(FuzzSanitizer)) {
    enabledSanitizers |= CoverageSanitizer;
    sancovOpts.Inline8bitCounters = true;
    sancovOpts.PCTable = true;
    sancovOpts.IndirectCalls = true;
    sancovOpts.TraceCmp = true;
  }

  parseFSanitizeCoverageCmdlineParameter(sancovOpts);

  // trace-pc/trace-pc-guard/inline-8bit-counters without specifying the
  // insertion type implies edge
  if ((sancovOpts.CoverageType == llvm::SanitizerCoverageOptions::SCK_None) &&
      (sancovOpts.TracePC || sancovOpts.TracePCGuard ||
       sancovOpts.Inline8bitCounters)) {
    sancovOpts.CoverageType = llvm::SanitizerCoverageOptions::SCK_Edge;
  }

  if (isAnySanitizerEnabled() && !fSanitizeBlacklist.empty()) {
    std::string loadError;
    sanitizerBlacklist = llvm::SpecialCaseList::create(
        fSanitizeBlacklist, *llvm::vfs::getRealFileSystem(), loadError);
    if (!sanitizerBlacklist)
      error(Loc(), "-fsanitize-blacklist error: %s", loadError.c_str());
  }
}

llvm::SanitizerCoverageOptions getSanitizerCoverageOptions() {
  return sanitizerCoverageOptions;
}

// Output to `hash_os` all optimization settings that influence object code
// output and that are not observable in the IR before running LLVM passes. This
// is used to calculate the hash use for caching that uniquely identifies the
// object file output.
void outputSanitizerSettings(llvm::raw_ostream &hash_os) {
  hash_os << SanitizerBits(enabledSanitizers);

  hash_os.write(reinterpret_cast<char *>(&sanitizerCoverageOptions),
                sizeof(sanitizerCoverageOptions));
}

bool functionIsInSanitizerBlacklist(FuncDeclaration *funcDecl) {
  if (!sanitizerBlacklist)
    return false;

  auto funcName = mangleExact(funcDecl);
  auto fileName = funcDecl->loc.filename();

  // TODO: LLVM supports sections (e.g. "[address]") in the blacklist file to
  // only blacklist a function for a particular sanitizer. We could make use of
  // that too.
  return sanitizerBlacklist->inSection(/*Section=*/"", "fun", funcName) ||
         sanitizerBlacklist->inSection(/*Section=*/"", "src", fileName);
}

} // namespace opts
