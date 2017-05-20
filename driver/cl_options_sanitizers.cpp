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

#include "ddmd/errors.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace {

using namespace opts;

cl::list<std::string> fSanitize(
    "fsanitize", cl::CommaSeparated,
    cl::desc("Turn on runtime checks for various forms of undefined or "
             "suspicious behavior."),
    cl::value_desc("checks"));

#ifdef ENABLE_COVERAGE_SANITIZER
cl::list<std::string> fSanitizeCoverage(
    "fsanitize-coverage", cl::CommaSeparated,
    cl::desc("Specify the type of coverage instrumentation for -fsanitize"),
    cl::value_desc("type"));

llvm::SanitizerCoverageOptions sanitizerCoverageOptions;
#endif

// Parse sanitizer name passed on commandline and return the corresponding
// sanitizer bits.
SanitizerCheck parseSanitizerName(llvm::StringRef name) {
  SanitizerCheck parsedValue =
      llvm::StringSwitch<SanitizerCheck>(name)
          .Case("address", AddressSanitizer)
          .Case("fuzzer", FuzzSanitizer)
          .Case("memory", MemorySanitizer)
          .Case("thread", ThreadSanitizer)
          .Default(NoneSanitizer);

  if (parsedValue == NoneSanitizer) {
    error(Loc(), "Unrecognized -fsanitize value '%s'.", name.str().c_str());
  }

  return parsedValue;
}

SanitizerBits parseFSanitizeCmdlineParameter() {
  SanitizerBits retval = 0;
  for (const auto &name : fSanitize) {
    SanitizerCheck check = parseSanitizerName(name);
    retval |= SanitizerBits(check);
  }
  return retval;
}

#ifdef ENABLE_COVERAGE_SANITIZER
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
  else if (name == "8bit-counters") {
    opts.Use8bitCounters = true;
  }
  else if (name == "trace-pc") {
    opts.TracePC = true;
  }
  else if (name == "trace-pc-guard") {
    opts.TracePCGuard = true;
  }
#if LDC_LLVM_VER >= 500
  else if (name == "no-prune") {
    opts.NoPrune = true;
  }
#endif
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
#endif

} // anonymous namespace

namespace opts {

SanitizerBits enabledSanitizers = 0;

void initializeSanitizerOptionsFromCmdline()
{
  enabledSanitizers |= parseFSanitizeCmdlineParameter();

#ifdef ENABLE_COVERAGE_SANITIZER
  auto &sancovOpts = sanitizerCoverageOptions;

  // The Fuzz sanitizer implies -fsanitize-coverage=trace-pc-guard,indirect-calls,trace-cmp
  if (isSanitizerEnabled(FuzzSanitizer)) {
    enabledSanitizers |= CoverageSanitizer;
    sancovOpts.TracePCGuard = true;
    sancovOpts.IndirectCalls = true;
    sancovOpts.TraceCmp = true;
  }

  parseFSanitizeCoverageCmdlineParameter(sancovOpts);

  // trace-pc and trace-pc-guard without specifying the insertion type implies
  // edge
  if ((sancovOpts.CoverageType == llvm::SanitizerCoverageOptions::SCK_None) &&
      (sancovOpts.TracePC || sancovOpts.TracePCGuard)) {
    sancovOpts.CoverageType = llvm::SanitizerCoverageOptions::SCK_Edge;
  }
#endif
}

#ifdef ENABLE_COVERAGE_SANITIZER
llvm::SanitizerCoverageOptions getSanitizerCoverageOptions() {
  return sanitizerCoverageOptions;
}
#endif

// Output to `hash_os` all optimization settings that influence object code
// output and that are not observable in the IR before running LLVM passes. This
// is used to calculate the hash use for caching that uniquely identifies the
// object file output.
void outputSanitizerSettings(llvm::raw_ostream &hash_os) {
  hash_os << SanitizerBits(enabledSanitizers);

#ifdef ENABLE_COVERAGE_SANITIZER
  hash_os << sanitizerCoverageOptions.CoverageType;
  hash_os << sanitizerCoverageOptions.IndirectCalls;
  hash_os << sanitizerCoverageOptions.TraceBB;
  hash_os << sanitizerCoverageOptions.TraceCmp;
  hash_os << sanitizerCoverageOptions.TraceDiv;
  hash_os << sanitizerCoverageOptions.TraceGep;
  hash_os << sanitizerCoverageOptions.Use8bitCounters;
  hash_os << sanitizerCoverageOptions.TracePC;
  hash_os << sanitizerCoverageOptions.TracePCGuard;
#if LDC_LLVM_VER >= 500
  hash_os << sanitizerCoverageOptions.NoPrune;
#endif
#endif
}

} // namespace opts
