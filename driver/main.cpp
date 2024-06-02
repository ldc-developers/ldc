//===-- main.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/compiler.h"
#include "dmd/cond.h"
#include "dmd/errors.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/hdrgen.h"
#include "dmd/json.h"
#include "dmd/ldcbindings.h"
#include "dmd/mars.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/root/rmem.h"
#include "dmd/scope.h"
#include "dmd/target.h"
#include "driver/args.h"
#include "driver/cache.h"
#include "driver/cl_helpers.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/codegenerator.h"
#include "driver/configfile.h"
#include "driver/cpreprocessor.h"
#include "driver/dcomputecodegenerator.h"
#include "driver/exe_path.h"
#include "driver/ldc-version.h"
#include "driver/linker.h"
#include "driver/plugins.h"
#include "driver/targetmachine.h"
#include "driver/timetrace.h"
#include "gen/abi/abi.h"
#include "gen/irstate.h"
#include "gen/ldctraits.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/objcgen.h"
#include "gen/optimizer.h"
#include "gen/passes/metadata.h"
#include "gen/passes/Passes.h"
#include "gen/runtime.h"
#include "gen/uda.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/LinkAllIR.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/FileSystem.h"
#if LDC_LLVM_VER >= 1700
#include "llvm/TargetParser/Host.h"
#else
#include "llvm/Support/Host.h"
#endif
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#if LDC_MLIR_ENABLED
#include "mlir/IR/MLIRContext.h"
#endif
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#if _WIN32
#include "llvm/Support/ConvertUTF.h"
#include <windows.h>
#endif

// In dmd/doc.d
void gendocfile(Module *m);

// In dmd/mars.d
void generateJson(Modules *modules);

using namespace dmd;
using namespace opts;

static StringsAdapter impPathsStore("I", global.params.imppath);
static cl::list<std::string, StringsAdapter>
    importPaths("I", cl::desc("Look for imports also in <directory>"),
                cl::value_desc("directory"), cl::location(impPathsStore),
                cl::Prefix);

// Note: this option is parsed manually in C main().
static cl::opt<bool> enableGC(
    "lowmem", cl::ZeroOrMore,
    cl::desc("Enable the garbage collector for the LDC front-end. This reduces "
             "the compiler memory requirements but increases compile times."));

// Deprecated options
extern llvm::cl::opt<bool> checkPrintf;

namespace {

// This function exits the program.
void printVersion(llvm::raw_ostream &OS) {
  OS << "LDC - the LLVM D compiler (" << ldc::ldc_version << "):\n";
  OS << "  based on DMD " << ldc::dmd_version << " and LLVM "
     << ldc::llvm_version << "\n";
  OS << "  built with " << ldc::built_with_Dcompiler_version << "\n";
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
  OS << "  compiled with address sanitizer enabled\n";
#endif
#endif
  OS << "  Default target: " << llvm::sys::getDefaultTargetTriple() << "\n";
  std::string CPU(llvm::sys::getHostCPUName());
  if (CPU == "generic" || env::has("SOURCE_DATE_EPOCH")) {
    // Env variable SOURCE_DATE_EPOCH indicates that a reproducible build is
    // wanted. Don't print the actual host CPU in such an environment to aid
    // in man page generation etc.
    CPU = "(unknown)";
  }
  OS << "  Host CPU: " << CPU << "\n";
  OS << "  http://dlang.org - http://wiki.dlang.org/LDC\n";
  OS << "\n";

  // Without explicitly flushing here, only the target list is visible when
  // redirecting stdout to a file.
  OS.flush();

  llvm::TargetRegistry::printRegisteredTargetsForVersion(OS);

  exit(EXIT_SUCCESS);
}

// Helper function to handle -d-debug=* and -d-version=*
template <typename Condition>
void processVersions(const std::vector<std::string> &list, const char *type,
                     unsigned &globalLevel) {
  for (const auto &i : list) {
    const char *value = i.c_str();
    if (isdigit(value[0])) {
      errno = 0;
      char *end;
      long level = strtol(value, &end, 10);
      if (*end || errno || level > INT_MAX) {
        error(Loc(), "Invalid %s level: %s", type, i.c_str());
      } else {
        globalLevel = static_cast<unsigned>(level);
      }
    } else {
      char *cstr = mem.xstrdup(value);
      if (Identifier::isValidIdentifier(cstr)) {
        Condition::addGlobalIdent(cstr);
      } else {
        error(Loc(), "Invalid %s identifier or level: '%s'", type, cstr);
      }
    }
  }
}

// Tries to parse the value of `-[-]<option>{=, }<value>` at `args[i]`.
template <int N> // option length incl. terminating null
void tryParse(const llvm::SmallVectorImpl<const char *> &args, size_t i,
              const char *&value, const char (&option)[N]) {
  const char *arg = args[i];
  if (arg[0] != '-')
    return;

  const char *start = arg + (arg[1] == '-' ? 2 : 1);
  if (strncmp(start, option, N - 1) != 0)
    return;

  const char nextChar = start[N - 1];
  if (nextChar == '=')
    value = start + N;
  else if (nextChar == 0 && i < args.size() - 1)
    value = args[i + 1];
}

bool tryParseLowmem(const llvm::SmallVectorImpl<const char *> &args) {
#if LDC_LLVM_VER >= 1800
  #define startswith starts_with
#endif

  bool lowmem = false;
  for (size_t i = 1; i < args.size(); ++i) {
    if (args::isRunArg(args[i]))
      break;

    llvm::StringRef arg = args[i];
    if (arg.startswith("-lowmem") || arg.startswith("--lowmem")) {
      auto remainder = arg.substr(arg[1] == '-' ? 8 : 7);
      if (remainder.empty()) {
        lowmem = true;
      } else if (remainder[0] == '=') {
        auto value = remainder.substr(1);
        lowmem = (value == "true" || value == "TRUE");
      }
    }
  }
  return lowmem;

#if LDC_LLVM_VER >= 1800
  #undef startswith
#endif
}

const char *
tryGetExplicitConfFile(const llvm::SmallVectorImpl<const char *> &args) {
  const char *conf = nullptr;
  for (size_t i = 1; i < args.size(); ++i) {
    if (args::isRunArg(args[i]))
      break;
    tryParse(args, i, conf, "conf");
  }
  return conf;
}

llvm::Triple
tryGetExplicitTriple(const llvm::SmallVectorImpl<const char *> &args) {
  // most combinations of flags are illegal, this mimicks command line
  //  behaviour for legal ones only
  llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
  unsigned explicitBitness = 0;
  const char *mtriple = nullptr;
  const char *march = nullptr;
  for (size_t i = 1; i < args.size(); ++i) {
    if (args::isRunArg(args[i]))
      break;

    const llvm::StringRef arg = args[i];
    if (arg == "-m32" || arg == "--m32") {
      explicitBitness = 32;
    } else if (arg == "-m64" || arg == "--m64") {
      explicitBitness = 64;
    } else {
      tryParse(args, i, mtriple, "mtriple");
      tryParse(args, i, march, "march");
    }
  }

  // -m{32,64} and -mtriple/-march are mutually exclusive (checked later on)
  if (explicitBitness) {
    if (!triple.isArch32Bit() && explicitBitness == 32) {
      triple = triple.get32BitArchVariant();
      if (triple.getArch() == llvm::Triple::ArchType::x86)
        triple.setArchName("i686"); // instead of i386
    } else if (!triple.isArch64Bit() && explicitBitness == 64) {
      triple = triple.get64BitArchVariant();
    }
    return triple;
  }

  if (mtriple)
    triple = llvm::Triple(llvm::Triple::normalize(mtriple));
  if (march) {
    std::string errorMsg; // ignore error, will show up later anyway
    lookupTarget(march, triple, errorMsg); // modifies triple
  }
  return triple;
}

/// Parses switches from the command line, any response files and the global
/// config file and sets up global.params accordingly.
///
/// Returns a list of source file names.
void parseCommandLine(Strings &sourceFiles) {
  const auto &exePath = exe_path::getExePath();
  global.params.argv0 = {exePath.length(), exePath.data()};

  // read config file
  ConfigFile &cfg_file = ConfigFile::instance;
  const char *explicitConfFile = tryGetExplicitConfFile(allArguments);
  const std::string cfg_triple = tryGetExplicitTriple(allArguments).getTriple();
  // just ignore errors for now, they are still printed
  cfg_file.read(explicitConfFile, cfg_triple.c_str());

  cfg_file.extendCommandLine(allArguments);

  // finalize by expanding response files specified in config file
  args::expandResponseFiles(allArguments);

  cl::SetVersionPrinter(&printVersion);

  opts::hideLLVMOptions();
  opts::createClashingOptions();

  // Filter out druntime options in the cmdline, e.g., to configure the GC.
  std::vector<const char *> filteredArgs;
  filteredArgs.reserve(allArguments.size());
  for (size_t i = 0; i < allArguments.size(); ++i) {
    const char *arg = allArguments[i];
    if (args::isRunArg(arg)) {
      filteredArgs.insert(filteredArgs.end(), allArguments.begin() + i,
                          allArguments.end());
      break;
    }
    if (strncmp(arg, "--DRT-", 6) != 0)
      filteredArgs.push_back(arg);
  }

  cl::ParseCommandLineOptions(filteredArgs.size(),
                              const_cast<char **>(filteredArgs.data()),
                              "LDC - the LLVM D compiler\n");

  if (opts::printTargetFeaturesHelp()) {
    auto triple = llvm::Triple(cfg_triple);
    std::string errMsg;
    if (auto target = lookupTarget("", triple, errMsg)) {
      llvm::errs() << "Targeting " << target->getName() << ". ";
      // this prints the available CPUs and features of the target to stderr...
      target->createMCSubtargetInfo(cfg_triple, "help", "");
    } else {
      error(Loc(), "%s", errMsg.c_str());
      fatal();
    }
    exit(EXIT_SUCCESS);
  }

  if (!cfg_file.path().empty())
    global.inifilename = dupPathString(cfg_file.path());

  // Print some information if -v was passed
  // - path to compiler binary
  // - version number
  // - used config file
  if (global.params.v.verbose) {
    message("binary    %s", exe_path::getExePath().c_str());
    message("version   %s (DMD %s, LLVM %s)", ldc::ldc_version,
            ldc::dmd_version, ldc::llvm_version);
    if (global.inifilename.length) {
      message("config    %.*s (%s)", (int)global.inifilename.length,
              global.inifilename.ptr, cfg_triple.c_str());
    }
  }

  // Negated options
  global.params.link = !compileOnly;
  global.params.obj = !dontWriteObj;
  global.params.useInlineAsm = !noAsm;
  global.params.useExceptions = !fNoExceptions;
  global.params.useModuleInfo = !fNoModuleInfo;
  global.params.useTypeInfo = !fNoRTTI;

  // String options
  global.params.objname = opts::fromPathString(objectFile);
  global.params.objdir = opts::fromPathString(objectDir);

  global.params.ddoc.dir = opts::fromPathString(ddocDir);
  global.params.ddoc.name = opts::fromPathString(ddocFile);
  global.params.ddoc.doOutput |=
      global.params.ddoc.dir.length || global.params.ddoc.name.length;

  global.params.json.name = opts::fromPathString(jsonFile);
  if (global.params.json.name.length || !jsonFields.empty()) {
    global.params.json.doOutput = true;
  }

  global.params.dihdr.dir = opts::fromPathString(hdrDir);
  global.params.dihdr.name = opts::fromPathString(hdrFile);
  global.params.dihdr.doOutput |=
      global.params.dihdr.dir.length || global.params.dihdr.name.length;

  global.params.cxxhdr.dir = opts::fromPathString(cxxHdrDir);
  global.params.cxxhdr.name = opts::fromPathString(cxxHdrFile);
  global.params.cxxhdr.doOutput |=
      global.params.cxxhdr.dir.length || global.params.cxxhdr.name.length;

  global.params.mixinOut.name = opts::fromPathString(mixinFile);
  global.params.mixinOut.doOutput |= global.params.mixinOut.name.length;

  if (moduleDeps.getNumOccurrences() != 0) {
    global.params.moduleDeps.doOutput = true;
    global.params.moduleDeps.buffer = createOutBuffer();
    if (!moduleDeps.empty())
      global.params.moduleDeps.name = opts::dupPathString(moduleDeps);
  }

  if (makeDeps.getNumOccurrences() != 0) {
    global.params.makeDeps.doOutput = true;
    if (!makeDeps.empty())
      global.params.makeDeps.name = opts::dupPathString(makeDeps);
  }

#if _WIN32
  const auto toWinPaths = [](Strings &paths) {
    for (auto &path : paths)
      path = opts::dupPathString(path).ptr;
  };
  toWinPaths(global.params.imppath);
  toWinPaths(global.params.fileImppath);
#endif

  for (const auto &field : jsonFields) {
    const unsigned flag = tryParseJsonField(field.c_str());
    if (flag == 0) {
      error(Loc(), "unknown JSON field `-Xi=%s`", field.c_str());
    } else {
      global.params.jsonFieldFlags |= flag;
    }
  }

  includeImports = !opts::includeModulePatterns.empty();
  for (const auto &pattern : opts::includeModulePatterns) {
    // a value-less `-i` only enables `includeImports`
    if (!pattern.empty())
      ::includeModulePatterns.push_back(pattern.c_str());
  }
  // When including imports, their object files aren't tracked in
  // global.params.objfiles etc. Enforce `-singleobj` to avoid related issues.
  if (includeImports)
    global.params.oneobj = true;

  if (saveOptimizationRecord.getNumOccurrences() > 0) {
    global.params.outputSourceLocations = true;
  }

  opts::initializeSanitizerOptionsFromCmdline();

  processVersions<DebugCondition>(debugArgs, "debug", global.params.debuglevel);
  processVersions<VersionCondition>(versions, "version",
                                    global.params.versionlevel);

  for (const auto &id : transitions)
    parseTransitionOption(global.params, id.c_str());
  for (const auto &id : previews)
    parsePreviewOption(global.params, id.c_str());
  for (const auto &id : reverts)
    parseRevertOption(global.params, id.c_str());

  if (global.params.useDIP1021) // DIP1021 implies DIP1000
    global.params.useDIP1000 = FeatureState::enabled;
  // legacy -dip1000 option
  if (global.params.useDIP1000 == FeatureState::default_ &&
      opts::useDIP1000.getNumOccurrences()) {
    global.params.useDIP1000 =
        opts::useDIP1000 ? FeatureState::enabled : FeatureState::disabled;
  }

  if (global.params.useDIP1000 == FeatureState::enabled) // DIP1000 implies DIP25
    global.params.useDIP25 = FeatureState::enabled;
  // legacy -dip25 option
  if (opts::useDIP25.getNumOccurrences()) {
    deprecation(Loc(), "`-dip25` no longer has any effect");
  }

  // -betterC implies -allinst (since D v2.105)
  if (global.params.betterC) {
    global.params.allInst = true;
  }

  global.params.output_o =
      (opts::output_o == cl::BOU_UNSET &&
       !(opts::output_bc || opts::output_ll || opts::output_s ||
         opts::output_mlir))
          ? OUTPUTFLAGdefault
          : opts::output_o == cl::BOU_TRUE ? OUTPUTFLAGset : OUTPUTFLAGno;
  global.params.output_bc = opts::output_bc ? OUTPUTFLAGset : OUTPUTFLAGno;
  global.params.output_ll = opts::output_ll ? OUTPUTFLAGset : OUTPUTFLAGno;
  global.params.output_mlir = opts::output_mlir ? OUTPUTFLAGset : OUTPUTFLAGno;
  global.params.output_s = opts::output_s ? OUTPUTFLAGset : OUTPUTFLAGno;

  templateLinkage = global.params.linkonceTemplates == LinkonceTemplates::no
                        ? LLGlobalValue::WeakODRLinkage
                        : LLGlobalValue::LinkOnceODRLinkage;

  if (global.params.run || !runargs.empty()) {
    // FIXME: how to properly detect the presence of a PositionalEatsArgs
    // option without parameters? We want to emit an error in that case...
    // You'd think getNumOccurrences would do it, but it just returns the
    // number of parameters)
    // NOTE: Hacked around it by detecting -run in getenv_setargv(), where
    // we're looking for it anyway, and pre-setting the flag...
    global.params.run = true;
    if (!runargs.empty()) {
      if (runargs[0] == "-") {
        sourceFiles.push("__stdin.d");
      } else {
        char const *name = runargs[0].c_str();
        char const *ext = FileName::ext(name);
        if (ext && !FileName::equals(ext, mars_ext.ptr) &&
            !FileName::equals(ext, hdr_ext.ptr) &&
            !FileName::equals(ext, i_ext.ptr) &&
            !FileName::equals(ext, c_ext.ptr)) {
          error(Loc(), "-run must be followed by a source file, not '%s'",
                name);
        }
        sourceFiles.push(mem.xstrdup(name));
      }
      runargs.erase(runargs.begin());
    } else {
      global.params.run = false;
      error(Loc(), "Expected at least one argument to '-run'\n");
    }

    // -run: enforce -oq and -cleanup-obj for non-conflicting temporary objects
    if (global.params.run) {
      global.params.fullyQualifiedObjectFiles = true;
      global.params.cleanupObjectFiles = true;
    }
  }

  sourceFiles.reserve(fileList.size());
  for (const auto &file : fileList) {
    if (!file.empty()) {
      sourceFiles.push(file == "-" ? "__stdin.d"
                                   : opts::dupPathString(file).ptr);
    }
  }

  // LDC output determination

  // if we don't link and there's no `-output-*` switch but an `-of` one,
  // autodetect type of desired 'object' file from file extension
  if (!global.params.link && !global.params.lib &&
      global.params.objname.length &&
      global.params.output_o == OUTPUTFLAGdefault) {
    const char *ext = FileName::ext(global.params.objname.ptr);
    if (!ext) {
      // keep things as they are
    } else if (opts::output_ll.getNumOccurrences() == 0 &&
               strcmp(ext, ll_ext.ptr) == 0) {
      global.params.output_ll = OUTPUTFLAGset;
      global.params.output_o = OUTPUTFLAGno;
    } else if (opts::output_bc.getNumOccurrences() == 0 &&
               strcmp(ext, bc_ext.ptr) == 0) {
      global.params.output_bc = OUTPUTFLAGset;
      global.params.output_o = OUTPUTFLAGno;
    } else if (opts::output_s.getNumOccurrences() == 0 &&
               strcmp(ext, s_ext.ptr) == 0) {
      global.params.output_s = OUTPUTFLAGset;
      global.params.output_o = OUTPUTFLAGno;
    } else if (opts::output_mlir.getNumOccurrences() == 0 &&
               strcmp(ext, mlir_ext.ptr) == 0) {
      global.params.output_mlir = OUTPUTFLAGset;
      global.params.output_o = OUTPUTFLAGno;
    }
  }

#ifndef LDC_MLIR_ENABLED
  if (global.params.output_mlir == OUTPUTFLAGset) {
    error(Loc(), "MLIR output requested but this LDC was built without MLIR support");
    fatal();
  }
#endif

  if (soname.getNumOccurrences() > 0 && !global.params.dll) {
    error(Loc(), "-soname can be used only when building a shared library");
  }

  global.params.dihdr.fullOutput = opts::hdrKeepAllBodies;
  global.params.disableRedZone = opts::disableRedZone();

  // enforce opaque IR pointers
#if LDC_LLVM_VER >= 1700
  // supports opaque IR pointers only
#else
  getGlobalContext().setOpaquePointers(true);
#endif
}

void initializePasses() {
  using namespace llvm;
  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeTransformUtils(Registry);
  initializeScalarOpts(Registry);
#if LDC_LLVM_VER < 1600
  initializeObjCARCOpts(Registry);
#endif
  initializeVectorization(Registry);
  initializeInstCombine(Registry);
#if LDC_LLVM_VER < 1600
  initializeAggressiveInstCombine(Registry);
#endif
  initializeIPO(Registry);
#if LDC_LLVM_VER < 1600
  initializeInstrumentation(Registry);
#endif
  initializeAnalysis(Registry);
  initializeCodeGen(Registry);
  initializeGlobalISel(Registry);
  initializeTarget(Registry);

#if LDC_LLVM_VER < 1700
// Initialize passes not included above
  initializeRewriteSymbolsLegacyPassPass(Registry);
  initializeSjLjEHPreparePass(Registry);
#endif
}

/// Register the MIPS ABI.
static void registerMipsABI() {
  switch (getMipsABI()) {
  case MipsABI::EABI:
    VersionCondition::addPredefinedGlobalIdent("MIPS_EABI");
    break;
  case MipsABI::O32:
    VersionCondition::addPredefinedGlobalIdent("MIPS_O32");
    break;
  case MipsABI::N32:
    VersionCondition::addPredefinedGlobalIdent("MIPS_N32");
    break;
  case MipsABI::N64:
    VersionCondition::addPredefinedGlobalIdent("MIPS_N64");
    break;
  case MipsABI::Unknown:
    break;
  }
}

// Handle target environments not directly supported by LLVM.
void fixupTripleEnv(std::string &tripleString) {
  size_t i;
  if ((i = tripleString.find("-uclibc")) != std::string::npos) {
    global.params.isUClibcEnvironment = true;
    tripleString.replace(i + 1, 6, "gnu"); // -uclibceabihf => -gnueabihf
  } else if ((i = tripleString.find("-newlib")) != std::string::npos) {
    global.params.isNewlibEnvironment = true;
    tripleString.replace(i + 1, 6, ""); // -newlibeabi => -eabi
  }
}

/// Register the float ABI.
/// Also defines D_HardFloat or D_SoftFloat depending if FPU should be used
void registerPredefinedFloatABI(const char *soft, const char *hard,
                                const char *softfp = nullptr) {
  switch (floatABI) {
  case FloatABI::Soft:
    VersionCondition::addPredefinedGlobalIdent(soft);
    break;
  case FloatABI::SoftFP:
    VersionCondition::addPredefinedGlobalIdent(softfp ? softfp : soft);
    break;
  case FloatABI::Hard:
    VersionCondition::addPredefinedGlobalIdent(hard);
    break;
  default:
    llvm_unreachable("Unknown float ABI");
  }

  VersionCondition::addPredefinedGlobalIdent(
      floatABI == FloatABI::Soft ? "D_SoftFloat" : "D_HardFloat");
}

/// Registers the predefined versions specific to the current target triple
/// and other target specific options with VersionCondition.
void registerPredefinedTargetVersions() {
  const auto &triple = *global.params.targetTriple;
  const auto arch = triple.getArch();

  switch (arch) {
  case llvm::Triple::x86:
    VersionCondition::addPredefinedGlobalIdent("X86");
    if (global.params.useInlineAsm) {
      VersionCondition::addPredefinedGlobalIdent("D_InlineAsm_X86");
    }
    VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
    break;
  case llvm::Triple::x86_64:
    VersionCondition::addPredefinedGlobalIdent("X86_64");
    if (global.params.useInlineAsm) {
      VersionCondition::addPredefinedGlobalIdent("D_InlineAsm_X86_64");
    }
    VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
    break;
  case llvm::Triple::ppc:
    VersionCondition::addPredefinedGlobalIdent("PPC");
    registerPredefinedFloatABI("PPC_SoftFloat", "PPC_HardFloat");
    break;
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    VersionCondition::addPredefinedGlobalIdent("PPC64");
    registerPredefinedFloatABI("PPC_SoftFloat", "PPC_HardFloat");
    if (triple.getOS() == llvm::Triple::Linux) {
      VersionCondition::addPredefinedGlobalIdent(
          triple.getArch() == llvm::Triple::ppc64 ? "ELFv1" : "ELFv2");
    }
    break;
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
    VersionCondition::addPredefinedGlobalIdent("ARM");
    registerPredefinedFloatABI("ARM_SoftFloat", "ARM_HardFloat", "ARM_SoftFP");
    break;
  case llvm::Triple::thumb:
    VersionCondition::addPredefinedGlobalIdent("ARM");
    VersionCondition::addPredefinedGlobalIdent(
        "Thumb"); // For backwards compatibility.
    VersionCondition::addPredefinedGlobalIdent("ARM_Thumb");
    registerPredefinedFloatABI("ARM_SoftFloat", "ARM_HardFloat", "ARM_SoftFP");
    break;
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    VersionCondition::addPredefinedGlobalIdent("AArch64");
    registerPredefinedFloatABI("ARM_SoftFloat", "ARM_HardFloat", "ARM_SoftFP");
    break;
  case llvm::Triple::avr:
    VersionCondition::addPredefinedGlobalIdent("AVR");
    VersionCondition::addPredefinedGlobalIdent("D_SoftFloat");
    break;
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    VersionCondition::addPredefinedGlobalIdent("MIPS");
    VersionCondition::addPredefinedGlobalIdent("MIPS32");
    registerPredefinedFloatABI("MIPS_SoftFloat", "MIPS_HardFloat");
    registerMipsABI();
    break;
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    VersionCondition::addPredefinedGlobalIdent("MIPS64");
    registerPredefinedFloatABI("MIPS_SoftFloat", "MIPS_HardFloat");
    registerMipsABI();
    break;
  case llvm::Triple::msp430:
    VersionCondition::addPredefinedGlobalIdent("MSP430");
    break;
  case llvm::Triple::riscv32:
    VersionCondition::addPredefinedGlobalIdent("RISCV32");
    break;
  case llvm::Triple::riscv64:
    VersionCondition::addPredefinedGlobalIdent("RISCV64");
    break;
  case llvm::Triple::sparc:
    // FIXME: Detect SPARC v8+ (SPARC_V8Plus).
    VersionCondition::addPredefinedGlobalIdent("SPARC");
    registerPredefinedFloatABI("SPARC_SoftFloat", "SPARC_HardFloat");
    break;
  case llvm::Triple::sparcv9:
    VersionCondition::addPredefinedGlobalIdent("SPARC64");
    registerPredefinedFloatABI("SPARC_SoftFloat", "SPARC_HardFloat");
    break;
  case llvm::Triple::nvptx:
    VersionCondition::addPredefinedGlobalIdent("NVPTX");
    VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
    break;
  case llvm::Triple::nvptx64:
    VersionCondition::addPredefinedGlobalIdent("NVPTX64");
    VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
    break;
  case llvm::Triple::systemz:
    VersionCondition::addPredefinedGlobalIdent("SystemZ");
    VersionCondition::addPredefinedGlobalIdent(
        "S390X"); // For backwards compatibility.
    VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
    break;
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    VersionCondition::addPredefinedGlobalIdent("WebAssembly");
    break;
#if LDC_LLVM_VER >= 1600
  case llvm::Triple::loongarch32:
    VersionCondition::addPredefinedGlobalIdent("LoongArch32");
    registerPredefinedFloatABI("LoongArch_SoftFloat", "LoongArch_HardFloat");
    break;
  case llvm::Triple::loongarch64:
    VersionCondition::addPredefinedGlobalIdent("LoongArch64");
    registerPredefinedFloatABI("LoongArch_SoftFloat", "LoongArch_HardFloat");
    break;
#endif // LDC_LLVM_VER >= 1600
  default:
    warning(Loc(), "unknown target CPU architecture: %s",
            triple.getArchName().str().c_str());
  }

  // endianness
  if (gDataLayout->isLittleEndian()) {
    VersionCondition::addPredefinedGlobalIdent("LittleEndian");
  } else {
    VersionCondition::addPredefinedGlobalIdent("BigEndian");
  }

  // Set versions for arch bitwidth
  if (gDataLayout->getPointerSizeInBits() == 64) {
    VersionCondition::addPredefinedGlobalIdent("D_LP64");
  } else if (triple.isArch64Bit()) {
    VersionCondition::addPredefinedGlobalIdent("D_X32");
  } else if (triple.isArch16Bit()) {
    VersionCondition::addPredefinedGlobalIdent("D_P16");
  }

  if (gTargetMachine->getRelocationModel() == llvm::Reloc::PIC_) {
    VersionCondition::addPredefinedGlobalIdent("D_PIC");
  }

  if (arch == llvm::Triple::x86 || arch == llvm::Triple::x86_64) {
    /* LDC doesn't support DMD's core.simd interface.
    if (traitsTargetHasFeature("sse2"))
      VersionCondition::addPredefinedGlobalIdent("D_SIMD");
    */
    if (traitsTargetHasFeature("avx"))
      VersionCondition::addPredefinedGlobalIdent("D_AVX");
    if (traitsTargetHasFeature("avx2"))
      VersionCondition::addPredefinedGlobalIdent("D_AVX2");
  }

  // parse the OS out of the target triple
  // see http://gcc.gnu.org/install/specific.html for details
  // also llvm's different SubTargets have useful information
  switch (triple.getOS()) {
  case llvm::Triple::Win32:
    VersionCondition::addPredefinedGlobalIdent("Windows");
    VersionCondition::addPredefinedGlobalIdent(triple.isArch64Bit() ? "Win64"
                                                                    : "Win32");
    if (triple.isWindowsMSVCEnvironment()) {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Microsoft");
      VersionCondition::addPredefinedGlobalIdent("CppRuntime_Microsoft");
    }
    if (triple.isWindowsGNUEnvironment()) {
      VersionCondition::addPredefinedGlobalIdent(
          "mingw32"); // For backwards compatibility.
      VersionCondition::addPredefinedGlobalIdent("MinGW");
    }
    if (triple.isWindowsCygwinEnvironment()) {
      error(Loc(), "Cygwin is not yet supported");
      fatal();
      VersionCondition::addPredefinedGlobalIdent("Cygwin");
    }
    break;
  case llvm::Triple::Linux:
    VersionCondition::addPredefinedGlobalIdent("linux");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    if (triple.getEnvironment() == llvm::Triple::Android) {
      VersionCondition::addPredefinedGlobalIdent("Android");
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Bionic");
      VersionCondition::addPredefinedGlobalIdent("CppRuntime_Clang");
    } else if (triple.isMusl()) {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Musl");
      VersionCondition::addPredefinedGlobalIdent("CppRuntime_Gcc");
      // use libunwind for backtraces
      VersionCondition::addPredefinedGlobalIdent("DRuntime_Use_Libunwind");
    } else if (global.params.isUClibcEnvironment) {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_UClibc");
    } else {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Glibc");
      VersionCondition::addPredefinedGlobalIdent("CppRuntime_Gcc");
    }
    break;
  case llvm::Triple::Haiku:
    VersionCondition::addPredefinedGlobalIdent("Haiku");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
    VersionCondition::addPredefinedGlobalIdent("OSX");
    VersionCondition::addPredefinedGlobalIdent(
        "darwin"); // For backwards compatibility.
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("CppRuntime_Clang");
    break;
  case llvm::Triple::FreeBSD:
    VersionCondition::addPredefinedGlobalIdent("FreeBSD");
    if (unsigned major = triple.getOSMajorVersion()) {
      const auto withMajor = "FreeBSD_" + std::to_string(major);
      VersionCondition::addPredefinedGlobalIdent(withMajor.c_str());
    } else {
      warning(Loc(), "FreeBSD major version not specified in target triple");
    }
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("CppRuntime_Clang");
    break;
  case llvm::Triple::Solaris:
    VersionCondition::addPredefinedGlobalIdent("Solaris");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("CppRuntime_Sun");
    break;
  case llvm::Triple::DragonFly:
    VersionCondition::addPredefinedGlobalIdent("DragonFlyBSD");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("CppRuntime_Gcc");
    break;
  case llvm::Triple::NetBSD:
    VersionCondition::addPredefinedGlobalIdent("NetBSD");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  case llvm::Triple::OpenBSD:
    VersionCondition::addPredefinedGlobalIdent("OpenBSD");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("CppRuntime_Gcc");
    break;
  case llvm::Triple::AIX:
    VersionCondition::addPredefinedGlobalIdent("AIX");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  case llvm::Triple::IOS:
    VersionCondition::addPredefinedGlobalIdent("iOS");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("CppRuntime_Clang");
    break;
  case llvm::Triple::TvOS:
    VersionCondition::addPredefinedGlobalIdent("TVOS");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("CppRuntime_Clang");
    break;
  case llvm::Triple::WatchOS:
    VersionCondition::addPredefinedGlobalIdent("WatchOS");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    VersionCondition::addPredefinedGlobalIdent("CppRuntime_Clang");
    break;
  case llvm::Triple::WASI:
    VersionCondition::addPredefinedGlobalIdent("WASI");
    VersionCondition::addPredefinedGlobalIdent("CRuntime_WASI");
    break;
  case llvm::Triple::Emscripten:
    VersionCondition::addPredefinedGlobalIdent("Emscripten");
    break;
  default:
    if (triple.getEnvironment() == llvm::Triple::Android) {
      VersionCondition::addPredefinedGlobalIdent("Android");
    } else {
      llvm::StringRef osName = triple.getOSName();
      if (osName.empty() || osName == "unknown" || osName == "none") {
        VersionCondition::addPredefinedGlobalIdent("FreeStanding");
        if (global.params.isNewlibEnvironment) {
            VersionCondition::addPredefinedGlobalIdent("CRuntime_Newlib");
        }
      } else {
        warning(Loc(), "unknown target OS: %s", osName.str().c_str());
      }
    }
    break;
  }

  //Issue deprecations for deprecated arguments
  // N.B `-nodefaultlib` is checked in `getDefaultLibNames`
  if (checkPrintf)
    deprecation(Loc(), "'-check-printf-calls' is deprecated, use `pragma(printf)` instead.");
}

} // anonymous namespace

/// Registers all predefined D version identifiers for the current
/// configuration with VersionCondition.
void registerPredefinedVersions() {
  VersionCondition::addPredefinedGlobalIdent("LDC");
  VersionCondition::addPredefinedGlobalIdent("all");
  VersionCondition::addPredefinedGlobalIdent("D_Version2");

#if LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX
  if (dcomputeTargets.size() != 0) {
    VersionCondition::addPredefinedGlobalIdent("LDC_DCompute");
  }
#endif

  if (global.params.ddoc.doOutput) {
    VersionCondition::addPredefinedGlobalIdent("D_Ddoc");
  }

  if (global.params.cov) {
    VersionCondition::addPredefinedGlobalIdent("D_Coverage");
  }

  if (global.params.useUnitTests) {
    VersionCondition::addPredefinedGlobalIdent("unittest");
  }

  if (global.params.useAssert == CHECKENABLEon) {
    VersionCondition::addPredefinedGlobalIdent("assert");
  }

  if (global.params.useIn == CHECKENABLEon) {
    VersionCondition::addPredefinedGlobalIdent("D_PreConditions");
  }

  if (global.params.useOut == CHECKENABLEon) {
    VersionCondition::addPredefinedGlobalIdent("D_PostConditions");
  }

  if (global.params.useInvariants == CHECKENABLEon) {
    VersionCondition::addPredefinedGlobalIdent("D_Invariants");
  }

  if (global.params.useArrayBounds == CHECKENABLEoff) {
    VersionCondition::addPredefinedGlobalIdent("D_NoBoundsChecks");
  }

  if (global.params.betterC) {
    VersionCondition::addPredefinedGlobalIdent("D_BetterC");
  } else {
    if (global.params.useModuleInfo)
      VersionCondition::addPredefinedGlobalIdent("D_ModuleInfo");
    if (global.params.useExceptions)
      VersionCondition::addPredefinedGlobalIdent("D_Exceptions");
    if (global.params.useTypeInfo)
      VersionCondition::addPredefinedGlobalIdent("D_TypeInfo");
  }

  if (global.params.tracegc) {
    VersionCondition::addPredefinedGlobalIdent("D_ProfileGC");
  }

  if (isOptimizationEnabled()) {
    VersionCondition::addPredefinedGlobalIdent("D_Optimized");
  }

  registerPredefinedTargetVersions();

  // `D_ObjectiveC` is added by the dmd.objc.Supported ctor

  if (opts::enableDynamicCompile) {
    VersionCondition::addPredefinedGlobalIdent("LDC_DynamicCompilation");
  }

  // Define sanitizer versions.
  if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
    VersionCondition::addPredefinedGlobalIdent("LDC_AddressSanitizer");
  }
  if (opts::isSanitizerEnabled(opts::CoverageSanitizer)) {
    VersionCondition::addPredefinedGlobalIdent("LDC_CoverageSanitizer");
  }
  if (opts::isSanitizerEnabled(opts::MemorySanitizer)) {
    VersionCondition::addPredefinedGlobalIdent("LDC_MemorySanitizer");
  }
  if (opts::isSanitizerEnabled(opts::ThreadSanitizer)) {
    VersionCondition::addPredefinedGlobalIdent("LDC_ThreadSanitizer");
  }

// Expose LLVM version to runtime
#define STR(x) #x
#define XSTR(x) STR(x)
  VersionCondition::addPredefinedGlobalIdent("LDC_LLVM_" XSTR(LDC_LLVM_VER));
#undef XSTR
#undef STR
}

static llvm::SmallString<32> tempObjectsDir;

// With -cleanup-obj, potentially invoked (once) by Module.setOutfilename() to
// create a unique temporary directory for generated object files (if -od hasn't
// been specified), e.g., `/tmp/objtmp-ldc-688445`.
const char *createTempObjectsDir() {
  assert(tempObjectsDir.empty());

  auto ec = llvm::sys::fs::createUniqueDirectory("objtmp-ldc", tempObjectsDir);
  if (ec) {
    error(Loc(),
          "failed to create temporary directory for object files: %s\n%s",
          tempObjectsDir.c_str(), ec.message().c_str());
    fatal();
  }

  return tempObjectsDir.c_str();
}

/// LDC's entry point, C main.
/// Without `-lowmem`, we need to switch to the bump-pointer allocation scheme
/// right from the start, before any module ctors are run, so we need this hook
/// before druntime is initialized and `_Dmain` is called.
#if LDC_WINDOWS_WMAIN
int wmain(int argc, const wchar_t **originalArgv)
#else
int main(int argc, const char **originalArgv)
#endif
{
  // initialize `opts::allArguments` with the UTF-8 command-line args
  args::getCommandLineArguments(argc, originalArgv, allArguments);

  llvm::sys::PrintStackTraceOnErrorSignal(allArguments[0]);

  // expand response files (`@<file>`, e.g., used by dub) in-place
  args::expandResponseFiles(allArguments);

  if (!tryParseLowmem(allArguments))
    mem.disableGC();

  // Only pass --DRT-* options (before a first potential -run) to _d_run_main;
  // we don't need any args for _Dmain.
  llvm::SmallVector<const args::CArgChar *, 4> drunmainArgs;
  drunmainArgs.push_back(originalArgv[0]);
  for (size_t i = 1; i < allArguments.size(); ++i) {
    const char *arg = allArguments[i];
    if (args::isRunArg(arg))
      break;
    if (strncmp(arg, "--DRT-", 6) == 0) {
#if LDC_WINDOWS_WMAIN
      // cannot use originalArgv, as the arg may originate from a response file
      llvm::SmallVector<wchar_t, 64> warg;
      llvm::sys::windows::UTF8ToUTF16(arg, warg);
      warg.push_back(0);
      drunmainArgs.push_back(wcsdup(warg.data()));
#else
      drunmainArgs.push_back(arg);
#endif
    }
  }

  // move on to _d_run_main, _Dmain, and finally cppmain below
  return args::forwardToDruntime(drunmainArgs.size(), drunmainArgs.data());
}

int cppmain() {
  exe_path::initialize(allArguments[0]);

  global._init();
  global.ldc_version = {strlen(ldc::ldc_version), ldc::ldc_version};
  global.llvm_version = {strlen(ldc::llvm_version), ldc::llvm_version};

  // Initialize LLVM before parsing the command line so that --version shows
  // registered targets.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  initializePasses();

  Strings files;
  parseCommandLine(files);

  if (allArguments.size() == 1) {
    cl::PrintHelpMessage(/*Hidden=*/false, /*Categorized=*/true);
    exit(EXIT_FAILURE);
  }

  if (global.errors) {
    fatal();
  }

  if (opts::fTimeTrace) {
    initializeTimeTrace(opts::fTimeTraceGranularity, 0, opts::allArguments[0]);
  }

  // Set up the TargetMachine.
  const auto arch = getArchStr();
  if ((m32bits || m64bits) && (!arch.empty() || !mTargetTriple.empty())) {
    error(Loc(), "-m32 and -m64 switches cannot be used together with -march "
                 "and -mtriple switches");
  }

  ExplicitBitness::Type bitness = ExplicitBitness::None;
  if (m32bits)
    bitness = ExplicitBitness::M32;
  if (m64bits && (!m32bits || m32bits.getPosition() < m64bits.getPosition()))
    bitness = ExplicitBitness::M64;

  if (global.errors) {
    fatal();
  }

  auto relocModel = getRelocModel();
#if LDC_LLVM_VER >= 1600
  if (global.params.dll && !relocModel.has_value()) {
    relocModel = llvm::Reloc::PIC_;
  }
#else
  if (global.params.dll && !relocModel.hasValue()) {
    relocModel = llvm::Reloc::PIC_;
  }
#endif

  fixupTripleEnv(mTargetTriple);

  // create target machine and finalize floatABI
  gTargetMachine = createTargetMachine(
      mTargetTriple, arch, opts::getCPUStr(), opts::getFeaturesStr(), bitness,
      floatABI, relocModel, opts::getCodeModel(), codeGenOptLevel(),
      disableLinkerStripDead);

  opts::setDefaultMathOptions(gTargetMachine->Options);

  static llvm::DataLayout DL = gTargetMachine->createDataLayout();
  gDataLayout = &DL;

  llvm::Triple *triple = new llvm::Triple(gTargetMachine->getTargetTriple());
  global.params.targetTriple = triple;

  global.params.dwarfVersion = gTargetMachine->Options.MCOptions.DwarfVersion;

  // -gdwarf implies -g if not specified explicitly
  if (opts::emitDwarfDebugInfo && global.params.symdebug == 0) {
    global.params.symdebug = 1;
  }

  if (triple->isOSWindows()) {
    const auto v = opts::symbolVisibility.getValue();
    global.params.dllexport =
        v == opts::SymbolVisibility::public_ ||
        // default with -shared
        (v == opts::SymbolVisibility::default_ && global.params.dll);
    if (opts::dllimport.getNumOccurrences() == 0) {
      global.params.dllimport =
          !linkAgainstSharedDefaultLibs() ? DLLImport::none
          : global.params.dllexport       ? DLLImport::all
                                          : DLLImport::defaultLibsOnly;
    }
  } else {
    global.params.dllexport = false;
    global.params.dllimport = DLLImport::none;
  }

  global.preprocess = &runCPreprocessor;

  // allocate the target abi
  gABI = TargetABI::getTarget();

  opts::initializeInstrumentationOptionsFromCmdline(*triple);

  loadAllPlugins();

  int status;
  {
    TimeTraceScope timeScope("ExecuteCompiler");
    status = mars_tryMain(global.params, files);
  }

  // try to remove the temp objects dir if created for -cleanup-obj
  if (!tempObjectsDir.empty())
    llvm::sys::fs::remove(tempObjectsDir);

  std::string fTimeTraceFile = opts::fTimeTraceFile;
  writeTimeTraceProfile(fTimeTraceFile.empty() ? "" : fTimeTraceFile.c_str());
  deinitializeTimeTrace();

  llvm::llvm_shutdown();

  return status;
}

void codegenModules(Modules &modules) {
  // Generate one or more object/IR/bitcode files/dcompute kernels.
  if (global.params.obj && !modules.empty()) {
    TimeTraceScope timeScope("Codegen all modules");

#if LDC_MLIR_ENABLED
    mlir::MLIRContext mlircontext;
    ldc::CodeGenerator cg(getGlobalContext(), mlircontext,
                          global.params.oneobj);
#else
    ldc::CodeGenerator cg(getGlobalContext(), global.params.oneobj);
#endif
    DComputeCodeGenManager dccg(getGlobalContext());
    std::vector<Module *> computeModules;
    // When inlining is enabled, we are calling semantic3 on function
    // declarations, which may _add_ members to the first module in the modules
    // array. These added functions must be codegenned, because these functions
    // may be "alwaysinline" and linker problems arise otherwise with templates
    // that have __FILE__ as parameters (which must be `pragma(inline, true);`)
    // Therefore, codegen is done in reverse order with members[0] last, to make
    // sure these functions (added to members[0] by members[x>0]) are
    // codegenned.
    for (d_size_t i = modules.length; i-- > 0;) {
      Module *const m = modules[i];

      if (m->filetype == FileType::dhdr)
        continue;

      if (global.params.v.verbose)
        message("code      %s", m->toChars());

      const auto atCompute = hasComputeAttr(m);
      if (atCompute == DComputeCompileFor::hostOnly ||
          atCompute == DComputeCompileFor::hostAndDevice) {
        TimeTraceScope timeScope(
            ("Codegen module " + llvm::SmallString<20>(m->toChars()))
                .str()
                .c_str(),
            m->loc);
#if LDC_MLIR_ENABLED
        if (global.params.output_mlir == OUTPUTFLAGset)
          cg.emitMLIR(m);
        else
#endif
          cg.emit(m);
      }
      if (atCompute != DComputeCompileFor::hostOnly) {
        computeModules.push_back(m);
        if (atCompute == DComputeCompileFor::deviceOnly) {
          // Remove m's object file from list of object files
          auto s = m->objfile.toChars();
          for (size_t j = 0; j < global.params.objfiles.length; j++) {
            if (s == global.params.objfiles[j]) {
              global.params.objfiles.remove(j);
              break;
            }
          }
        }
      }
      if (global.errors)
        fatal();
    }

    if (!computeModules.empty()) {
      TimeTraceScope timeScope("Codegen DCompute device modules");
      for (auto &mod : computeModules) {
        TimeTraceScope timeScope(("Codegen DCompute device module " +
                                  llvm::SmallString<20>(mod->toChars()))
                                     .str()
                                     .c_str(),
                                 mod->loc);
        dccg.emit(mod);
      }
    }
    dccg.writeModules();

    // We may have removed all object files, if so don't link.
    if (global.params.objfiles.length == 0)
      global.params.link = false;
  }

  {
    TimeTraceScope timeScope("Prune object file cache");
    cache::pruneCache();
  }

  freeRuntime();
}
