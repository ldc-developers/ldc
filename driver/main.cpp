//===-- main.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "module.h"
#include "errors.h"
#include "id.h"
#include "hdrgen.h"
#include "json.h"
#include "mars.h"
#include "mtype.h"
#include "identifier.h"
#include "rmem.h"
#include "root.h"
#include "scope.h"
#include "dmd/target.h"
#include "driver/cache.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/codegenerator.h"
#include "driver/configfile.h"
#include "driver/dcomputecodegenerator.h"
#include "driver/exe_path.h"
#include "driver/ldc-version.h"
#include "driver/linker.h"
#include "driver/plugins.h"
#include "driver/targetmachine.h"
#include "gen/cl_helpers.h"
#include "gen/irstate.h"
#include "gen/ldctraits.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/metadata.h"
#include "gen/modules.h"
#include "gen/objcgen.h"
#include "gen/optimizer.h"
#include "gen/passes/Passes.h"
#include "gen/runtime.h"
#include "gen/uda.h"
#include "gen/abi.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#if LDC_LLVM_VER >= 308
#include "llvm/Support/StringSaver.h"
#endif
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#if LDC_LLVM_VER >= 600
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#else
#include "llvm/Target/TargetSubtargetInfo.h"
#endif
#include "llvm/LinkAllIR.h"
#include "llvm/IR/LLVMContext.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#if _WIN32
#include <windows.h>
#endif

// Needs Type already declared.
#include "cond.h"

// From druntime/src/core/runtime.d.
extern "C" {
int rt_init();
}

// In dmd/doc.d
void gendocfile(Module *m);

// In dmd/mars.d
extern bool includeImports;
extern Strings includeModulePatterns;
void generateJson(Modules *modules);

using namespace opts;

extern void getenv_setargv(const char *envvar, int *pargc, char ***pargv);

static StringsAdapter impPathsStore("I", global.params.imppath);
static cl::list<std::string, StringsAdapter>
    importPaths("I", cl::desc("Look for imports also in <directory>"),
                cl::value_desc("directory"), cl::location(impPathsStore),
                cl::Prefix);

// This function exits the program.
void printVersion(llvm::raw_ostream &OS) {
  OS << "LDC - the LLVM D compiler (" << global.ldc_version << "):\n";
  OS << "  based on DMD " << global.version << " and LLVM " << global.llvm_version << "\n";
  OS << "  built with " << ldc::built_with_Dcompiler_version << "\n";
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
  OS << "  compiled with address sanitizer enabled\n";
#endif
#endif
  OS << "  Default target: " << llvm::sys::getDefaultTargetTriple() << "\n";
  std::string CPU = llvm::sys::getHostCPUName();
  if (CPU == "generic") {
    CPU = "(unknown)";
  }
  OS << "  Host CPU: " << CPU << "\n";
  OS << "  http://dlang.org - http://wiki.dlang.org/LDC\n";
  OS << "\n";

  // Without explicitly flushing here, only the target list is visible when
  // redirecting stdout to a file.
  OS.flush();

  llvm::TargetRegistry::printRegisteredTargetsForVersion(
#if LDC_LLVM_VER >= 600
    OS
#endif
    );

  exit(EXIT_SUCCESS);
}

// This function exits the program.
void printVersionStdout() {
  printVersion(llvm::outs());
  assert(false);
}


namespace {

// True when target triple has an uClibc environment
bool isUClibc = false;

// Helper function to handle -d-debug=* and -d-version=*
void processVersions(std::vector<std::string> &list, const char *type,
                     unsigned &globalLevel, Strings *&globalIDs) {
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
        if (!globalIDs)
          globalIDs = new Strings();
        globalIDs->push(cstr);
        continue;
      } else {
        error(Loc(), "Invalid %s identifier or level: '%s'", type, i.c_str());
      }
    }
  }
}

// Helper function to handle -transition=*
void processTransitions(std::vector<std::string> &list) {
  for (const auto &i : list) {
    if (i == "?") {
      printf("\n"
             "Language changes listed by -transition=id:\n"
             "  =all           list information on all language changes\n"
             "  =checkimports  give deprecation messages about 10378 "
             "anomalies\n"
             "  =complex,14488 list all usages of complex or imaginary types\n"
             "  =field,3449    list all non-mutable fields which occupy an "
             "object instance\n"
             "  =import,10378  revert to single phase name lookup\n"
             "  =intpromote,16997 fix integral promotions for unary + - ~ operators\n"
             "  =tls           list all variables going into thread local "
             "storage\n");
      exit(EXIT_SUCCESS);
    } else if (i == "all") {
      global.params.vtls = true;
      global.params.vfield = true;
      global.params.vcomplex = true;
      global.params.bug10378 = true;   // not set in DMD
      global.params.check10378 = true; // not set in DMD
    } else if (i == "checkimports") {
      global.params.check10378 = true;
    } else if (i == "complex" || i == "14488") {
      global.params.vcomplex = true;
    } else if (i == "field" || i == "3449") {
      global.params.vfield = true;
    } else if (i == "import" || i == "10378") {
      global.params.bug10378 = true;
    } else if (i == "intpromote" || i == "16997") {
      global.params.fix16997 = true;
    } else if (i == "tls") {
      global.params.vtls = true;
    } else {
      error(Loc(), "Invalid transition %s", i.c_str());
    }
  }
}

template <int N> // option length incl. terminating null
void tryParse(const llvm::SmallVectorImpl<const char *> &args, size_t i,
              const char *&output, const char (&option)[N]) {
  if (strncmp(args[i], option, N - 1) != 0)
    return;

  char nextChar = args[i][N - 1];
  if (nextChar == '=')
    output = args[i] + N;
  else if (nextChar == 0 && i < args.size() - 1)
    output = args[i + 1];
}

const char *
tryGetExplicitConfFile(const llvm::SmallVectorImpl<const char *> &args) {
  const char *conf = nullptr;
  // begin at the back => use latest -conf specification
  assert(args.size() >= 1);
  for (size_t i = args.size() - 1; !conf && i >= 1; --i) {
    tryParse(args, i, conf, "-conf");
  }
  return conf;
}

llvm::Triple
tryGetExplicitTriple(const llvm::SmallVectorImpl<const char *> &args) {
  // most combinations of flags are illegal, this mimicks command line
  //  behaviour for legal ones only
  llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
  const char *mtriple = nullptr;
  const char *march = nullptr;
  for (size_t i = 1; i < args.size(); ++i) {
    if (sizeof(void *) != 4 && strcmp(args[i], "-m32") == 0) {
      triple = triple.get32BitArchVariant();
      if (triple.getArch() == llvm::Triple::ArchType::x86)
        triple.setArchName("i686"); // instead of i386
      return triple;
    }

    if (sizeof(void *) != 8 && strcmp(args[i], "-m64") == 0)
      return triple.get64BitArchVariant();

    tryParse(args, i, mtriple, "-mtriple");
    tryParse(args, i, march, "-march");
  }
  if (mtriple)
    triple = llvm::Triple(llvm::Triple::normalize(mtriple));
  if (march) {
    std::string errorMsg; // ignore error, will show up later anyway
    lookupTarget(march, triple, errorMsg); // modifies triple
  }
  return triple;
}

void expandResponseFiles(llvm::BumpPtrAllocator &A,
                         llvm::SmallVectorImpl<const char *> &args) {
#if LDC_LLVM_VER >= 308
  llvm::StringSaver Saver(A);
  cl::ExpandResponseFiles(Saver,
#ifdef _WIN32
                          cl::TokenizeWindowsCommandLine
#else
                          cl::TokenizeGNUCommandLine
#endif
                          ,
                          args);
#endif
}

/// Parses switches from the command line, any response files and the global
/// config file and sets up global.params accordingly.
///
/// Returns a list of source file names.
void parseCommandLine(int argc, char **argv, Strings &sourceFiles,
                      bool &helpOnly) {
  global.params.argv0 = exe_path::getExePath().data();

  // Set up `opts::allArguments`, the combined list of command line arguments.
  using opts::allArguments;

  // initialize with the actual command line
  allArguments.insert(allArguments.end(), argv, argv + argc);

  // expand response files (`@<file>`) in-place
  llvm::BumpPtrAllocator allocator;
  expandResponseFiles(allocator, allArguments);

  // read config file
  ConfigFile &cfg_file = ConfigFile::instance;
  const char *explicitConfFile = tryGetExplicitConfFile(allArguments);
  const std::string cfg_triple = tryGetExplicitTriple(allArguments).getTriple();
  // just ignore errors for now, they are still printed
  cfg_file.read(explicitConfFile, cfg_triple.c_str());

  cfg_file.extendCommandLine(allArguments);

  // finalize by expanding response files specified in config file
  expandResponseFiles(allocator, allArguments);

#if LDC_LLVM_VER >= 600
  cl::SetVersionPrinter(&printVersion);
#else
  cl::SetVersionPrinter(&printVersionStdout);
#endif

  opts::hideLLVMOptions();
  opts::createClashingOptions();

  cl::ParseCommandLineOptions(allArguments.size(),
                              const_cast<char **>(allArguments.data()),
                              "LDC - the LLVM D compiler\n");

  helpOnly = opts::printTargetFeaturesHelp();
  if (helpOnly) {
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
    return;
  }

  if (!cfg_file.path().empty())
    global.inifilename = dupPathString(cfg_file.path());

  // Print some information if -v was passed
  // - path to compiler binary
  // - version number
  // - used config file
  if (global.params.verbose) {
    message("binary    %s", exe_path::getExePath().c_str());
    message("version   %s (DMD %s, LLVM %s)", global.ldc_version,
            global.version, global.llvm_version);
    if (global.inifilename) {
      message("config    %s (%s)", global.inifilename, cfg_triple.c_str());
    }
  }

  // Negated options
  global.params.link = !compileOnly;
  global.params.obj = !dontWriteObj;
  global.params.release = !opts::invReleaseMode;
  global.params.useInlineAsm = !noAsm;

  // String options: std::string --> char*
  opts::initFromPathString(global.params.objname, objectFile);
  opts::initFromPathString(global.params.objdir, objectDir);

  opts::initFromPathString(global.params.docdir, ddocDir);
  opts::initFromPathString(global.params.docname, ddocFile);
  global.params.doDocComments |= global.params.docdir || global.params.docname;

  opts::initFromPathString(global.params.jsonfilename, jsonFile);
  if (global.params.jsonfilename) {
    global.params.doJsonGeneration = true;
  }

  opts::initFromPathString(global.params.hdrdir, hdrDir);
  opts::initFromPathString(global.params.hdrname, hdrFile);
  global.params.doHdrGeneration |=
      global.params.hdrdir || global.params.hdrname;

  if (moduleDeps.getNumOccurrences() != 0) {
    global.params.moduleDeps = new OutBuffer;
    if (!moduleDeps.empty())
      global.params.moduleDepsFile = opts::dupPathString(moduleDeps);
  }

#if _WIN32
  const auto toWinPaths = [](Strings *paths) {
    if (!paths)
      return;
    for (auto &path : *paths)
      path = opts::dupPathString(path);
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

#if LDC_LLVM_VER >= 400
  if (saveOptimizationRecord.getNumOccurrences() > 0) {
    global.params.outputSourceLocations = true;
  }
#endif

  opts::initializeSanitizerOptionsFromCmdline();

  processVersions(debugArgs, "debug", global.params.debuglevel,
                  global.params.debugids);
  processVersions(versions, "version", global.params.versionlevel,
                  global.params.versionids);

  processTransitions(transitions);

  if (useDIP1000) {
    global.params.useDIP25 = true;
    global.params.vsafe = true;
  }

  global.params.output_o =
      (opts::output_o == cl::BOU_UNSET &&
       !(opts::output_bc || opts::output_ll || opts::output_s))
          ? OUTPUTFLAGdefault
          : opts::output_o == cl::BOU_TRUE ? OUTPUTFLAGset : OUTPUTFLAGno;
  global.params.output_bc = opts::output_bc ? OUTPUTFLAGset : OUTPUTFLAGno;
  global.params.output_ll = opts::output_ll ? OUTPUTFLAGset : OUTPUTFLAGno;
  global.params.output_s = opts::output_s ? OUTPUTFLAGset : OUTPUTFLAGno;

  global.params.cov = (global.params.covPercent <= 100);

  templateLinkage = opts::linkonceTemplates ? LLGlobalValue::LinkOnceODRLinkage
                                            : LLGlobalValue::WeakODRLinkage;

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
        if (ext && !FileName::equals(ext, "d") &&
            !FileName::equals(ext, "di")) {
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
  }

  sourceFiles.reserve(fileList.size());
  for (const auto &file : fileList) {
    if (!file.empty()) {
      if (file == "-") {
        sourceFiles.push("__stdin.d");
      } else {
        char *copy = opts::dupPathString(file);
        sourceFiles.push(copy);
      }
    }
  }

  if (global.params.betterC) {
    global.params.checkAction = CHECKACTION_C;
    global.params.useModuleInfo = false;
    global.params.useTypeInfo = false;
    global.params.useExceptions = false;
  }

  if (global.params.useUnitTests) {
    global.params.useAssert = CHECKENABLEon;
  }

  // -release downgrades default checks
  if (global.params.useArrayBounds == CHECKENABLEdefault)
    global.params.useArrayBounds = global.params.release ? CHECKENABLEsafeonly : CHECKENABLEon;
  if (global.params.useAssert == CHECKENABLEdefault)
    global.params.useAssert = global.params.release ? CHECKENABLEoff : CHECKENABLEon;
  if (global.params.useSwitchError == CHECKENABLEdefault)
    global.params.useSwitchError = global.params.release ? CHECKENABLEoff : CHECKENABLEon;

  // LDC output determination

  // if we don't link and there's no `-output-*` switch but an `-of` one,
  // autodetect type of desired 'object' file from file extension
  if (!global.params.link && !global.params.lib && global.params.objname &&
      global.params.output_o == OUTPUTFLAGdefault) {
    const char *ext = FileName::ext(global.params.objname);
    if (!ext) {
      // keep things as they are
    } else if (opts::output_ll.getNumOccurrences() == 0 &&
               strcmp(ext, global.ll_ext) == 0) {
      global.params.output_ll = OUTPUTFLAGset;
      global.params.output_o = OUTPUTFLAGno;
    } else if (opts::output_bc.getNumOccurrences() == 0 &&
               strcmp(ext, global.bc_ext) == 0) {
      global.params.output_bc = OUTPUTFLAGset;
      global.params.output_o = OUTPUTFLAGno;
    } else if (opts::output_s.getNumOccurrences() == 0 &&
               strcmp(ext, global.s_ext) == 0) {
      global.params.output_s = OUTPUTFLAGset;
      global.params.output_o = OUTPUTFLAGno;
    }
  }

  // only link if possible
  if (!global.params.obj || !global.params.output_o || global.params.lib) {
    global.params.link = false;
  }

  if (global.params.lib && global.params.dll) {
    error(Loc(), "-lib and -shared switches cannot be used together");
  }

  if (soname.getNumOccurrences() > 0 && !global.params.dll) {
    error(Loc(), "-soname can be used only when building a shared library");
  }

  global.params.hdrStripPlainFunctions = !opts::hdrKeepAllBodies;
  global.params.disableRedZone = opts::disableRedZone();
}

void initializePasses() {
  using namespace llvm;
  // Initialize passes
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeTransformUtils(Registry);
  initializeScalarOpts(Registry);
  initializeObjCARCOpts(Registry);
  initializeVectorization(Registry);
  initializeInstCombine(Registry);
  initializeIPO(Registry);
  initializeInstrumentation(Registry);
  initializeAnalysis(Registry);
  initializeCodeGen(Registry);
#if LDC_LLVM_VER >= 309
  initializeGlobalISel(Registry);
#endif
  initializeTarget(Registry);

// Initialize passes not included above
#if LDC_LLVM_VER < 308
  initializeIPA(Registry);
#endif
#if LDC_LLVM_VER >= 400
  initializeRewriteSymbolsLegacyPassPass(Registry);
#else
  initializeRewriteSymbolsPass(Registry);
#endif
  initializeSjLjEHPreparePass(Registry);
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

// Check if triple environment name starts with "uclibc" and change it to "gnu"
void fixupUClibcEnv()
{
  llvm::Triple triple(mTargetTriple);
  if (triple.getEnvironmentName().find("uclibc") != 0)
    return;
  std::string envName = triple.getEnvironmentName();
  envName.replace(0, 6, "gnu");
  triple.setEnvironmentName(envName);
  mTargetTriple = triple.normalize();
  isUClibc = true;
}

/// Register the float ABI.
/// Also defines D_HardFloat or D_SoftFloat depending if FPU should be used
void registerPredefinedFloatABI(const char *soft, const char *hard,
                                const char *softfp = nullptr) {
  // Use target floating point unit instead of s/w float routines
  // FIXME: This is a semantic change!
  bool useFPU = gTargetMachine->Options.FloatABIType == llvm::FloatABI::Hard;
  VersionCondition::addPredefinedGlobalIdent(useFPU ? "D_HardFloat"
                                                    : "D_SoftFloat");

  if (gTargetMachine->Options.FloatABIType == llvm::FloatABI::Soft) {
    VersionCondition::addPredefinedGlobalIdent(useFPU && softfp ? softfp
                                                                : soft);
  } else if (gTargetMachine->Options.FloatABIType == llvm::FloatABI::Hard) {
    assert(useFPU && "Should be using the FPU if using float-abi=hard");
    VersionCondition::addPredefinedGlobalIdent(hard);
  } else {
    assert(0 && "FloatABIType neither Soft or Hard");
  }
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
#if defined RISCV_LLVM_DEV || LDC_LLVM_VER >= 400
#if defined RISCV_LLVM_DEV
  case llvm::Triple::riscv:
#else
  case llvm::Triple::riscv32:
#endif
    VersionCondition::addPredefinedGlobalIdent("RISCV32");
    break;
  case llvm::Triple::riscv64:
    VersionCondition::addPredefinedGlobalIdent("RISCV64");
    break;
#endif
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
  default:
    error(Loc(), "invalid cpu architecture specified: %s",
          triple.getArchName().str().c_str());
    fatal();
  }

  // endianness
  if (gDataLayout->isLittleEndian()) {
    VersionCondition::addPredefinedGlobalIdent("LittleEndian");
  } else {
    VersionCondition::addPredefinedGlobalIdent("BigEndian");
  }

  // Set versions for arch bitwidth
  if (global.params.isLP64) {
    VersionCondition::addPredefinedGlobalIdent("D_LP64");
  } else if (triple.isArch16Bit()) {
    VersionCondition::addPredefinedGlobalIdent("D_P16");
  }

  if (gTargetMachine->getRelocationModel() == llvm::Reloc::PIC_) {
    VersionCondition::addPredefinedGlobalIdent("D_PIC");
  }

  /* LDC doesn't support DMD's core.simd interface.
  if (arch == llvm::Triple::x86 || arch == llvm::Triple::x86_64) {
    if (traitsTargetHasFeature("sse2"))
      VersionCondition::addPredefinedGlobalIdent("D_SIMD");
    if (traitsTargetHasFeature("avx"))
      VersionCondition::addPredefinedGlobalIdent("D_AVX");
    if (traitsTargetHasFeature("avx2"))
      VersionCondition::addPredefinedGlobalIdent("D_AVX2");
  }
  */

  // parse the OS out of the target triple
  // see http://gcc.gnu.org/install/specific.html for details
  // also llvm's different SubTargets have useful information
  switch (triple.getOS()) {
  case llvm::Triple::Win32:
    VersionCondition::addPredefinedGlobalIdent("Windows");
    VersionCondition::addPredefinedGlobalIdent(global.params.is64bit ? "Win64"
                                                                     : "Win32");
    if (triple.isWindowsMSVCEnvironment()) {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Microsoft");
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
    } else if (isMusl()) {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Musl");
    } else if (isUClibc) {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_UClibc");
    } else {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Glibc");
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
    break;
  case llvm::Triple::FreeBSD:
    VersionCondition::addPredefinedGlobalIdent("FreeBSD");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  case llvm::Triple::Solaris:
    VersionCondition::addPredefinedGlobalIdent("Solaris");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  case llvm::Triple::DragonFly:
    VersionCondition::addPredefinedGlobalIdent("DragonFlyBSD");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  case llvm::Triple::NetBSD:
    VersionCondition::addPredefinedGlobalIdent("NetBSD");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  case llvm::Triple::OpenBSD:
    VersionCondition::addPredefinedGlobalIdent("OpenBSD");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  case llvm::Triple::AIX:
    VersionCondition::addPredefinedGlobalIdent("AIX");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    break;
  default:
    if (triple.getEnvironment() == llvm::Triple::Android) {
      VersionCondition::addPredefinedGlobalIdent("Android");
    } else {
      warning(Loc(), "unknown OS for target '%s'", triple.str().c_str());
    }
    break;
  }
}

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

  if (global.params.doDocComments) {
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

  if (global.params.useArrayBounds == CHECKENABLEoff) {
    VersionCondition::addPredefinedGlobalIdent("D_NoBoundsChecks");
  }

  if (global.params.betterC) {
    VersionCondition::addPredefinedGlobalIdent("D_BetterC");
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

} // anonymous namespace

int cppmain(int argc, char **argv) {
#if LDC_LLVM_VER >= 309
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
#else
  llvm::sys::PrintStackTraceOnErrorSignal();
#endif

  exe_path::initialize(argv[0]);

  global._init();
  global.version = ldc::dmd_version;
  global.ldc_version = ldc::ldc_version;
  global.llvm_version = ldc::llvm_version;

  // Initialize LLVM before parsing the command line so that --version shows
  // registered targets.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  initializePasses();

  bool helpOnly;
  Strings files;
  parseCommandLine(argc, argv, files, helpOnly);

  if (helpOnly) {
    return 0;
  }

  if (files.dim == 0) {
    if (global.params.jsonFieldFlags) {
      generateJson(nullptr);
      return EXIT_SUCCESS;
    }

    cl::PrintHelpMessage(/*Hidden=*/false, /*Categorized=*/true);
    return EXIT_FAILURE;
  }

  if (global.errors) {
    fatal();
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
#if LDC_LLVM_VER >= 309
  if (global.params.dll && !relocModel.hasValue()) {
#else
  if (global.params.dll && relocModel == llvm::Reloc::Default) {
#endif
    relocModel = llvm::Reloc::PIC_;
  }

  // check and fix environment for uClibc
  fixupUClibcEnv();

  gTargetMachine = createTargetMachine(
      mTargetTriple, arch, opts::getCPUStr(), opts::getFeaturesStr(), bitness,
      floatABI, relocModel, opts::getCodeModel(), codeGenOptLevel(),
      disableLinkerStripDead);

  opts::setDefaultMathOptions(gTargetMachine->Options);

#if LDC_LLVM_VER >= 308
  static llvm::DataLayout DL = gTargetMachine->createDataLayout();
  gDataLayout = &DL;
#else
  gDataLayout = gTargetMachine->getDataLayout();
#endif

  {
    llvm::Triple *triple = new llvm::Triple(gTargetMachine->getTargetTriple());
    global.params.targetTriple = triple;
    global.params.isLinux = triple->isOSLinux();
    global.params.isOSX = triple->isOSDarwin();
    global.params.isWindows = triple->isOSWindows();
    global.params.isFreeBSD = triple->isOSFreeBSD();
    global.params.isOpenBSD = triple->isOSOpenBSD();
    global.params.isSolaris = triple->isOSSolaris();
    global.params.isLP64 = gDataLayout->getPointerSizeInBits() == 64;
    global.params.is64bit = triple->isArch64Bit();
    global.params.hasObjectiveC = objc_isSupported(*triple);
    global.params.dwarfVersion = gTargetMachine->Options.MCOptions.DwarfVersion;
    // mscoff enables slightly different handling of interface functions
    // in the front end
    global.params.mscoff = triple->isKnownWindowsMSVCEnvironment();
    if (global.params.mscoff)
      global.obj_ext = "obj";
  }

  // allocate the target abi
  gABI = TargetABI::getTarget();

  if (global.params.targetTriple->isOSWindows()) {
    global.dll_ext = "dll";
    global.lib_ext = (global.params.mscoff ? "lib" : "a");
  } else {
    global.dll_ext = global.params.targetTriple->isOSDarwin() ? "dylib" : "so";
    global.lib_ext = "a";
  }

  opts::initializeInstrumentationOptionsFromCmdline(
      *global.params.targetTriple);

  loadAllPlugins();

  Strings libmodules;
  return mars_mainBody(files, libmodules);
}

void addDefaultVersionIdentifiers() {
  registerPredefinedVersions();
}

void codegenModules(Modules &modules) {
  // Generate one or more object/IR/bitcode files/dcompute kernels.
  if (global.params.obj && !modules.empty()) {
    ldc::CodeGenerator cg(getGlobalContext(), global.params.oneobj);
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
    for (d_size_t i = modules.dim; i-- > 0;) {
      Module *const m = modules[i];
      if (global.params.verbose)
        message("code      %s", m->toChars());

      const auto atCompute = hasComputeAttr(m);
      if (atCompute == DComputeCompileFor::hostOnly ||
           atCompute == DComputeCompileFor::hostAndDevice)
      {
        cg.emit(m);
      }
      if (atCompute != DComputeCompileFor::hostOnly) {
        computeModules.push_back(m);
        if (atCompute == DComputeCompileFor::deviceOnly) {
          // Remove m's object file from list of object files
          auto s = m->objfile->name->str;
          for (size_t j = 0; j < global.params.objfiles.dim; j++) {
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
      for (auto& mod : computeModules)
        dccg.emit(mod);

      dccg.writeModules();
    }
    // We may have removed all object files, if so don't link.
    if (global.params.objfiles.dim == 0)
      global.params.link = false;

  }

  cache::pruneCache();

  freeRuntime();
  llvm::llvm_shutdown();
}
