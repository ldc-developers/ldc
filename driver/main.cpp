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
#include "ddmd/target.h"
#include "driver/cache.h"
#include "driver/cl_options.h"
#include "driver/codegenerator.h"
#include "driver/configfile.h"
#include "driver/exe_path.h"
#include "driver/ldc-version.h"
#include "driver/linker.h"
#include "driver/targetmachine.h"
#include "gen/cl_helpers.h"
#include "gen/irstate.h"
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
#if LDC_LLVM_VER >= 306
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

// In ddmd/doc.d
void gendocfile(Module *m);

using namespace opts;

extern void getenv_setargv(const char *envvar, int *pargc, char ***pargv);

static cl::opt<bool>
    noDefaultLib("nodefaultlib",
                 cl::desc("Don't add a default library for linking implicitly"),
                 cl::ZeroOrMore, cl::Hidden);

static StringsAdapter impPathsStore("I", global.params.imppath);
static cl::list<std::string, StringsAdapter>
    importPaths("I", cl::desc("Where to look for imports"),
                cl::value_desc("path"), cl::location(impPathsStore),
                cl::Prefix);

static cl::opt<std::string>
    defaultLib("defaultlib",
               cl::desc("Default libraries to link with (overrides previous)"),
               cl::value_desc("lib1,lib2,..."), cl::ZeroOrMore);

static cl::opt<std::string> debugLib(
    "debuglib",
    cl::desc("Debug versions of default libraries (overrides previous)"),
    cl::value_desc("lib1,lib2,..."), cl::ZeroOrMore);

static cl::opt<bool> linkDebugLib(
    "link-debuglib",
    cl::desc("Link with libraries specified in -debuglib, not -defaultlib"),
    cl::ZeroOrMore);

#if LDC_LLVM_VER >= 309
static inline llvm::Optional<llvm::Reloc::Model> getRelocModel() {
  if (mRelocModel.getNumOccurrences()) {
    llvm::Reloc::Model R = mRelocModel;
    return R;
  }
  return llvm::None;
}
#else
static inline llvm::Reloc::Model getRelocModel() { return mRelocModel; }
#endif

void printVersion() {
  printf("LDC - the LLVM D compiler (%s):\n", global.ldc_version);
  printf("  based on DMD %s and LLVM %s\n", global.version,
         global.llvm_version);
  printf("  built with %s\n", ldc::built_with_Dcompiler_version);
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
  printf("  compiled with address sanitizer enabled\n");
#endif
#endif
  printf("  Default target: %s\n", llvm::sys::getDefaultTargetTriple().c_str());
  std::string CPU = llvm::sys::getHostCPUName();
  if (CPU == "generic") {
    CPU = "(unknown)";
  }
  printf("  Host CPU: %s\n", CPU.c_str());
  printf("  http://dlang.org - http://wiki.dlang.org/LDC\n");
  printf("\n");

  // Without explicitly flushing here, only the target list is visible when
  // redirecting stdout to a file.
  fflush(stdout);

  llvm::TargetRegistry::printRegisteredTargetsForVersion();
  exit(EXIT_SUCCESS);
}

namespace {

// Helper function to handle -d-debug=* and -d-version=*
void processVersions(std::vector<std::string> &list, const char *type,
                     void (*setLevel)(unsigned),
                     void (*addIdent)(const char *)) {
  for (const auto &i : list) {
    const char *value = i.c_str();
    if (isdigit(value[0])) {
      errno = 0;
      char *end;
      long level = strtol(value, &end, 10);
      if (*end || errno || level > INT_MAX) {
        error(Loc(), "Invalid %s level: %s", type, i.c_str());
      } else {
        setLevel(static_cast<unsigned>(level));
      }
    } else {
      char *cstr = mem.xstrdup(value);
      if (Identifier::isValidIdentifier(cstr)) {
        addIdent(cstr);
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
      printf("Language changes listed by -transition=id:\n");
      printf("  = all           list information on all language changes\n");
      printf("  = checkimports  give deprecation messages about 10378 "
             "anomalies\n");
      printf(
          "  = complex,14488 list all usages of complex or imaginary types\n");
      printf("  = field,3449    list all non - mutable fields which occupy an "
             "object instance\n");
      printf("  = import,10378  revert to single phase name lookup\n");
      printf("  = tls           list all variables going into thread local "
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
    } else if (i == "tls") {
      global.params.vtls = true;
    } else {
      error(Loc(), "Invalid transition %s", i.c_str());
    }
  }
}

char *dupPathString(const std::string &src) {
  char *r = mem.xstrdup(src.c_str());
#if _WIN32
  std::replace(r, r + src.length(), '/', '\\');
#endif
  return r;
}

// Helper function to handle -of, -od, etc.
void initFromPathString(const char *&dest, const cl::opt<std::string> &src) {
  dest = nullptr;
  if (src.getNumOccurrences() != 0) {
    if (src.empty()) {
      error(Loc(), "Expected argument to '-%s'", src.ArgStr);
    }
    dest = dupPathString(src);
  }
}

void hide(llvm::StringMap<cl::Option *> &map, const char *name) {
  // Check if option exists first for resilience against LLVM changes
  // between versions.
  if (map.count(name)) {
    map[name]->setHiddenFlag(cl::Hidden);
  }
}

#if LDC_LLVM_VER >= 307
void rename(llvm::StringMap<cl::Option *> &map, const char *from,
            const char *to) {
  auto i = map.find(from);
  if (i != map.end()) {
    cl::Option *opt = i->getValue();
    map.erase(i);
    opt->setArgStr(to);
    map[to] = opt;
  }
}
#endif

/// Removes command line options exposed from within LLVM that are unlikely
/// to be useful for end users from the -help output.
void hideLLVMOptions() {
#if LDC_LLVM_VER >= 307
  llvm::StringMap<cl::Option *> &map = cl::getRegisteredOptions();
#else
  llvm::StringMap<cl::Option *> map;
  cl::getRegisteredOptions(map);
#endif
  hide(map, "bounds-checking-single-trap");
  hide(map, "disable-debug-info-verifier");
  hide(map, "disable-objc-arc-checkforcfghazards");
  hide(map, "disable-spill-fusing");
  hide(map, "cppfname");
  hide(map, "cppfor");
  hide(map, "cppgen");
  hide(map, "enable-correct-eh-support");
  hide(map, "enable-load-pre");
  hide(map, "enable-misched");
  hide(map, "enable-objc-arc-annotations");
  hide(map, "enable-objc-arc-opts");
  hide(map, "enable-scoped-noalias");
  hide(map, "enable-tbaa");
  hide(map, "exhaustive-register-search");
  hide(map, "fatal-assembler-warnings");
  hide(map, "internalize-public-api-file");
  hide(map, "internalize-public-api-list");
  hide(map, "join-liveintervals");
  hide(map, "limit-float-precision");
  hide(map, "mc-x86-disable-arith-relaxation");
  hide(map, "mips16-constant-islands");
  hide(map, "mips16-hard-float");
  hide(map, "mlsm");
  hide(map, "mno-ldc1-sdc1");
  hide(map, "nvptx-sched4reg");
  hide(map, "no-discriminators");
  hide(map, "objc-arc-annotation-target-identifier"), hide(map, "pre-RA-sched");
  hide(map, "print-after-all");
  hide(map, "print-before-all");
  hide(map, "print-machineinstrs");
  hide(map, "profile-estimator-loop-weight");
  hide(map, "profile-estimator-loop-weight");
  hide(map, "profile-file");
  hide(map, "profile-info-file");
  hide(map, "profile-verifier-noassert");
  hide(map, "regalloc");
  hide(map, "rewrite-map-file");
  hide(map, "rng-seed");
  hide(map, "sample-profile-max-propagate-iterations");
  hide(map, "shrink-wrap");
  hide(map, "spiller");
  hide(map, "stackmap-version");
  hide(map, "stats");
  hide(map, "strip-debug");
  hide(map, "struct-path-tbaa");
  hide(map, "time-passes");
  hide(map, "unit-at-a-time");
  hide(map, "verify-debug-info");
  hide(map, "verify-dom-info");
  hide(map, "verify-loop-info");
  hide(map, "verify-regalloc");
  hide(map, "verify-region-info");
  hide(map, "verify-scev");
  hide(map, "x86-early-ifcvt");
  hide(map, "x86-use-vzeroupper");
  hide(map, "x86-recip-refinement-steps");

  // We enable -fdata-sections/-ffunction-sections by default where it makes
  // sense for reducing code size, so hide them to avoid confusion.
  //
  // We need our own switch as these two are defined by LLVM and linked to
  // static TargetMachine members, but the default we want to use depends
  // on the target triple (and thus we do not know it until after the command
  // line has been parsed).
  hide(map, "fdata-sections");
  hide(map, "ffunction-sections");

#if LDC_LLVM_VER >= 307
  // LLVM 3.7 introduces compiling as shared library. The result
  // is a clash in the command line options.
  rename(map, "color", "llvm-color");
  hide(map, "llvm-color");
  opts::CreateColorOption();
#endif
}

const char *tryGetExplicitConfFile(int argc, char **argv) {
  // begin at the back => use latest -conf= specification
  for (int i = argc - 1; i >= 1; --i) {
    if (strncmp(argv[i], "-conf=", 6) == 0) {
      return argv[i] + 6;
    }
  }
  return nullptr;
}

llvm::Triple tryGetExplicitTriple(int argc, char **argv) {
  // most combinations of flags are illegal, this mimicks command line
  //  behaviour for legal ones only
  llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
  const char *mtriple = nullptr;
  const char *march = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (sizeof(void *) != 4 && strcmp(argv[i], "-m32") == 0) {
      triple = triple.get32BitArchVariant();
      if (triple.getArch() == llvm::Triple::ArchType::x86)
        triple.setArchName("i686"); // instead of i386
      return triple;
    }

    if (sizeof(void *) != 8 && strcmp(argv[i], "-m64") == 0)
      return triple.get64BitArchVariant();

    if (strncmp(argv[i], "-mtriple=", 9) == 0)
      mtriple = argv[i] + 9;
    else if (strncmp(argv[i], "-march=", 7) == 0)
      march = argv[i] + 7;
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
void parseCommandLine(int argc, char **argv, Strings &sourceFiles,
                      bool &helpOnly) {
  global.params.argv0 = exe_path::getExePath().data();

  // Set some default values.
  global.params.useSwitchError = 1;
  global.params.color = isConsoleColorSupported();

  global.params.linkswitches = new Strings();
  global.params.libfiles = new Strings();
  global.params.objfiles = new Strings();
  global.params.ddocfiles = new Strings();
  global.params.bitcodeFiles = new Strings();

  global.params.moduleDeps = nullptr;
  global.params.moduleDepsFile = nullptr;

  // Build combined list of command line arguments.
  opts::allArguments.push_back(argv[0]);

  ConfigFile cfg_file;
  const char *explicitConfFile = tryGetExplicitConfFile(argc, argv);
  std::string cfg_triple = tryGetExplicitTriple(argc, argv).getTriple();
  // just ignore errors for now, they are still printed
  cfg_file.read(explicitConfFile, cfg_triple.c_str());
  opts::allArguments.insert(opts::allArguments.end(), cfg_file.switches_begin(),
                            cfg_file.switches_end());

  opts::allArguments.insert(opts::allArguments.end(), &argv[1], &argv[argc]);

  cl::SetVersionPrinter(&printVersion);
  hideLLVMOptions();

// pre-expand response files (LLVM's ParseCommandLineOptions() always uses
// TokenizeGNUCommandLine which eats backslashes)
#if LDC_LLVM_VER >= 308
  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);
  cl::ExpandResponseFiles(Saver,
#ifdef _WIN32
                          cl::TokenizeWindowsCommandLine
#else
                          cl::TokenizeGNUCommandLine
#endif
                          ,
                          opts::allArguments);
#endif

  cl::ParseCommandLineOptions(opts::allArguments.size(),
                              const_cast<char **>(opts::allArguments.data()),
                              "LDC - the LLVM D compiler\n");

  helpOnly = mCPU == "help" ||
             (std::find(mAttrs.begin(), mAttrs.end(), "help") != mAttrs.end());

  // Print some information if -v was passed
  // - path to compiler binary
  // - version number
  // - used config file
  if (global.params.verbose) {
    fprintf(global.stdmsg, "binary    %s\n", exe_path::getExePath().c_str());
    fprintf(global.stdmsg, "version   %s (DMD %s, LLVM %s)\n",
            global.ldc_version, global.version, global.llvm_version);
    const std::string &path = cfg_file.path();
    if (!path.empty()) {
      fprintf(global.stdmsg, "config    %s\n", path.c_str());
    }
  }

  // Negated options
  global.params.link = !compileOnly;
  global.params.obj = !dontWriteObj;
  global.params.useInlineAsm = !noAsm;

  // String options: std::string --> char*
  initFromPathString(global.params.objname, objectFile);
  initFromPathString(global.params.objdir, objectDir);

  initFromPathString(global.params.docdir, ddocDir);
  initFromPathString(global.params.docname, ddocFile);
  global.params.doDocComments |= global.params.docdir || global.params.docname;

  initFromPathString(global.params.jsonfilename, jsonFile);
  if (global.params.jsonfilename) {
    global.params.doJsonGeneration = true;
  }

  initFromPathString(global.params.hdrdir, hdrDir);
  initFromPathString(global.params.hdrname, hdrFile);
  global.params.doHdrGeneration |=
      global.params.hdrdir || global.params.hdrname;

  if (moduleDeps.getNumOccurrences() != 0) {
    global.params.moduleDeps = new OutBuffer;
    if (!moduleDeps.empty())
      global.params.moduleDepsFile = dupPathString(moduleDeps);
  }

#if _WIN32
  const auto toWinPaths = [](Strings *paths) {
    if (!paths)
      return;
    for (unsigned i = 0; i < paths->dim; ++i)
      (*paths)[i] = dupPathString((*paths)[i]);
  };
  toWinPaths(global.params.imppath);
  toWinPaths(global.params.fileImppath);
#endif

// PGO options
#if LDC_WITH_PGO
  if (genfileInstrProf.getNumOccurrences() > 0) {
    global.params.genInstrProf = true;
    if (genfileInstrProf.empty()) {
#if LDC_LLVM_VER >= 309
      // profile-rt provides a default filename by itself
      global.params.datafileInstrProf = nullptr;
#else
      global.params.datafileInstrProf = "default.profraw";
#endif
    } else {
      initFromPathString(global.params.datafileInstrProf, genfileInstrProf);
    }
  } else {
    global.params.genInstrProf = false;
    // If we don't have to generate instrumentation, we could be given a
    // profdata file:
    initFromPathString(global.params.datafileInstrProf, usefileInstrProf);
  }
#endif

  processVersions(debugArgs, "debug", DebugCondition::setGlobalLevel,
                  DebugCondition::addGlobalIdent);
  processVersions(versions, "version", VersionCondition::setGlobalLevel,
                  VersionCondition::addGlobalIdent);

  processTransitions(transitions);

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
      char const *name = runargs[0].c_str();
      char const *ext = FileName::ext(name);
      if (ext && FileName::equals(ext, "d") == 0 &&
          FileName::equals(ext, "di") == 0) {
        error(Loc(), "-run must be followed by a source file, not '%s'", name);
      }

      sourceFiles.push(mem.xstrdup(name));
      runargs.erase(runargs.begin());
    } else {
      global.params.run = false;
      error(Loc(), "Expected at least one argument to '-run'\n");
    }
  }

  sourceFiles.reserve(fileList.size());
  for (const auto &file : fileList) {
    if (!file.empty()) {
      char *copy = dupPathString(file);
      sourceFiles.push(copy);
    }
  }

  if (noDefaultLib) {
    deprecation(
        Loc(),
        "-nodefaultlib is deprecated, as "
        "-defaultlib/-debuglib now override the existing list instead of "
        "appending to it. Please use the latter instead.");
  } else {
    // Parse comma-separated default library list.
    std::stringstream libNames(linkDebugLib ? debugLib : defaultLib);
    while (libNames.good()) {
      std::string lib;
      std::getline(libNames, lib, ',');
      if (lib.empty()) {
        continue;
      }

      char *arg = static_cast<char *>(mem.xmalloc(lib.size() + 3));
      strcpy(arg, "-l");
      strcpy(arg + 2, lib.c_str());
      global.params.linkswitches->push(arg);
    }
  }

  if (global.params.useUnitTests) {
    global.params.useAssert = 1;
  }

  // -release downgrades default bounds checking level to BOUNDSCHECKsafeonly
  // (only for safe functions).
  global.params.useArrayBounds =
      opts::nonSafeBoundsChecks ? BOUNDSCHECKon : BOUNDSCHECKsafeonly;
  if (opts::boundsCheck != BOUNDSCHECKdefault) {
    global.params.useArrayBounds = opts::boundsCheck;
  }

  // LDC output determination

  // if we don't link, autodetect target from extension
  if (!global.params.link && !global.params.lib && global.params.objname) {
    const char *ext = FileName::ext(global.params.objname);
    bool autofound = false;
    if (!ext) {
      // keep things as they are
    } else if (strcmp(ext, global.ll_ext) == 0) {
      global.params.output_ll = OUTPUTFLAGset;
      autofound = true;
    } else if (strcmp(ext, global.bc_ext) == 0) {
      global.params.output_bc = OUTPUTFLAGset;
      autofound = true;
    } else if (strcmp(ext, global.s_ext) == 0) {
      global.params.output_s = OUTPUTFLAGset;
      autofound = true;
    } else if (strcmp(ext, global.obj_ext) == 0 || strcmp(ext, "obj") == 0) {
      // global.obj_ext hasn't been corrected yet for MSVC targets as we first
      // need the command line to figure out the target...
      // so treat both 'o' and 'obj' extensions as object files
      global.params.output_o = OUTPUTFLAGset;
      autofound = true;
    }
    if (autofound && global.params.output_o == OUTPUTFLAGdefault) {
      global.params.output_o = OUTPUTFLAGno;
    }
  }

  // only link if possible
  if (!global.params.obj || !global.params.output_o || global.params.lib) {
    global.params.link = 0;
  }

  if (global.params.lib && global.params.dll) {
    error(Loc(), "-lib and -shared switches cannot be used together");
  }

#if LDC_LLVM_VER >= 309
  if (global.params.dll && !mRelocModel.getNumOccurrences()) {
#else
  if (global.params.dll && mRelocModel == llvm::Reloc::Default) {
#endif
    mRelocModel = llvm::Reloc::PIC_;
  }

  if (soname.getNumOccurrences() > 0 && !global.params.dll) {
    error(Loc(), "-soname can be used only when building a shared library");
  }
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
#if LDC_LLVM_VER < 306
  initializeDebugIRPass(Registry);
#endif
#if LDC_LLVM_VER < 308
  initializeIPA(Registry);
#endif
#if LDC_LLVM_VER >= 400
  initializeRewriteSymbolsLegacyPassPass(Registry);
#elif LDC_LLVM_VER >= 306
  initializeRewriteSymbolsPass(Registry);
#endif
#if LDC_LLVM_VER >= 307
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

/// Register the float ABI.
/// Also defines D_HardFloat or D_SoftFloat depending if FPU should be used
void registerPredefinedFloatABI(const char *soft, const char *hard,
                                const char *softfp = nullptr) {
// Use target floating point unit instead of s/w float routines
#if LDC_LLVM_VER >= 307
  // FIXME: This is a semantic change!
  bool useFPU = gTargetMachine->Options.FloatABIType == llvm::FloatABI::Hard;
#else
  bool useFPU = !gTargetMachine->Options.UseSoftFloat;
#endif
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
  switch (global.params.targetTriple->getArch()) {
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
    if (global.params.targetTriple->getOS() == llvm::Triple::Linux) {
      VersionCondition::addPredefinedGlobalIdent(
          global.params.targetTriple->getArch() == llvm::Triple::ppc64
              ? "ELFv1"
              : "ELFv2");
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
#if LDC_LLVM_VER == 305
  case llvm::Triple::arm64:
  case llvm::Triple::arm64_be:
#endif
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    VersionCondition::addPredefinedGlobalIdent("AArch64");
    registerPredefinedFloatABI("ARM_SoftFloat", "ARM_HardFloat", "ARM_SoftFP");
    break;
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    VersionCondition::addPredefinedGlobalIdent("MIPS");
    registerPredefinedFloatABI("MIPS_SoftFloat", "MIPS_HardFloat");
    registerMipsABI();
    break;
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    VersionCondition::addPredefinedGlobalIdent("MIPS64");
    registerPredefinedFloatABI("MIPS_SoftFloat", "MIPS_HardFloat");
    registerMipsABI();
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
  default:
    error(Loc(), "invalid cpu architecture specified: %s",
          global.params.targetTriple->getArchName().str().c_str());
    fatal();
  }

  // endianness
  if (gDataLayout->isLittleEndian()) {
    VersionCondition::addPredefinedGlobalIdent("LittleEndian");
  } else {
    VersionCondition::addPredefinedGlobalIdent("BigEndian");
  }

  // a generic 64bit version
  if (global.params.isLP64) {
    VersionCondition::addPredefinedGlobalIdent("D_LP64");
  }

  if (gTargetMachine->getRelocationModel() == llvm::Reloc::PIC_) {
    VersionCondition::addPredefinedGlobalIdent("D_PIC");
  }

  // parse the OS out of the target triple
  // see http://gcc.gnu.org/install/specific.html for details
  // also llvm's different SubTargets have useful information
  switch (global.params.targetTriple->getOS()) {
  case llvm::Triple::Win32:
    VersionCondition::addPredefinedGlobalIdent("Windows");
    VersionCondition::addPredefinedGlobalIdent(global.params.is64bit ? "Win64"
                                                                     : "Win32");
    if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Microsoft");
    }
    if (global.params.targetTriple->isWindowsGNUEnvironment()) {
      VersionCondition::addPredefinedGlobalIdent(
          "mingw32"); // For backwards compatibility.
      VersionCondition::addPredefinedGlobalIdent("MinGW");
    }
    if (global.params.targetTriple->isWindowsCygwinEnvironment()) {
      error(Loc(), "Cygwin is not yet supported");
      fatal();
      VersionCondition::addPredefinedGlobalIdent("Cygwin");
    }
    break;
  case llvm::Triple::Linux:
    VersionCondition::addPredefinedGlobalIdent("linux");
    VersionCondition::addPredefinedGlobalIdent("Posix");
    if (global.params.targetTriple->getEnvironment() == llvm::Triple::Android) {
      VersionCondition::addPredefinedGlobalIdent("Android");
      VersionCondition::addPredefinedGlobalIdent("CRuntime_Bionic");
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
    switch (global.params.targetTriple->getEnvironment()) {
    case llvm::Triple::Android:
      VersionCondition::addPredefinedGlobalIdent("Android");
      break;
    default:
      error(Loc(), "target '%s' is not yet supported",
            global.params.targetTriple->str().c_str());
      fatal();
    }
  }
}

/// Registers all predefined D version identifiers for the current
/// configuration with VersionCondition.
void registerPredefinedVersions() {
  VersionCondition::addPredefinedGlobalIdent("LDC");
  VersionCondition::addPredefinedGlobalIdent("all");
  VersionCondition::addPredefinedGlobalIdent("D_Version2");

  if (global.params.doDocComments) {
    VersionCondition::addPredefinedGlobalIdent("D_Ddoc");
  }

  if (global.params.cov) {
    VersionCondition::addPredefinedGlobalIdent("D_Coverage");
  }

  if (global.params.useUnitTests) {
    VersionCondition::addPredefinedGlobalIdent("unittest");
  }

  if (global.params.useAssert) {
    VersionCondition::addPredefinedGlobalIdent("assert");
  }

  if (global.params.useArrayBounds == BOUNDSCHECKoff) {
    VersionCondition::addPredefinedGlobalIdent("D_NoBoundsChecks");
  }

  registerPredefinedTargetVersions();

  if (global.params.hasObjectiveC) {
    VersionCondition::addPredefinedGlobalIdent("D_ObjectiveC");
  }

  // Pass sanitizer arguments to linker. Requires clang.
  if (opts::sanitize == opts::AddressSanitizer) {
    VersionCondition::addPredefinedGlobalIdent("LDC_AddressSanitizer");
  }

  if (opts::sanitize == opts::MemorySanitizer) {
    VersionCondition::addPredefinedGlobalIdent("LDC_MemorySanitizer");
  }

  if (opts::sanitize == opts::ThreadSanitizer) {
    VersionCondition::addPredefinedGlobalIdent("LDC_ThreadSanitizer");
  }

// Expose LLVM version to runtime
#define STR(x) #x
#define XSTR(x) STR(x)
  VersionCondition::addPredefinedGlobalIdent("LDC_LLVM_" XSTR(LDC_LLVM_VER));
#undef XSTR
#undef STR
}

/// Dump all predefined version identifiers.
void dumpPredefinedVersions() {
  if (global.params.verbose && global.params.versionids) {
    fprintf(global.stdmsg, "predefs  ");
    int col = 10;
    for (auto id : *global.params.versionids) {
      int len = strlen(id) + 1;
      if (col + len > 80) {
        col = 10;
        fprintf(global.stdmsg, "\n         ");
      }
      col += len;
      fprintf(global.stdmsg, " %s", id);
    }
    fprintf(global.stdmsg, "\n");
  }
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

  if (files.dim == 0 && !helpOnly) {
    cl::PrintHelpMessage();
    return EXIT_FAILURE;
  }

  if (global.errors) {
    fatal();
  }

  // Set up the TargetMachine.
  ExplicitBitness::Type bitness = ExplicitBitness::None;
  if ((m32bits || m64bits) && (!mArch.empty() || !mTargetTriple.empty())) {
    error(Loc(), "-m32 and -m64 switches cannot be used together with -march "
                 "and -mtriple switches");
  }

  if (m32bits) {
    bitness = ExplicitBitness::M32;
  }
  if (m64bits) {
    if (bitness != ExplicitBitness::None) {
      error(Loc(), "cannot use both -m32 and -m64 options");
    }
    bitness = ExplicitBitness::M64;
  }

  if (global.errors) {
    fatal();
  }

  gTargetMachine = createTargetMachine(
      mTargetTriple, mArch, mCPU, mAttrs, bitness, mFloatABI, getRelocModel(),
      mCodeModel, codeGenOptLevel(), disableFpElim, disableLinkerStripDead);

#if LDC_LLVM_VER >= 308
  static llvm::DataLayout DL = gTargetMachine->createDataLayout();
  gDataLayout = &DL;
#elif LDC_LLVM_VER >= 307
  gDataLayout = gTargetMachine->getDataLayout();
#elif LDC_LLVM_VER >= 306
  gDataLayout = gTargetMachine->getSubtargetImpl()->getDataLayout();
#else
  gDataLayout = gTargetMachine->getDataLayout();
#endif

  {
    llvm::Triple *triple = new llvm::Triple(gTargetMachine->getTargetTriple());
    global.params.targetTriple = triple;
    global.params.isWindows = triple->isOSWindows();
    global.params.isLP64 = gDataLayout->getPointerSizeInBits() == 64;
    global.params.is64bit = triple->isArch64Bit();
    global.params.hasObjectiveC = objc_isSupported(*triple);
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
    global.dll_ext = "so";
    global.lib_ext = "a";
  }

  Strings libmodules;
  return mars_mainBody(files, libmodules);
}

void addDefaultVersionIdentifiers() {
  registerPredefinedVersions();
  dumpPredefinedVersions();
}

void codegenModules(Modules &modules) {
  // Generate one or more object/IR/bitcode files.
  if (global.params.obj && !modules.empty()) {
    ldc::CodeGenerator cg(getGlobalContext(), global.params.oneobj);

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
        fprintf(global.stdmsg, "code      %s\n", m->toChars());

      cg.emit(m);

      if (global.errors)
        fatal();
    }
  }

  cache::pruneCache();

  freeRuntime();
  llvm::llvm_shutdown();
}
