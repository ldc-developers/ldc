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
    importPaths("I", cl::desc("Look for imports also in <directory>"),
                cl::value_desc("directory"), cl::location(impPathsStore),
                cl::Prefix);

static cl::opt<std::string>
    defaultLib("defaultlib",
               cl::desc("Default libraries to link with (overrides previous)"),
               cl::value_desc("lib1,lib2,..."), cl::ZeroOrMore);

static cl::opt<std::string> debugLib(
    "debuglib",
    cl::desc("(deprecated) Debug versions of default libraries"),
    cl::value_desc("lib1,lib2,..."), cl::ZeroOrMore);

static cl::opt<bool> linkDebugLib(
    "link-debuglib",
    cl::desc("Link with debug versions of default libraries"),
    cl::ZeroOrMore);

static cl::opt<bool> linkSharedLib(
    "link-sharedlib",
    cl::desc("Link with shared versions of default libraries"),
    cl::ZeroOrMore);

static cl::opt<bool> staticFlag(
    "static",
    cl::desc(
        "Create a statically linked binary, including all system dependencies"),
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
      printf("\n"
             "Language changes listed by -transition=id:\n"
             "  =all           list information on all language changes\n"
             "  =checkimports  give deprecation messages about 10378 "
             "anomalies\n"
             "  =complex,14488 list all usages of complex or imaginary types\n"
             "  =field,3449    list all non-mutable fields which occupy an "
             "object instance\n"
             "  =import,10378  revert to single phase name lookup\n"
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

  // Set up `opts::allArguments`, the combined list of command line arguments.
  using opts::allArguments;

  // initialize with the actual command line
  allArguments.insert(allArguments.end(), argv, argv + argc);

  // expand response files (`@<file>`) in-place
  llvm::BumpPtrAllocator allocator;
  expandResponseFiles(allocator, allArguments);

  // read config file
  ConfigFile cfg_file;
  const char *explicitConfFile = tryGetExplicitConfFile(allArguments);
  const std::string cfg_triple = tryGetExplicitTriple(allArguments).getTriple();
  // just ignore errors for now, they are still printed
  cfg_file.read(explicitConfFile, cfg_triple.c_str());

  // insert switches from config file before all explicit ones
  allArguments.insert(allArguments.begin() + 1, cfg_file.switches_begin(),
                      cfg_file.switches_end());

  // finalize by expanding response files specified in config file
  expandResponseFiles(allocator, allArguments);

  cl::SetVersionPrinter(&printVersion);

  opts::hideLLVMOptions();
  opts::createClashingOptions();

  cl::ParseCommandLineOptions(allArguments.size(),
                              const_cast<char **>(allArguments.data()),
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
      fprintf(global.stdmsg, "config    %s (%s)\n", path.c_str(),
              cfg_triple.c_str());
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

  if (linkSharedLib && staticFlag)
  {
      error(
        Loc(),
        "Can't use -link-sharedlib and -static together"
      );
  }

  if (noDefaultLib) {
    deprecation(
        Loc(),
        "-nodefaultlib is deprecated, as "
        "-defaultlib now overrides the existing list instead of "
        "appending to it. Please use the latter instead.");
  } else {
    // Parse comma-separated default library list.
    bool generatedDebugLib = false;

    if (debugLib.length() > 0)
    {
        // temporarily disabled to not affect expected test output
        /*
        deprecation(
            Loc(),
            "-debuglib is deprecated, as LDC generates names of "
            "debug libraries automatically now by appending '-debug' "
            "suffix to default ones"
        );
        */
    }
    else
        generatedDebugLib = true;

    std::stringstream libNames(linkDebugLib && !generatedDebugLib
        ? debugLib : defaultLib);

    while (libNames.good()) {
      std::string lib;
      std::getline(libNames, lib, ',');
      if (lib.empty()) {
        continue;
      }

      size_t size = lib.size() + 3;
      if (linkDebugLib && generatedDebugLib)
          size += 6;
      if (linkSharedLib)
          size += 7;
      char *arg = static_cast<char *>(mem.xmalloc(size));
      strcpy(arg, "-l");
      strcpy(arg + 2, lib.c_str());
      if (linkDebugLib && generatedDebugLib)
          strcpy(arg + lib.length(), "-debug");
      if (linkSharedLib)
          strcpy(arg + size - 8, "-shared");

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

  global.params.hdrStripPlainFunctions = !opts::hdrKeepAllBodies;
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
  const auto arch = global.params.targetTriple->getArch();

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

  /* LDC doesn't support DMD's core.simd interface.
  if (arch == llvm::Triple::x86 || arch == llvm::Triple::x86_64) {
    if (traitsTargetHasFeature("sse2"))
      VersionCondition::addPredefinedGlobalIdent("D_SIMD");
    if (traitsTargetHasFeature("avx"))
      VersionCondition::addPredefinedGlobalIdent("D_AVX");
  }
  */

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
  if ((m32bits || m64bits) && (!mArch.empty() || !mTargetTriple.empty())) {
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

  opts::setDefaultMathOptions(*gTargetMachine);

  // allocate the target abi
  gABI = TargetABI::getTarget();

  if (global.params.targetTriple->isOSWindows()) {
    global.dll_ext = "dll";
    global.lib_ext = (global.params.mscoff ? "lib" : "a");
  } else {
    global.dll_ext = global.params.targetTriple->isOSDarwin() ? "dylib" : "so";
    global.lib_ext = "a";
  }

  Strings libmodules;
  return mars_mainBody(files, libmodules);
}

void addDefaultVersionIdentifiers() {
  registerPredefinedVersions();
  printPredefinedVersions();
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
