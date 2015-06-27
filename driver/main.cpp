//===-- main.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "module.h"
#include "color.h"
#include "doc.h"
#include "id.h"
#include "hdrgen.h"
#include "json.h"
#include "mars.h"
#include "mtype.h"
#include "parse.h"
#include "rmem.h"
#include "root.h"
#include "scope.h"
#include "dmd2/target.h"
#include "driver/cl_options.h"
#include "driver/configfile.h"
#include "driver/ldc-version.h"
#include "driver/linker.h"
#include "driver/targetmachine.h"
#include "driver/toobj.h"
#include "gen/cl_helpers.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "gen/metadata.h"
#include "gen/optimizer.h"
#include "gen/passes/Passes.h"
#include "gen/runtime.h"
#include "gen/abi.h"
#if LDC_LLVM_VER >= 304
#include "llvm/InitializePasses.h"
#endif
#include "llvm/LinkAllPasses.h"
#if LDC_LLVM_VER >= 305
#include "llvm/Linker/Linker.h"
#else
#include "llvm/Linker.h"
#endif
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#if LDC_LLVM_VER >= 306
#include "llvm/Target/TargetSubtargetInfo.h"
#endif
#if LDC_LLVM_VER >= 303
#include "llvm/LinkAllIR.h"
#include "llvm/IR/LLVMContext.h"
#else
#include "llvm/LinkAllVMCore.h"
#include "llvm/LLVMContext.h"
#endif
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#if LDC_POSIX
#include <errno.h>
#elif _WIN32
#include <windows.h>
#endif

// Needs Type already declared.
#include "cond.h"

// in traits.c
void initTraitsStringTable();

using namespace opts;

extern void getenv_setargv(const char *envvar, int *pargc, char** *pargv);

static cl::opt<bool> noDefaultLib("nodefaultlib",
    cl::desc("Don't add a default library for linking implicitly"),
    cl::ZeroOrMore,
    cl::Hidden);

static StringsAdapter impPathsStore("I", global.params.imppath);
static cl::list<std::string, StringsAdapter> importPaths("I",
    cl::desc("Where to look for imports"),
    cl::value_desc("path"),
    cl::location(impPathsStore),
    cl::Prefix);

static cl::opt<std::string> defaultLib("defaultlib",
    cl::desc("Default libraries for non-debug-info build (overrides previous)"),
    cl::value_desc("lib1,lib2,..."),
    cl::ZeroOrMore);

static cl::opt<std::string> debugLib("debuglib",
    cl::desc("Default libraries for debug info build (overrides previous)"),
    cl::value_desc("lib1,lib2,..."),
    cl::ZeroOrMore);


#if LDC_LLVM_VER < 304
namespace llvm {
namespace sys {
namespace fs {
static std::string getMainExecutable(const char *argv0, void *MainExecAddr) {
  return llvm::sys::Path::GetMainExecutable(argv0, MainExecAddr).str();
}
}
}
}
#endif

void printVersion() {
    printf("LDC - the LLVM D compiler (%s):\n", global.ldc_version);
    printf("  based on DMD %s and LLVM %s\n", global.version, global.llvm_version);
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    printf("  compiled with address sanitizer enabled\n");
#endif
#endif
    printf("  Default target: %s\n", llvm::sys::getDefaultTargetTriple().c_str());
    std::string CPU = llvm::sys::getHostCPUName();
    if (CPU == "generic") CPU = "(unknown)";
    printf("  Host CPU: %s\n", CPU.c_str());
    printf("  http://dlang.org - http://wiki.dlang.org/LDC\n");
    printf("\n");

    // Without explicitly flushing here, only the target list is visible when
    // redirecting stdout to a file.
    fflush(stdout);

    llvm::TargetRegistry::printRegisteredTargetsForVersion();
    exit(EXIT_SUCCESS);
}

// Helper function to handle -d-debug=* and -d-version=*
static void processVersions(std::vector<std::string>& list, const char* type,
        void (*setLevel)(unsigned), void (*addIdent)(const char*)) {
    typedef std::vector<std::string>::iterator It;

    for(It I = list.begin(), E = list.end(); I != E; ++I) {
        const char* value = I->c_str();
        if (isdigit(value[0])) {
            errno = 0;
            char* end;
            long level = strtol(value, &end, 10);
            if (*end || errno || level > INT_MAX) {
                error(Loc(), "Invalid %s level: %s", type, I->c_str());
            } else {
                setLevel((unsigned)level);
            }
        } else {
            char* cstr = mem.strdup(value);
            if (Lexer::isValidIdentifier(cstr)) {
                addIdent(cstr);
                continue;
            } else {
                error(Loc(), "Invalid %s identifier or level: '%s'", type, I->c_str());
            }
        }
    }
}

// Helper function to handle -of, -od, etc.
static void initFromString(const char*& dest, const cl::opt<std::string>& src) {
    dest = 0;
    if (src.getNumOccurrences() != 0) {
        if (src.empty())
            error(Loc(), "Expected argument to '-%s'", src.ArgStr);
        dest = mem.strdup(src.c_str());
    }
}


#if LDC_LLVM_VER >= 303
static void hide(llvm::StringMap<cl::Option *>& map, const char* name) {
    // Check if option exists first for resilience against LLVM changes
    // between versions.
    if (map.count(name))
        map[name]->setHiddenFlag(cl::Hidden);
}

/// Removes command line options exposed from within LLVM that are unlikely
/// to be useful for end users from the -help output.
static void hideLLVMOptions() {
#if LDC_LLVM_VER >= 307
    llvm::StringMap<cl::Option *>& map = cl::getRegisteredOptions();
#else
    llvm::StringMap<cl::Option *> map;
    cl::getRegisteredOptions(map);
#endif
    hide(map, "bounds-checking-single-trap");
    hide(map, "disable-debug-info-verifier");
    hide(map, "disable-spill-fusing");
    hide(map, "cppfname");
    hide(map, "cppfor");
    hide(map, "cppgen");
    hide(map, "enable-correct-eh-support");
    hide(map, "enable-load-pre");
    hide(map, "enable-misched");
    hide(map, "enable-objc-arc-opts");
    hide(map, "enable-tbaa");
    hide(map, "exhaustive-register-search");
    hide(map, "fatal-assembler-warnings");
    hide(map, "internalize-public-api-file");
    hide(map, "internalize-public-api-list");
    hide(map, "join-liveintervals");
    hide(map, "limit-float-precision");
    hide(map, "mc-x86-disable-arith-relaxation");
    hide(map, "mlsm");
    hide(map, "mno-ldc1-sdc1");
    hide(map, "nvptx-sched4reg");
    hide(map, "no-discriminators");
    hide(map, "pre-RA-sched");
    hide(map, "print-after-all");
    hide(map, "print-before-all");
    hide(map, "print-machineinstrs");
    hide(map, "profile-estimator-loop-weight");
    hide(map, "profile-estimator-loop-weight");
    hide(map, "profile-file");
    hide(map, "profile-info-file");
    hide(map, "profile-verifier-noassert");
    hide(map, "regalloc");
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

    // We enable -fdata-sections/-ffunction-sections by default where it makes
    // sense for reducing code size, so hide them to avoid confusion.
    //
    // We need our own switch as these two are defined by LLVM and linked to
    // static TargetMachine members, but the default we want to use depends
    // on the target triple (and thus we do not know it until after the command
    // line has been parsed).
    hide(map, "fdata-sections");
    hide(map, "ffunction-sections");
}
#endif

int main(int argc, char **argv);

/// Parses switches from the command line, any response files and the global
/// config file and sets up global.params accordingly.
///
/// Returns a list of source file names.
static void parseCommandLine(int argc, char **argv, Strings &sourceFiles, bool &helpOnly) {
#if _WIN32
    char buf[MAX_PATH];
    GetModuleFileName(NULL, buf, MAX_PATH);
    const char* argv0 = &buf[0];
    // FIXME: We cannot set params.argv0 here, as we would escape a stack
    // reference, but it is unused anyway.
    global.params.argv0 = NULL;
#else
    const char* argv0 = global.params.argv0 = argv[0];
#endif

    // Set some default values.
    global.params.useSwitchError = 1;
    global.params.useArrayBounds = 2;
    global.params.color = isConsoleColorSupported();

    global.params.linkswitches = new Strings();
    global.params.libfiles = new Strings();
    global.params.objfiles = new Strings();
    global.params.ddocfiles = new Strings();

    global.params.moduleDeps = NULL;
    global.params.moduleDepsFile = NULL;

    // Build combined list of command line arguments.
    std::vector<const char*> final_args;
    final_args.push_back(argv[0]);

    ConfigFile cfg_file;
    // just ignore errors for now, they are still printed
    cfg_file.read(argv0, (void*)main, "ldc2.conf");
    final_args.insert(final_args.end(), cfg_file.switches_begin(), cfg_file.switches_end());

    final_args.insert(final_args.end(), &argv[1], &argv[argc]);

    cl::SetVersionPrinter(&printVersion);
#if LDC_LLVM_VER >= 303
    hideLLVMOptions();
#endif
    cl::ParseCommandLineOptions(final_args.size(), const_cast<char**>(&final_args[0]),
        "LDC - the LLVM D compiler\n"
#if LDC_LLVM_VER < 302
        , true
#endif
    );

    helpOnly = mCPU == "help" ||
        (std::find(mAttrs.begin(), mAttrs.end(), "help") != mAttrs.end());

    // Print some information if -v was passed
    // - path to compiler binary
    // - version number
    // - used config file
    if (global.params.verbose) {
        fprintf(global.stdmsg, "binary    %s\n", llvm::sys::fs::getMainExecutable(argv0, (void*)main).c_str());
        fprintf(global.stdmsg, "version   %s (DMD %s, LLVM %s)\n", global.ldc_version, global.version, global.llvm_version);
        const std::string& path = cfg_file.path();
        if (!path.empty())
            fprintf(global.stdmsg, "config    %s\n", path.c_str());
    }

    // Negated options
    global.params.link = !compileOnly;
    global.params.obj = !dontWriteObj;
    global.params.useInlineAsm = !noAsm;

    // String options: std::string --> char*
    initFromString(global.params.objname, objectFile);
    initFromString(global.params.objdir, objectDir);

    initFromString(global.params.docdir, ddocDir);
    initFromString(global.params.docname, ddocFile);
    global.params.doDocComments |=
        global.params.docdir || global.params.docname;

    initFromString(global.params.jsonfilename, jsonFile);
    if (global.params.jsonfilename)
        global.params.doJsonGeneration = true;

    initFromString(global.params.hdrdir, hdrDir);
    initFromString(global.params.hdrname, hdrFile);
    global.params.doHdrGeneration |=
        global.params.hdrdir || global.params.hdrname;

    initFromString(global.params.moduleDepsFile, moduleDepsFile);
    if (global.params.moduleDepsFile != NULL)
    {
         global.params.moduleDeps = new OutBuffer;
    }

    processVersions(debugArgs, "debug",
        DebugCondition::setGlobalLevel,
        DebugCondition::addGlobalIdent);
    processVersions(versions, "version",
        VersionCondition::setGlobalLevel,
        VersionCondition::addGlobalIdent);

    global.params.output_o =
        (opts::output_o == cl::BOU_UNSET
            && !(opts::output_bc || opts::output_ll || opts::output_s))
        ? OUTPUTFLAGdefault
        : opts::output_o == cl::BOU_TRUE
            ? OUTPUTFLAGset
            : OUTPUTFLAGno;
    global.params.output_bc = opts::output_bc ? OUTPUTFLAGset : OUTPUTFLAGno;
    global.params.output_ll = opts::output_ll ? OUTPUTFLAGset : OUTPUTFLAGno;
    global.params.output_s  = opts::output_s  ? OUTPUTFLAGset : OUTPUTFLAGno;

    global.params.cov = (global.params.covPercent <= 100);

    templateLinkage =
        opts::linkonceTemplates ? LLGlobalValue::LinkOnceODRLinkage
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
            char const * name = runargs[0].c_str();
            char const * ext = FileName::ext(name);
            if (ext && FileName::equals(ext, "d") == 0 &&
                FileName::equals(ext, "di") == 0) {
                error(Loc(), "-run must be followed by a source file, not '%s'", name);
            }

            sourceFiles.push(mem.strdup(name));
            runargs.erase(runargs.begin());
        } else {
            global.params.run = false;
            error(Loc(), "Expected at least one argument to '-run'\n");
        }
    }

    sourceFiles.reserve(fileList.size());
    typedef std::vector<std::string>::iterator It;
    for(It I = fileList.begin(), E = fileList.end(); I != E; ++I)
        if (!I->empty())
            sourceFiles.push(mem.strdup(I->c_str()));

    if (noDefaultLib)
    {
        deprecation(Loc(), "-nodefaultlib is deprecated, as "
            "-defaultlib/-debuglib now override the existing list instead of "
            "appending to it. Please use the latter instead.");
    }
    else
    {
        // Parse comma-separated default library list.
        std::stringstream libNames(
            global.params.symdebug ? debugLib : defaultLib);
        while (libNames.good())
        {
            std::string lib;
            std::getline(libNames, lib, ',');
            if (lib.empty()) continue;

            char *arg = static_cast<char *>(mem.malloc(lib.size() + 3));
            strcpy(arg, "-l");
            strcpy(arg+2, lib.c_str());
            global.params.linkswitches->push(arg);
        }
    }

    if (global.params.useUnitTests)
        global.params.useAssert = 1;

    // -release downgrades default bounds checking level to BC_SafeOnly (only for safe functions).
    global.params.useArrayBounds = opts::nonSafeBoundsChecks ? opts::BC_On : opts::BC_SafeOnly;
    if (opts::boundsCheck != opts::BC_Default)
        global.params.useArrayBounds = opts::boundsCheck;

    // LDC output determination

    // if we don't link, autodetect target from extension
    if(!global.params.link && !createStaticLib && global.params.objname) {
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
        } else if (strcmp(ext, global.obj_ext) == 0 || strcmp(ext, global.obj_ext_alt) == 0) {
            global.params.output_o = OUTPUTFLAGset;
            autofound = true;
        } else {
            // append dot, so forceExt won't change existing name even if it contains dots
            size_t len = strlen(global.params.objname);
            char* s = static_cast<char *>(mem.malloc(len + 1 + 1));
            memcpy(s, global.params.objname, len);
            s[len] = '.';
            s[len+1] = 0;
            global.params.objname = s;

        }
        if(autofound && global.params.output_o == OUTPUTFLAGdefault)
            global.params.output_o = OUTPUTFLAGno;
    }

    // only link if possible
    if (!global.params.obj || !global.params.output_o || createStaticLib)
        global.params.link = 0;

    if (createStaticLib && createSharedLib)
        error(Loc(), "-lib and -shared switches cannot be used together");

    if (createSharedLib && mRelocModel == llvm::Reloc::Default)
        mRelocModel = llvm::Reloc::PIC_;

    if (global.params.link && !createSharedLib)
    {
        global.params.exefile = global.params.objname;
        if (sourceFiles.dim > 1)
            global.params.objname = NULL;
    }
    else if (global.params.run)
    {
        error(Loc(), "flags conflict with -run");
    }
    else if (global.params.objname && sourceFiles.dim > 1) {
        if (!(createStaticLib || createSharedLib) && !singleObj)
        {
            error(Loc(), "multiple source files, but only one .obj name");
        }
    }

    if (soname.getNumOccurrences() > 0 && !createSharedLib) {
        error(Loc(), "-soname can be used only when building a shared library");
    }
}

static void initializePasses() {
#if LDC_LLVM_VER >= 304
    using namespace llvm;
    // Initialize passes
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeCore(Registry);
#if LDC_LLVM_VER < 306
    initializeDebugIRPass(Registry);
#endif
    initializeScalarOpts(Registry);
    initializeVectorization(Registry);
    initializeIPO(Registry);
    initializeAnalysis(Registry);
    initializeIPA(Registry);
    initializeTransformUtils(Registry);
    initializeInstCombine(Registry);
    initializeInstrumentation(Registry);
    initializeTarget(Registry);
    // For codegen passes, only passes that do IR to IR transformation are
    // supported. For now, just add CodeGenPrepare.
    initializeCodeGenPreparePass(Registry);
#if LDC_LLVM_VER >= 306
    initializeAtomicExpandPass(Registry);
    initializeRewriteSymbolsPass(Registry);
#elif LDC_LLVM_VER == 305
    initializeAtomicExpandLoadLinkedPass(Registry);
#endif
#endif
}

/// Register the MIPS ABI.
static void registerMipsABI()
{
    switch (getMipsABI())
    {
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
static void registerPredefinedFloatABI(const char *soft, const char *hard, const char *softfp=NULL)
{
    // Use target floating point unit instead of s/w float routines
#if LDC_LLVM_VER >= 307
    // FIXME: This is a semantic change!
    bool useFPU = gTargetMachine->Options.FloatABIType == llvm::FloatABI::Hard;
#else
    bool useFPU = !gTargetMachine->Options.UseSoftFloat;
#endif
    VersionCondition::addPredefinedGlobalIdent(useFPU ? "D_HardFloat" : "D_SoftFloat");

    if (gTargetMachine->Options.FloatABIType == llvm::FloatABI::Soft)
    {
        VersionCondition::addPredefinedGlobalIdent(useFPU && softfp ? softfp : soft);
    }
    else if (gTargetMachine->Options.FloatABIType == llvm::FloatABI::Hard)
    {
        assert(useFPU && "Should be using the FPU if using float-abi=hard");
        VersionCondition::addPredefinedGlobalIdent(hard);
    }
    else
    {
        assert(0 && "FloatABIType neither Soft or Hard");
    }
}

/// Registers the predefined versions specific to the current target triple
/// and other target specific options with VersionCondition.
static void registerPredefinedTargetVersions() {
    switch (global.params.targetTriple.getArch())
    {
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
#if LDC_LLVM_VER >= 305
        case llvm::Triple::ppc64le:
#endif
            VersionCondition::addPredefinedGlobalIdent("PPC64");
            registerPredefinedFloatABI("PPC_SoftFloat", "PPC_HardFloat");
            if (global.params.targetTriple.getOS() == llvm::Triple::Linux)
                VersionCondition::addPredefinedGlobalIdent(global.params.targetTriple.getArch() == llvm::Triple::ppc64
                                                       ? "ELFv1" : "ELFv2");
            break;
        case llvm::Triple::arm:
#if LDC_LLVM_VER >= 305
        case llvm::Triple::armeb:
#endif
            VersionCondition::addPredefinedGlobalIdent("ARM");
            registerPredefinedFloatABI("ARM_SoftFloat", "ARM_HardFloat", "ARM_SoftFP");
            break;
        case llvm::Triple::thumb:
            VersionCondition::addPredefinedGlobalIdent("ARM");
            VersionCondition::addPredefinedGlobalIdent("Thumb"); // For backwards compatibility.
            VersionCondition::addPredefinedGlobalIdent("ARM_Thumb");
            registerPredefinedFloatABI("ARM_SoftFloat", "ARM_HardFloat", "ARM_SoftFP");
            break;
#if LDC_LLVM_VER == 305
        case llvm::Triple::arm64:
        case llvm::Triple::arm64_be:
#endif
#if LDC_LLVM_VER >= 303
        case llvm::Triple::aarch64:
#if LDC_LLVM_VER >= 305
        case llvm::Triple::aarch64_be:
#endif
            VersionCondition::addPredefinedGlobalIdent("AArch64");
            registerPredefinedFloatABI("ARM_SoftFloat", "ARM_HardFloat", "ARM_SoftFP");
            break;
#endif
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
#if LDC_LLVM_VER >= 302
        case llvm::Triple::nvptx:
            VersionCondition::addPredefinedGlobalIdent("NVPTX");
            VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
            break;
        case llvm::Triple::nvptx64:
            VersionCondition::addPredefinedGlobalIdent("NVPTX64");
            VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
            break;
#endif
        default:
            error(Loc(), "invalid cpu architecture specified: %s", global.params.targetTriple.getArchName().str().c_str());
            fatal();
    }

    // endianness
    if (gDataLayout->isLittleEndian()) {
        VersionCondition::addPredefinedGlobalIdent("LittleEndian");
    }
    else {
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
    switch (global.params.targetTriple.getOS())
    {
        case llvm::Triple::Win32:
            VersionCondition::addPredefinedGlobalIdent("Windows");
            VersionCondition::addPredefinedGlobalIdent(global.params.is64bit ? "Win64" : "Win32");
#if LDC_LLVM_VER >= 305
            if (global.params.targetTriple.isWindowsGNUEnvironment())
            {
                VersionCondition::addPredefinedGlobalIdent("mingw32"); // For backwards compatibility.
                VersionCondition::addPredefinedGlobalIdent("MinGW");
            }
            if (global.params.targetTriple.isWindowsCygwinEnvironment())
            {
                error(Loc(), "Cygwin is not yet supported");
                fatal();
                VersionCondition::addPredefinedGlobalIdent("Cygwin");
            }
            break;
#else
            break;
        case llvm::Triple::MinGW32:
            VersionCondition::addPredefinedGlobalIdent("Windows");
            VersionCondition::addPredefinedGlobalIdent(global.params.is64bit ? "Win64" : "Win32");
            VersionCondition::addPredefinedGlobalIdent("mingw32"); // For backwards compatibility.
            VersionCondition::addPredefinedGlobalIdent("MinGW");
            break;
        case llvm::Triple::Cygwin:
            error(Loc(), "Cygwin is not yet supported");
            fatal();
            VersionCondition::addPredefinedGlobalIdent("Cygwin");
            break;
#endif
        case llvm::Triple::Linux:
#if LDC_LLVM_VER >= 302
            if (global.params.targetTriple.getEnvironment() == llvm::Triple::Android)
            {
                VersionCondition::addPredefinedGlobalIdent("Android");
            }
            else
#endif
            {
                VersionCondition::addPredefinedGlobalIdent("linux");
                VersionCondition::addPredefinedGlobalIdent("Posix");
            }
            break;
        case llvm::Triple::Haiku:
            VersionCondition::addPredefinedGlobalIdent("Haiku");
            VersionCondition::addPredefinedGlobalIdent("Posix");
            break;
        case llvm::Triple::Darwin:
            VersionCondition::addPredefinedGlobalIdent("OSX");
            VersionCondition::addPredefinedGlobalIdent("darwin"); // For backwards compatibility.
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
#if LDC_LLVM_VER >= 302
        case llvm::Triple::AIX:
            VersionCondition::addPredefinedGlobalIdent("AIX");
            VersionCondition::addPredefinedGlobalIdent("Posix");
            break;
#endif
        default:
            switch (global.params.targetTriple.getEnvironment())
            {
#if LDC_LLVM_VER >= 302
                case llvm::Triple::Android:
                    VersionCondition::addPredefinedGlobalIdent("Android");
                    break;
#endif
                default:
                    error(Loc(), "target '%s' is not yet supported", global.params.targetTriple.str().c_str());
                    fatal();
            }
    }
}

/// Registers all predefined D version identifiers for the current
/// configuration with VersionCondition.
static void registerPredefinedVersions() {
    VersionCondition::addPredefinedGlobalIdent("LDC");
    VersionCondition::addPredefinedGlobalIdent("all");
    VersionCondition::addPredefinedGlobalIdent("D_Version2");

    if (global.params.doDocComments)
        VersionCondition::addPredefinedGlobalIdent("D_Ddoc");

    if (global.params.useUnitTests)
        VersionCondition::addPredefinedGlobalIdent("unittest");

    if (global.params.useAssert)
        VersionCondition::addPredefinedGlobalIdent("assert");

    if (!global.params.useArrayBounds)
        VersionCondition::addPredefinedGlobalIdent("D_NoBoundsChecks");

    registerPredefinedTargetVersions();

#if LDC_LLVM_VER >= 303
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
#endif

    // Expose LLVM version to runtime
#define STR(x) #x
#define XSTR(x) STR(x)
    VersionCondition::addPredefinedGlobalIdent("LDC_LLVM_" XSTR(LDC_LLVM_VER));
#undef XSTR
#undef STR
}

/// Dump all predefined version identifiers.
static void dumpPredefinedVersions()
{
    if (global.params.verbose && global.params.versionids)
    {
        fprintf(global.stdmsg, "predefs  ");
        int col = 10;
        for (Strings::iterator I = global.params.versionids->begin(),
                               E = global.params.versionids->end();
             I != E; ++I)
        {
            int len = strlen(*I) + 1;
            if (col + len > 80)
            {
                col = 10;
                fprintf(global.stdmsg, "\n         ");
            }
            col += len;
            fprintf(global.stdmsg, " %s", *I);
        }
        fprintf(global.stdmsg, "\n");
    }
}

static Module *entrypoint = NULL;
static Module *rootHasMain = NULL;

/// Callback to generate a C main() function, invoked by the frontend.
void genCmain(Scope *sc)
{
    if (entrypoint)
        return;

    /* The D code to be generated is provided as D source code in the form of a string.
     * Note that Solaris, for unknown reasons, requires both a main() and an _main()
     */
    static utf8_t code[] = "extern(C) {\n\
        int _d_run_main(int argc, char **argv, void* mainFunc);\n\
        int _Dmain(char[][] args);\n\
        int main(int argc, char **argv) { return _d_run_main(argc, argv, &_Dmain); }\n\
        version (Solaris) int _main(int argc, char** argv) { return main(argc, argv); }\n\
        }\n\
        pragma(LDC_no_moduleinfo);\n\
        ";

    Identifier *id = Id::entrypoint;
    Module *m = new Module("__entrypoint.d", id, 0, 0);

    Parser p(m, code, sizeof(code) / sizeof(code[0]), 0);
    p.scanloc = Loc();
    p.nextToken();
    m->members = p.parseModule();
    assert(p.token.value == TOKeof);

    char v = global.params.verbose;
    global.params.verbose = 0;
    m->importedFrom = m;
    m->importAll(NULL);
    m->semantic();
    m->semantic2();
    m->semantic3();
    global.params.verbose = v;

    entrypoint = m;
    rootHasMain = sc->module;
}

/// Emits a declaration for the given symbol, which is assumed to be of type
/// i8*, and defines a second globally visible i8* that contains the address
/// of the first symbol.
static void emitSymbolAddrGlobal(llvm::Module& lm, const char* symbolName,
    const char* addrName)
{
    llvm::Type* voidPtr = llvm::PointerType::get(
        llvm::Type::getInt8Ty(lm.getContext()), 0);
    llvm::GlobalVariable* targetSymbol = new llvm::GlobalVariable(
        lm, voidPtr, false, llvm::GlobalValue::ExternalWeakLinkage,
        NULL, symbolName
    );
    new llvm::GlobalVariable(lm, voidPtr, false,
        llvm::GlobalValue::ExternalLinkage,
        llvm::ConstantExpr::getBitCast(targetSymbol, voidPtr),
        addrName
    );
}

/// Adds the __entrypoint module and related support code into the given LLVM
/// module. This assumes that genCmain() has already been called.
static void emitEntryPointInto(llvm::Module* lm)
{
    assert(entrypoint && "Entry point Dmodule has not been generated.");
#if LDC_LLVM_VER >= 303
    llvm::Linker linker(lm);
#else
    llvm::Linker linker("ldc", lm);
#endif

    llvm::LLVMContext& context = lm->getContext();
    llvm::Module* entryModule = entrypoint->genLLVMModule(context);

    // On Linux, strongly define the excecutabe BSS bracketing symbols in the
    // main module for druntime use (see rt.sections_linux).
    if (global.params.isLinux)
    {
        emitSymbolAddrGlobal(*entryModule, "__bss_start", "_d_execBssBegAddr");
        emitSymbolAddrGlobal(*entryModule, "_end", "_d_execBssEndAddr");
    }

#if LDC_LLVM_VER >= 306
    // FIXME: A possible error message is written to the diagnostic context
    //        Do we show these messages?
    linker.linkInModule(entryModule);
#else
    std::string linkError;
#if LDC_LLVM_VER >= 303
    const bool hadError = linker.linkInModule(entryModule, &linkError);
#else
    const bool hadError = linker.LinkInModule(entryModule, &linkError);
    linker.releaseModule();
#endif
    if (hadError)
        error(Loc(), "%s", linkError.c_str());
#endif
}


int main(int argc, char **argv)
{
    // stack trace on signals
    llvm::sys::PrintStackTraceOnErrorSignal();

    global.init();
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

    if (files.dim == 0 && !helpOnly)
    {
        cl::PrintHelpMessage();
        return EXIT_FAILURE;
    }

    if (global.errors)
        fatal();

    // Set up the TargetMachine.
    ExplicitBitness::Type bitness = ExplicitBitness::None;
    if ((m32bits || m64bits) && (!mArch.empty() || !mTargetTriple.empty()))
        error(Loc(), "-m32 and -m64 switches cannot be used together with -march and -mtriple switches");

    if (m32bits)
        bitness = ExplicitBitness::M32;
    if (m64bits)
    {
        if (bitness != ExplicitBitness::None)
            error(Loc(), "cannot use both -m32 and -m64 options");
        bitness = ExplicitBitness::M64;
    }

    if (global.errors)
        fatal();

    gTargetMachine = createTargetMachine(mTargetTriple, mArch, mCPU, mAttrs,
        bitness, mFloatABI, mRelocModel, mCodeModel, codeGenOptLevel(),
        global.params.symdebug || disableFpElim, disableLinkerStripDead);

#if LDC_LLVM_VER >= 307
    gDataLayout = gTargetMachine->getDataLayout();
#elif LDC_LLVM_VER >= 306
    gDataLayout = gTargetMachine->getSubtargetImpl()->getDataLayout();
#elif LDC_LLVM_VER >= 302
    gDataLayout = gTargetMachine->getDataLayout();
#else
    gDataLayout = gTargetMachine->getTargetData();
#endif

    {
        llvm::Triple triple = llvm::Triple(gTargetMachine->getTargetTriple());
        global.params.targetTriple = triple;
        global.params.isLinux      = triple.getOS() == llvm::Triple::Linux;
        global.params.isOSX        = triple.isMacOSX();
        global.params.isWindows    = triple.isOSWindows();
        global.params.isFreeBSD    = triple.getOS() == llvm::Triple::FreeBSD;
        global.params.isOpenBSD    = triple.getOS() == llvm::Triple::OpenBSD;
        global.params.isSolaris    = triple.getOS() == llvm::Triple::Solaris;
        global.params.isLP64       = gDataLayout->getPointerSizeInBits() == 64;
        global.params.is64bit      = triple.isArch64Bit();
    }

    // allocate the target abi
    gABI = TargetABI::getTarget();

    // Set predefined version identifiers.
    registerPredefinedVersions();
    dumpPredefinedVersions();

    if (global.params.targetTriple.isOSWindows()) {
        global.dll_ext = "dll";
        global.lib_ext = "lib";
    } else {
        global.dll_ext = "so";
        global.lib_ext = "a";
    }

    // Initialization
    Type::init();
    Id::initialize();
    Module::init();
    Target::init();
    Expression::init();
    initPrecedence();
    builtin_init();
    initTraitsStringTable();

    // Build import search path
    if (global.params.imppath)
    {
        for (unsigned i = 0; i < global.params.imppath->dim; i++)
        {
            const char *path = static_cast<const char *>(global.params.imppath->data[i]);
            Strings *a = FileName::splitPath(path);

            if (a)
            {
                if (!global.path)
                    global.path = new Strings();
                global.path->append(a);
            }
        }
    }

    // Build string import search path
    if (global.params.fileImppath)
    {
        for (unsigned i = 0; i < global.params.fileImppath->dim; i++)
        {
            const char *path = static_cast<const char *>(global.params.fileImppath->data[i]);
            Strings *a = FileName::splitPath(path);

            if (a)
            {
                if (!global.filePath)
                    global.filePath = new Strings();
                global.filePath->append(a);
            }
        }
    }

    if (global.params.addMain)
    {
        // a dummy name, we never actually look up this file
        files.push(const_cast<char*>(global.main_d));
    }

    // Create Modules
    Modules modules;
    modules.reserve(files.dim);
    for (unsigned i = 0; i < files.dim; i++)
    {   Identifier *id;
        const char *ext;
        const char *name;

        const char *p = static_cast<const char *>(files.data[i]);

        p = FileName::name(p);      // strip path
        ext = FileName::ext(p);
        if (ext)
        {
#if LDC_POSIX
            if (strcmp(ext, global.obj_ext) == 0 ||
                strcmp(ext, global.bc_ext) == 0)
#else
            if (Port::stricmp(ext, global.obj_ext) == 0 ||
                Port::stricmp(ext, global.obj_ext_alt) == 0 ||
                Port::stricmp(ext, global.bc_ext) == 0)
#endif
            {
                global.params.objfiles->push(static_cast<const char *>(files.data[i]));
                continue;
            }

#if LDC_POSIX
            if (strcmp(ext, "a") == 0)
#elif __MINGW32__
            if (Port::stricmp(ext, "a") == 0)
#else
            if (Port::stricmp(ext, "lib") == 0)
#endif
            {
                global.params.libfiles->push(static_cast<const char *>(files.data[i]));
                continue;
            }

            if (strcmp(ext, global.ddoc_ext) == 0)
            {
                global.params.ddocfiles->push(static_cast<const char *>(files.data[i]));
                continue;
            }

            if (FileName::equals(ext, global.json_ext))
            {
                global.params.doJsonGeneration = 1;
                global.params.jsonfilename = static_cast<const char *>(files.data[i]);
                continue;
            }

#if !LDC_POSIX
            if (Port::stricmp(ext, "res") == 0)
            {
                global.params.resfile = static_cast<const char *>(files.data[i]);
                continue;
            }

            if (Port::stricmp(ext, "def") == 0)
            {
                global.params.deffile = static_cast<const char *>(files.data[i]);
                continue;
            }

            if (Port::stricmp(ext, "exe") == 0)
            {
                global.params.exefile = static_cast<const char *>(files.data[i]);
                continue;
            }
#endif

            if (Port::stricmp(ext, global.mars_ext) == 0 ||
                Port::stricmp(ext, global.hdr_ext) == 0 ||
                FileName::equals(ext, "dd"))
            {
                ext--;          // skip onto '.'
                assert(*ext == '.');
                char *tmp = static_cast<char *>(mem.malloc((ext - p) + 1));
                memcpy(tmp, p, ext - p);
                tmp[ext - p] = 0;      // strip extension
                name = tmp;

                if (name[0] == 0 ||
                    strcmp(name, "..") == 0 ||
                    strcmp(name, ".") == 0)
                {
                    goto Linvalid;
                }
            }
            else
            {   error(Loc(), "unrecognized file extension %s\n", ext);
                fatal();
            }
        }
        else
        {   name = p;
            if (!*p)
            {
        Linvalid:
                error(Loc(), "invalid file name '%s'", static_cast<const char *>(files.data[i]));
                fatal();
            }
            name = p;
        }

        id = Lexer::idPool(name);
        Module *m = new Module(static_cast<const char *>(files.data[i]), id, global.params.doDocComments, global.params.doHdrGeneration);
        modules.push(m);
    }

    // Read files, parse them
    for (unsigned i = 0; i < modules.dim; i++)
    {
        Module *m = modules[i];
        if (global.params.verbose)
            fprintf(global.stdmsg, "parse     %s\n", m->toChars());
        if (!Module::rootModule)
            Module::rootModule = m;
        m->importedFrom = m;

        if (strcmp(m->srcfile->name->str, global.main_d) == 0)
        {
            static const char buf[] = "void main(){}";
            m->srcfile->setbuffer(const_cast<char *>(buf), sizeof(buf));
            m->srcfile->ref = 1;
        }
        else
        {
            m->read(Loc());
        }

        m->parse(global.params.doDocComments);
        m->buildTargetFiles(singleObj, createSharedLib || createStaticLib);
        m->deleteObjFile();
        if (m->isDocFile)
        {
            gendocfile(m);

            // Remove m from list of modules
            modules.remove(i);
            i--;
        }
    }
    if (global.errors)
        fatal();

    if (global.params.doHdrGeneration)
    {
        /* Generate 'header' import files.
         * Since 'header' import files must be independent of command
         * line switches and what else is imported, they are generated
         * before any semantic analysis.
         */
        for (unsigned i = 0; i < modules.dim; i++)
        {
            if (global.params.verbose)
                fprintf(global.stdmsg, "import    %s\n", modules[i]->toChars());
            genhdrfile(modules[i]);
        }
    }
    if (global.errors)
        fatal();

    // load all unconditional imports for better symbol resolving
    for (unsigned i = 0; i < modules.dim; i++)
    {
       if (global.params.verbose)
           fprintf(global.stdmsg, "importall %s\n", modules[i]->toChars());
       modules[i]->importAll(0);
    }
    if (global.errors)
       fatal();

    // Do semantic analysis
    for (unsigned i = 0; i < modules.dim; i++)
    {
        if (global.params.verbose)
            fprintf(global.stdmsg, "semantic  %s\n", modules[i]->toChars());
        modules[i]->semantic();
    }
    if (global.errors)
        fatal();

    Module::dprogress = 1;
    Module::runDeferredSemantic();

    // Do pass 2 semantic analysis
    for (unsigned i = 0; i < modules.dim; i++)
    {
        if (global.params.verbose)
            fprintf(global.stdmsg, "semantic2 %s\n", modules[i]->toChars());
        modules[i]->semantic2();
    }
    if (global.errors)
        fatal();

    // Do pass 3 semantic analysis
    for (unsigned i = 0; i < modules.dim; i++)
    {
        if (global.params.verbose)
            fprintf(global.stdmsg, "semantic3 %s\n", modules[i]->toChars());
        modules[i]->semantic3();
    }
    if (global.errors)
        fatal();

    Module::runDeferredSemantic3();

    if (global.errors || global.warnings)
        fatal();

    // Now that we analyzed all modules, write the module dependency file if
    // the user requested it.
    if (global.params.moduleDepsFile != NULL)
    {
        assert (global.params.moduleDepsFile != NULL);

        File deps(global.params.moduleDepsFile);
        OutBuffer* ob = global.params.moduleDeps;
        deps.setbuffer(static_cast<void*>(ob->data), ob->offset);
        deps.write();
    }

    // collects llvm modules to be linked if singleobj is passed
    std::vector<llvm::Module*> llvmModules;
    llvm::LLVMContext& context = llvm::getGlobalContext();

    // Generate output files
    for (unsigned i = 0; i < modules.dim; i++)
    {
        Module *m = modules[i];
        if (global.params.verbose)
            fprintf(global.stdmsg, "code      %s\n", m->toChars());
        if (global.params.obj)
        {
            llvm::Module* lm = m->genLLVMModule(context);

            if (global.errors)
                fatal();

            if (entrypoint && rootHasMain == m)
                emitEntryPointInto(lm);

            if (!singleObj)
            {
                m->deleteObjFile();
                writeModule(lm, m->objfile->name->str);
                global.params.objfiles->push(const_cast<char*>(m->objfile->name->str));
                delete lm;
            }
            else
            {
                llvmModules.push_back(lm);
            }
        }
        if (global.params.doDocComments)
            gendocfile(m);
    }

    // internal linking for singleobj
    if (singleObj && llvmModules.size() > 0)
    {
        Module *m = modules[0];

        const char* oname;
        const char* filename;
        if ((oname = global.params.exefile) || (oname = global.params.objname))
        {
            filename = FileName::forceExt(oname, global.obj_ext);
            if (global.params.objdir)
            {
                filename = FileName::combine(global.params.objdir, FileName::name(filename));
            }
        }
        else
            filename = m->objfile->name->str;

#if LDC_LLVM_VER < 306
#if 1
        // Temporary workaround for http://llvm.org/bugs/show_bug.cgi?id=11479.
        char* moduleName = const_cast<char*>(filename);
#else
        char* moduleName = m->toChars();
#endif
#endif

#if LDC_LLVM_VER >= 306
        llvm::Linker linker(llvmModules[0]);
#elif LDC_LLVM_VER >= 303
        llvm::Linker linker(new llvm::Module(moduleName, context));
#else
        llvm::Linker linker("ldc", moduleName, context);
#endif

        std::string errormsg;
#if LDC_LLVM_VER >= 306
        for (size_t i = 1; i < llvmModules.size(); i++)
#else
        for (size_t i = 0; i < llvmModules.size(); i++)
#endif
        {
#if LDC_LLVM_VER >= 306
            linker.linkInModule(llvmModules[i]);
#else
#if LDC_LLVM_VER >= 303
            if (linker.linkInModule(llvmModules[i], &errormsg))
#else
            if (linker.LinkInModule(llvmModules[i], &errormsg))
#endif
                error(Loc(), "%s", errormsg.c_str());
#endif
            delete llvmModules[i];
        }

        m->deleteObjFile();
        writeModule(linker.getModule(), filename);
        global.params.objfiles->push(const_cast<char*>(filename));

#if LDC_LLVM_VER >= 304
        linker.deleteModule();
#elif LDC_LLVM_VER == 303
        delete linker.getModule();
#endif
    }

    // output json file
    if (global.params.doJsonGeneration)
    {
        OutBuffer buf;
        json_generate(&buf, &modules);

        // Write buf to file
        const char *name = global.params.jsonfilename;

        if (name && name[0] == '-' && name[1] == 0)
        {   // Write to stdout; assume it succeeds
            (void)fwrite(buf.data, 1, buf.offset, stdout);
        }
        else
        {
            /* The filename generation code here should be harmonized with Module::setOutfile()
             */

            const char *jsonfilename;

            if (name && *name)
            {
                jsonfilename = FileName::defaultExt(name, global.json_ext);
            }
            else
            {
                // Generate json file name from first obj name
                const char *n = (*global.params.objfiles)[0];
                n = FileName::name(n);

                //if (!FileName::absolute(name))
                    //name = FileName::combine(dir, name);

                jsonfilename = FileName::forceExt(n, global.json_ext);
            }

            ensurePathToNameExists(Loc(), jsonfilename);

            File *jsonfile = new File(jsonfilename);

            jsonfile->setbuffer(buf.data, buf.offset);
            jsonfile->ref = 1;
            writeFile(Loc(), jsonfile);
        }
    }


    LLVM_D_FreeRuntime();
    llvm::llvm_shutdown();

    if (global.errors)
        fatal();

    int status = EXIT_SUCCESS;
    if (!global.params.objfiles->dim)
    {
        if (global.params.link)
            error(Loc(), "no object files to link");
        else if (createStaticLib)
            error(Loc(), "no object files");
    }
    else
    {
        if (global.params.link)
            status = linkObjToBinary(createSharedLib);
        else if (createStaticLib)
            createStaticLibrary();

        if (global.params.run && status == EXIT_SUCCESS)
        {
            status = runExecutable();

            /* Delete .obj files and .exe file
                */
            for (unsigned i = 0; i < modules.dim; i++)
            {
                modules[i]->deleteObjFile();
            }
            deleteExecutable();
        }
    }

    return status;
}
