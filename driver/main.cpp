//===-- main.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "id.h"
#include "json.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "rmem.h"
#include "root.h"
#include "driver/cl_options.h"
#include "driver/configfile.h"
#include "driver/linker.h"
#include "driver/target.h"
#include "driver/toobj.h"
#include "gen/cl_helpers.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "gen/metadata.h"
#include "gen/optimizer.h"
#include "gen/passes/Passes.h"
#include "llvm/Linker.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
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
#if POSIX
#include <errno.h>
#elif _WIN32
#include <windows.h>
#endif

// Needs Type already declared.
#include "cond.h"

using namespace opts;

extern void getenv_setargv(const char *envvar, int *pargc, char** *pargv);
extern void backend_init();
extern void backend_term();

static cl::opt<bool> noDefaultLib("nodefaultlib",
    cl::desc("Don't add a default library for linking implicitly"),
    cl::ZeroOrMore);

static StringsAdapter impPathsStore("I", global.params.imppath);
static cl::list<std::string, StringsAdapter> importPaths("I",
    cl::desc("Where to look for imports"),
    cl::value_desc("path"),
    cl::location(impPathsStore),
    cl::Prefix);

static StringsAdapter defaultLibStore("defaultlib", global.params.defaultlibnames);
static cl::list<std::string, StringsAdapter> defaultlibs("defaultlib",
    cl::desc("Set default libraries for non-debug build"),
    cl::value_desc("lib,..."),
    cl::location(defaultLibStore),
    cl::CommaSeparated);

static StringsAdapter debugLibStore("debuglib", global.params.debuglibnames);
static cl::list<std::string, StringsAdapter> debuglibs("debuglib",
    cl::desc("Set default libraries for debug build"),
    cl::value_desc("lib,..."),
    cl::location(debugLibStore),
    cl::CommaSeparated);

void printVersion() {
    printf("LDC - the LLVM D compiler (%s):\n", global.ldc_version);
    printf("  based on DMD %s and %s\n", global.version, global.llvm_version);
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
                error("Invalid %s level: %s", type, I->c_str());
            } else {
                setLevel((unsigned)level);
            }
        } else {
            char* cstr = mem.strdup(value);
            if (Lexer::isValidIdentifier(cstr)) {
                addIdent(cstr);
                continue;
            } else {
                error("Invalid %s identifier or level: '%s'", type, I->c_str());
            }
        }
    }
}

// Helper function to handle -of, -od, etc.
static void initFromString(char*& dest, const cl::opt<std::string>& src) {
    dest = 0;
    if (src.getNumOccurrences() != 0) {
        if (src.empty())
            error("Expected argument to '-%s'", src.ArgStr);
        dest = mem.strdup(src.c_str());
    }
}

#if _WIN32 && __DMC__
extern "C"
{
    extern int _xi_a;
    extern int _end;
}
#endif

int main(int argc, char** argv)
{
    mem.init();                         // initialize storage allocator
    mem.setStackBottom(&argv);
#if _WIN32 && __DMC__
    mem.addroots((char *)&_xi_a, (char *)&_end);
#endif

    // stack trace on signals
    llvm::sys::PrintStackTraceOnErrorSignal();

    Strings files;
    const char *p, *ext;
    Module *m;
    int status = EXIT_SUCCESS;

    // Set some default values
#if _WIN32
    char buf[MAX_PATH];
    GetModuleFileName(NULL, buf, MAX_PATH);
    global.params.argv0 = buf;
#else
    global.params.argv0 = argv[0];
#endif
    global.params.useSwitchError = 1;

    global.params.linkswitches = new Strings();
    global.params.libfiles = new Strings();
    global.params.objfiles = new Strings();
    global.params.ddocfiles = new Strings();

    global.params.moduleDeps = NULL;
    global.params.moduleDepsFile = NULL;

    // Set predefined version identifiers
    VersionCondition::addPredefinedGlobalIdent("LLVM"); // For backwards compatibility.
    VersionCondition::addPredefinedGlobalIdent("LDC");
    VersionCondition::addPredefinedGlobalIdent("all");
    VersionCondition::addPredefinedGlobalIdent("D_Version2");

    // build complete fixed up list of command line arguments
    std::vector<const char*> final_args;
    final_args.reserve(argc);

    // insert command line args until -run is reached
    int run_argnum = 1;
    while (run_argnum < argc && strncmp(argv[run_argnum], "-run", 4) != 0)
        ++run_argnum;
    final_args.insert(final_args.end(), &argv[0], &argv[run_argnum]);

    // read the configuration file
    ConfigFile cfg_file;

    // just ignore errors for now, they are still printed
#define CFG_FILENAME "ldc2.conf"
    cfg_file.read(global.params.argv0, (void*)main, CFG_FILENAME);
#undef CFG_FILENAME

    // insert config file additions to the argument list
    final_args.insert(final_args.end(), cfg_file.switches_begin(), cfg_file.switches_end());

    // insert -run and everything beyond
    final_args.insert(final_args.end(), &argv[run_argnum], &argv[argc]);

#if 0
    for (size_t i = 0; i < final_args.size(); ++i)
    {
        printf("final_args[%zu] = %s\n", i, final_args[i]);
    }
#endif

    // Initialize LLVM.
    // Initialize targets first, so that --version shows registered targets.
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    // Handle fixed-up arguments!
    cl::SetVersionPrinter(&printVersion);
    cl::ParseCommandLineOptions(final_args.size(), const_cast<char**>(&final_args[0]),
        "LDC - the LLVM D compiler\n"
#if LDC_LLVM_VER < 302
        , true
#endif
    );

    // Print config file path if -v was passed
    if (global.params.verbose) {
        const std::string& path = cfg_file.path();
        if (!path.empty())
            printf("config    %s\n", path.c_str());
    }

    bool skipModules = mCPU == "help" ||(!mAttrs.empty() && mAttrs.front() == "help");

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

    initFromString(global.params.xfilename, jsonFile);
    if (global.params.xfilename)
        global.params.doXGeneration = true;

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
                error("-run must be followed by a source file, not '%s'", name);
            }

            files.push(mem.strdup(name));
            runargs.erase(runargs.begin());
        } else {
            global.params.run = false;
            error("Expected at least one argument to '-run'\n");
        }
    }


    files.reserve(fileList.size());
    typedef std::vector<std::string>::iterator It;
    for(It I = fileList.begin(), E = fileList.end(); I != E; ++I)
        if (!I->empty())
            files.push(mem.strdup(I->c_str()));

    if (global.errors)
    {
        fatal();
    }
    if (files.dim == 0 && !skipModules)
    {
        cl::PrintHelpMessage();
        return EXIT_FAILURE;
    }

    Array* libs;
    if (global.params.symdebug)
    {
        libs = global.params.debuglibnames;
    }
    else
        libs = global.params.defaultlibnames;

    if (!noDefaultLib)
    {
        if (libs)
        {
            for (unsigned i = 0; i < libs->dim; i++)
            {
                char* lib = static_cast<char *>(libs->data[i]);
                char *arg = static_cast<char *>(mem.malloc(strlen(lib) + 3));
                strcpy(arg, "-l");
                strcpy(arg+2, lib);
                global.params.linkswitches->push(arg);
            }
        }
        else
        {
            global.params.linkswitches->push(mem.strdup("-ldruntime-ldc"));
        }
    }

    if (global.params.useUnitTests)
        global.params.useAssert = 1;

    // LDC output determination

    // if we don't link, autodetect target from extension
    if(!global.params.link && !createStaticLib && global.params.objname) {
        ext = FileName::ext(global.params.objname);
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
        error("-lib and -shared switches cannot be used together");

    if (createSharedLib && mRelocModel == llvm::Reloc::Default)
        mRelocModel = llvm::Reloc::PIC_;

    if (global.params.link && !createSharedLib)
    {
        global.params.exefile = global.params.objname;
        if (files.dim > 1)
            global.params.objname = NULL;
    }
    else if (global.params.run)
    {
        error("flags conflict with -run");
        fatal();
    }
    else if (global.params.objname && files.dim > 1) {
        if (createStaticLib || createSharedLib)
        {
            singleObj = true;
        }
        if (!singleObj)
        {
            error("multiple source files, but only one .obj name");
            fatal();
        }
    }

    if (soname.getNumOccurrences() > 0 && !createSharedLib) {
        error("-soname can be used only when building a shared library");
        fatal();
    }

    // create a proper target
    Ir ir;

    // Set up the TargetMachine.
    ExplicitBitness::Type bitness = ExplicitBitness::None;
    if ((m32bits || m64bits) && (!mArch.empty() || !mTargetTriple.empty()))
        error("-m32 and -m64 switches cannot be used together with -march and -mtriple switches");

    if (m32bits)
        bitness = ExplicitBitness::M32;
    if (m64bits)
    {
        if (bitness != ExplicitBitness::None)
        {
            error("cannot use both -m32 and -m64 options");
        }
    }

    if (global.errors)
        fatal();

    gTargetMachine = createTargetMachine(mTargetTriple, mArch, mCPU, mAttrs, bitness,
        mRelocModel, mCodeModel, codeGenOptLevel(), global.params.symdebug);
    global.params.targetTriple = llvm::Triple(gTargetMachine->getTargetTriple());

#if LDC_LLVM_VER >= 302
    gDataLayout = gTargetMachine->getDataLayout();
#else
    gDataLayout = gTargetMachine->getTargetData();
#endif

    // Starting with LLVM 3.1 we could also use global.params.targetTriple.isArch64Bit();
    global.params.is64bit = gDataLayout->getPointerSizeInBits(ADDRESS_SPACE) == 64;

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
            // FIXME: Detect soft float (PPC_SoftFP/PPC_HardFP).
            VersionCondition::addPredefinedGlobalIdent("PPC");
            break;
        case llvm::Triple::ppc64:
            VersionCondition::addPredefinedGlobalIdent("PPC64");
            VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
            break;
        case llvm::Triple::arm:
            // FIXME: Detect various FP ABIs (ARM_Soft, ARM_SoftFP, ARM_HardFP).
            VersionCondition::addPredefinedGlobalIdent("ARM");
            break;
        case llvm::Triple::thumb:
            VersionCondition::addPredefinedGlobalIdent("ARM");
            VersionCondition::addPredefinedGlobalIdent("Thumb"); // For backwards compatibility.
            VersionCondition::addPredefinedGlobalIdent("ARM_Thumb");
            VersionCondition::addPredefinedGlobalIdent("ARM_Soft");
            VersionCondition::addPredefinedGlobalIdent("D_SoftFloat");
            break;
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            // FIXME: Detect O32/N32 variants (MIPS_{O32,N32}[_SoftFP,_HardFP]).
            VersionCondition::addPredefinedGlobalIdent("MIPS");
            break;
#if LDC_LLVM_VER >= 301
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            // FIXME: Detect N64 variants (MIPS64_N64[_SoftFP,_HardFP]).
            VersionCondition::addPredefinedGlobalIdent("MIPS64");
            break;
#endif
        case llvm::Triple::sparc:
            // FIXME: Detect SPARC v8+ (SPARC_V8Plus).
            // FIXME: Detect soft float (SPARC_SoftFP/SPARC_HardFP).
            VersionCondition::addPredefinedGlobalIdent("SPARC");
            break;
        case llvm::Triple::sparcv9:
            VersionCondition::addPredefinedGlobalIdent("SPARC64");
            VersionCondition::addPredefinedGlobalIdent("D_HardFloat");
            break;
        default:
            error("invalid cpu architecture specified: %s", global.params.targetTriple.getArchName().str().c_str());
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
    if (global.params.is64bit) {
        VersionCondition::addPredefinedGlobalIdent("LLVM64"); // For backwards compatibility.
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
            break;
        case llvm::Triple::MinGW32:
            VersionCondition::addPredefinedGlobalIdent("Windows");
            VersionCondition::addPredefinedGlobalIdent(global.params.is64bit ? "Win64" : "Win32");
            VersionCondition::addPredefinedGlobalIdent("mingw32"); // For backwards compatibility.
            VersionCondition::addPredefinedGlobalIdent("MinGW");
            break;
        case llvm::Triple::Cygwin:
            error("Cygwin is not yet supported");
            fatal();
            VersionCondition::addPredefinedGlobalIdent("Cygwin");
            break;
        case llvm::Triple::Linux:
            VersionCondition::addPredefinedGlobalIdent("linux");
            VersionCondition::addPredefinedGlobalIdent("Posix");
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
            VersionCondition::addPredefinedGlobalIdent("freebsd"); // For backwards compatibility.
            VersionCondition::addPredefinedGlobalIdent("FreeBSD");
            VersionCondition::addPredefinedGlobalIdent("Posix");
            break;
        case llvm::Triple::Solaris:
            VersionCondition::addPredefinedGlobalIdent("solaris"); // For backwards compatibility.
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
            error("target '%s' is not yet supported", global.params.targetTriple.str().c_str());
            fatal();
    }

    // Expose LLVM version to runtime
#define STR(x) #x
#define XSTR(x) STR(x)
    VersionCondition::addPredefinedGlobalIdent("LDC_LLVM_"XSTR(LDC_LLVM_VER));
#undef XSTR
#undef STR

    if (global.params.targetTriple.isOSWindows()) {
        global.dll_ext = "dll";
        global.lib_ext = "lib";
    } else {
        global.dll_ext = "so";
        global.lib_ext = "a";
    }

    // added in 1.039
    if (global.params.doDocComments)
        VersionCondition::addPredefinedGlobalIdent("D_Ddoc");

    // unittests?
    if (global.params.useUnitTests)
        VersionCondition::addPredefinedGlobalIdent("unittest");

    if (global.params.useAssert)
        VersionCondition::addPredefinedGlobalIdent("assert");

    if (!global.params.useArrayBounds)
        VersionCondition::addPredefinedGlobalIdent("D_NoBoundsChecks");

    // Initialization
    Type::init(&ir);
    Id::initialize();
    Module::init();
    initPrecedence();

    backend_init();

    //printf("%d source files\n",files.dim);

    // Build import search path
    if (global.params.imppath)
    {
        for (unsigned i = 0; i < global.params.imppath->dim; i++)
        {
            char *path = static_cast<char *>(global.params.imppath->data[i]);
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
            char *path = static_cast<char *>(global.params.fileImppath->data[i]);
            Strings *a = FileName::splitPath(path);

            if (a)
            {
                if (!global.filePath)
                    global.filePath = new Strings();
                global.filePath->append(a);
            }
        }
    }

    // Create Modules
    Modules modules;
    modules.reserve(files.dim);
    for (unsigned i = 0; i < files.dim; i++)
    {   Identifier *id;
        const char *ext;
        const char *name;

        p = static_cast<char *>(files.data[i]);

        p = FileName::name(p);      // strip path
        ext = FileName::ext(p);
        if (ext)
        {
#if POSIX
            if (strcmp(ext, global.obj_ext) == 0 ||
                strcmp(ext, global.bc_ext) == 0)
#else
            if (Port::stricmp(ext, global.obj_ext) == 0 ||
                Port::stricmp(ext, global.obj_ext_alt) == 0 ||
                Port::stricmp(ext, global.bc_ext) == 0)
#endif
            {
                global.params.objfiles->push(static_cast<char *>(files.data[i]));
                continue;
            }

#if POSIX
            if (strcmp(ext, "a") == 0)
#elif __MINGW32__
            if (Port::stricmp(ext, "a") == 0)
#else
            if (Port::stricmp(ext, "lib") == 0)
#endif
            {
                global.params.libfiles->push(static_cast<char *>(files.data[i]));
                continue;
            }

            if (strcmp(ext, global.ddoc_ext) == 0)
            {
                global.params.ddocfiles->push(static_cast<char *>(files.data[i]));
                continue;
            }

            if (FileName::equals(ext, global.json_ext))
            {
                global.params.doXGeneration = 1;
                global.params.xfilename = static_cast<char *>(files.data[i]);
                continue;
            }

#if !POSIX
            if (Port::stricmp(ext, "res") == 0)
            {
                global.params.resfile = static_cast<char *>(files.data[i]);
                continue;
            }

            if (Port::stricmp(ext, "def") == 0)
            {
                global.params.deffile = static_cast<char *>(files.data[i]);
                continue;
            }

            if (Port::stricmp(ext, "exe") == 0)
            {
                global.params.exefile = static_cast<char *>(files.data[i]);
                continue;
            }
#endif

            if (Port::stricmp(ext, global.mars_ext) == 0 ||
                Port::stricmp(ext, global.hdr_ext) == 0)
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
            {   error("unrecognized file extension %s\n", ext);
                fatal();
            }
        }
        else
        {   name = p;
            if (!*p)
            {
        Linvalid:
                error("invalid file name '%s'", static_cast<char *>(files.data[i]));
                fatal();
            }
            name = p;
        }

        id = Lexer::idPool(name);
        m = new Module(static_cast<char *>(files.data[i]), id, global.params.doDocComments, global.params.doHdrGeneration);
        m->isRoot = true;
        modules.push(m);
    }

    // Read files, parse them
    for (unsigned i = 0; i < modules.dim; i++)
    {
        m = static_cast<Module *>(modules.data[i]);
        if (global.params.verbose)
            printf("parse     %s\n", m->toChars());
        if (!Module::rootModule)
            Module::rootModule = m;
        m->importedFrom = m;
        m->read(0);
        m->parse(global.params.doDocComments);
        m->buildTargetFiles(singleObj);
        m->deleteObjFile();
        if (m->isDocFile)
        {
            m->gendocfile();

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
            m = static_cast<Module *>(modules.data[i]);
            if (global.params.verbose)
                printf("import    %s\n", m->toChars());
            m->genhdrfile();
        }
    }
    if (global.errors)
        fatal();

    // load all unconditional imports for better symbol resolving
    for (unsigned i = 0; i < modules.dim; i++)
    {
       m = static_cast<Module *>(modules.data[i]);
       if (global.params.verbose)
           printf("importall %s\n", m->toChars());
       m->importAll(0);
    }
    if (global.errors)
       fatal();

    // Do semantic analysis
    for (unsigned i = 0; i < modules.dim; i++)
    {
        m = static_cast<Module *>(modules.data[i]);
        if (global.params.verbose)
            printf("semantic  %s\n", m->toChars());
        m->semantic();
    }
    if (global.errors)
        fatal();

    Module::dprogress = 1;
    Module::runDeferredSemantic();

    // Do pass 2 semantic analysis
    for (unsigned i = 0; i < modules.dim; i++)
    {
        m = static_cast<Module *>(modules.data[i]);
        if (global.params.verbose)
            printf("semantic2 %s\n", m->toChars());
        m->semantic2();
    }
    if (global.errors)
        fatal();

    // Do pass 3 semantic analysis
    for (unsigned i = 0; i < modules.dim; i++)
    {
        m = static_cast<Module *>(modules.data[i]);
        if (global.params.verbose)
            printf("semantic3 %s\n", m->toChars());
        m->semantic3();
    }
    if (global.errors)
        fatal();

    // This doesn't play nice with debug info at the moment.
    //
    // Also, don't run the additional semantic3 passes when building unit tests.
    // This is basically a huge hack around the fact that linking against a
    // library is supposed to require the same compiler flags as when it was
    // built, but -unittest is usually not thought to behave like this from a
    // user perspective.
    //
    // Thus, if a library contained some functions in version(unittest), for
    // example the tests in std.concurrency, and we ended up inline-scannin
    // these functions while doing an -unittest build of a client application,
    // we could end up referencing functions that we think are
    // availableExternally, but have never been touched when the library was built.
    //
    // Alternatively, we could also amend the availableExternally detection
    // logic (e.g. just codegen everything on -unittest builds), but the extra
    // inlining is unlikely to be important for test builds anyway.
    if (!global.params.symdebug && willInline() && !global.params.useUnitTests)
    {
        global.inExtraInliningSemantic = true;
        Logger::println("Running some extra semantic3's for inlining purposes");
        {
            // Do pass 3 semantic analysis on all imported modules,
            // since otherwise functions in them cannot be inlined
            for (unsigned i = 0; i < Module::amodules.dim; i++)
            {
                m = static_cast<Module *>(Module::amodules.data[i]);
                if (global.params.verbose)
                    printf("semantic3 %s\n", m->toChars());
                m->semantic2();
                m->semantic3();
            }
            if (global.errors)
                fatal();
        }
    }
    if (global.errors || global.warnings)
        fatal();

    // write module dependencies to file if requested
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
        m = static_cast<Module *>(modules.data[i]);
        if (global.params.verbose)
            printf("code      %s\n", m->toChars());
        if (global.params.obj)
        {
            llvm::Module* lm = m->genLLVMModule(context, &ir);
            if (!singleObj)
            {
                m->deleteObjFile();
                writeModule(lm, m->objfile->name->str);
                global.params.objfiles->push(const_cast<char*>(m->objfile->name->str));
                delete lm;
            }
            else
                llvmModules.push_back(lm);
        }
        if (global.errors)
            m->deleteObjFile();
        else
        {
            if (global.params.doDocComments)
            m->gendocfile();
        }
    }

    // internal linking for singleobj
    if (singleObj && llvmModules.size() > 0)
    {
        Module* m = static_cast<Module*>(modules.data[0]);

        char* oname;
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

#if 1
        // Temporary workaround for http://llvm.org/bugs/show_bug.cgi?id=11479.
        char* moduleName = const_cast<char*>(filename);
#else
        char* moduleName = m->toChars();
#endif

#if LDC_LLVM_VER >= 303
        llvm::Module *dest = new llvm::Module(moduleName, context);
        llvm::Linker linker(dest);
#else
        llvm::Linker linker("ldc", moduleName, context);
#endif

        std::string errormsg;
        for (size_t i = 0; i < llvmModules.size(); i++)
        {
#if LDC_LLVM_VER >= 303
            if (linker.linkInModule(llvmModules[i], llvm::Linker::DestroySource, &errormsg))
#else
            if (linker.LinkInModule(llvmModules[i], &errormsg))
#endif
                error("%s", errormsg.c_str());
            delete llvmModules[i];
        }

        m->deleteObjFile();
        writeModule(linker.getModule(), filename);
        global.params.objfiles->push(const_cast<char*>(filename));

#if LDC_LLVM_VER >= 303
        delete dest;
#endif
    }

    // output json file
    if (global.params.doXGeneration)
    {
        OutBuffer buf;
        json_generate(&buf, &modules);

        // Write buf to file
        const char *name = global.params.xfilename;

        if (name && name[0] == '-' && name[1] == 0)
        {   // Write to stdout; assume it succeeds
            int n = fwrite(buf.data, 1, buf.offset, stdout);
            assert(n == buf.offset);        // keep gcc happy about return values
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

            FileName::ensurePathToNameExists(jsonfilename);

            File *jsonfile = new File(jsonfilename);

            jsonfile->setbuffer(buf.data, buf.offset);
            jsonfile->ref = 1;
            jsonfile->writev();
        }
    }

    backend_term();
    if (global.errors)
        fatal();

    if (!global.params.objfiles->dim)
    {
        if (global.params.link)
            error("no object files to link");
        else if (createStaticLib)
            error("no object files");
    }
    else
    {
        if (global.params.link)
            status = linkObjToBinary(createSharedLib);
        else if (createStaticLib)
            createStaticLibrary();

        if (global.params.run)
        {
            if (!status)
            {
                status = runExecutable();

                /* Delete .obj files and .exe file
                 */
                for (unsigned i = 0; i < modules.dim; i++)
                {
                    m = static_cast<Module *>(modules.data[i]);
                    m->deleteObjFile();
                }
                deleteExecutable();
            }
        }
    }

    return status;
}
