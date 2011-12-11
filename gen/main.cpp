// Pulled out of dmd/mars.c

// some things are taken from llvm's llc tool
// which uses the llvm license

#include "gen/llvm.h"
#include "llvm/LinkAllVMCore.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/MC/SubtargetFeature.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#if POSIX
#include <errno.h>
#elif _WIN32
#include <windows.h>
#endif

#include "rmem.h"
#include "root.h"

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "id.h"
#include "cond.h"
#include "json.h"

#include "gen/logger.h"
#include "gen/linkage.h"
#include "gen/linker.h"
#include "gen/irstate.h"
#include "gen/optimizer.h"
#include "gen/toobj.h"
#include "gen/metadata.h"
#include "gen/passes/Passes.h"

#include "gen/cl_options.h"
#include "gen/cl_helpers.h"
using namespace opts;

#include "gen/configfile.h"

#if DMDV1
typedef Array Modules;
#endif

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
    printf("LLVM D Compiler %s\nbased on DMD %s and %s\n%s\n%s\n",
    global.ldc_version, global.version, global.llvm_version, global.copyright, global.written);
    printf("D Language Documentation: http://d-programming-language.org/index.html\n"
           "LDC Homepage: https://github.com/ldc-developers/ldc\n");
}

// Helper function to handle -d-debug=* and -d-version=*
static void processVersions(std::vector<std::string>& list, char* type,
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

#if _WIN32
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
#if _WIN32
    mem.addroots((char *)&_xi_a, (char *)&_end);
#endif

    // stack trace on signals
    llvm::sys::PrintStackTraceOnErrorSignal();

    Strings files;
    char *p, *ext;
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
    VersionCondition::addPredefinedGlobalIdent("LLVM");
    VersionCondition::addPredefinedGlobalIdent("LDC");
    VersionCondition::addPredefinedGlobalIdent("all");
#if DMDV2
    VersionCondition::addPredefinedGlobalIdent("D_Version2");
#endif

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
#if DMDV2
#define CFG_FILENAME "ldc2.conf"
#else
#define CFG_FILENAME "ldc.conf"
#endif
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

    // Handle fixed-up arguments!
    cl::SetVersionPrinter(&printVersion);
    cl::ParseCommandLineOptions(final_args.size(), (char**)&final_args[0], "LLVM-based D Compiler\n", true);

    // Print config file path if -v was passed
    if (global.params.verbose) {
        const std::string& path = cfg_file.path();
        if (!path.empty())
            printf("config    %s\n", path.c_str());
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

    initFromString(global.params.xfilename, jsonFile);
    if (global.params.xfilename)
        global.params.doXGeneration = true;

#ifdef _DH
    initFromString(global.params.hdrdir, hdrDir);
    initFromString(global.params.hdrname, hdrFile);
    global.params.doHdrGeneration |=
        global.params.hdrdir || global.params.hdrname;
#endif

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
            files.push(mem.strdup(runargs[0].c_str()));
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
    if (files.dim == 0)
    {
        cl::PrintHelpMessage();
        return EXIT_FAILURE;
    }

    llvm::TargetOptions targetOptions;
    // FIXME: Options here are { None, Less, Default, Aggressive } as defined http://llvm.org/docs/doxygen/html/namespacellvm_1_1CodeGenOpt.html
    llvm::CodeGenOpt::Level codeGenOptLevel = llvm::CodeGenOpt::None; // I am setting this to None for the moment as I dont know how this changes generation

    Array* libs;
    if (global.params.symdebug)
    {
        libs = global.params.debuglibnames;
        targetOptions.NoFramePointerElim = true;
    }
    else
        libs = global.params.defaultlibnames;

    if (!noDefaultLib)
    {
        if (libs)
        {
            for (unsigned i = 0; i < libs->dim; i++)
            {
                char* lib = (char *)libs->data[i];
                char *arg = (char *)mem.malloc(strlen(lib) + 3);
                strcpy(arg, "-l");
                strcpy(arg+2, lib);
                global.params.linkswitches->push(arg);
            }
        }
        else
        {
#if DMDV2
            global.params.linkswitches->push(mem.strdup("-ldruntime-ldc"));
#else
            global.params.linkswitches->push(mem.strdup("-lldc-runtime"));
            global.params.linkswitches->push(mem.strdup("-ltango-cc-tango"));
            global.params.linkswitches->push(mem.strdup("-ltango-gc-basic"));
            // pass the runtime again to resolve issues
            // with linking order
            global.params.linkswitches->push(mem.strdup("-lldc-runtime"));
#endif
        }
    }

    if (global.params.run)
        quiet = 1;

    if (global.params.useUnitTests)
        global.params.useAssert = 1;

    // LDC output determination

    // if we don't link, autodetect target from extension
    if(!global.params.link && global.params.objname) {
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
        } else if (strcmp(ext, global.obj_ext) == 0) {
            global.params.output_o = OUTPUTFLAGset;
            autofound = true;
        } else {
            // append dot, so forceExt won't change existing name even if it contains dots
            size_t len = strlen(global.params.objname);
            size_t extlen = strlen(".");
            char* s = (char *)mem.malloc(len + 1 + extlen + 1);
            memcpy(s, global.params.objname, len);
            s[len] = '.';
            s[len+1+extlen] = 0;
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
    else
    {
        if (global.params.objname && files.dim > 1 && !singleObj)
        {
            error("multiple source files, but only one .obj name");
            fatal();
        }
    }

    // create a proper target
    Ir ir;

    // check -m32/64 sanity
    if (m32bits && m64bits)
        error("cannot use both -m32 and -m64 options");
    else if ((m32bits || m64bits) && (!mArch.empty() || !mTargetTriple.empty()))
        error("-m32 and -m64 switches cannot be used together with -march and -mtriple switches");
    if (global.errors)
        fatal();

    // override triple if needed
    const char* defaultTriple = DEFAULT_TARGET_TRIPLE;
    if ((sizeof(void*) == 4 && m64bits) || (sizeof(void*) == 8 && m32bits))
    {
        defaultTriple = DEFAULT_ALT_TARGET_TRIPLE;
    }

    // did the user override the target triple?
    if (mTargetTriple.empty())
    {
        if (!mArch.empty())
        {
            error("you must specify a target triple as well with -mtriple when using the -march option");
            fatal();
        }
        global.params.targetTriple = defaultTriple;
    }
    else
    {
        global.params.targetTriple = mTargetTriple.c_str();
    }

    std::string triple = global.params.targetTriple;

    // Allocate target machine.

    // first initialize llvm
#if 0
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();
#endif
#define LLVM_TARGET(A) \
    LLVMInitialize##A##TargetInfo(); \
    LLVMInitialize##A##Target(); \
    LLVMInitialize##A##AsmPrinter(); \
    LLVMInitialize##A##AsmParser(); \
    LLVMInitialize##A##TargetMC();
LDC_TARGETS
#undef LLVM_TARGET

    const llvm::Target *theTarget = NULL;
    // Check whether the user has explicitly specified an architecture to compile for.
    if (mArch.empty())
    {
        std::string Err;
        theTarget = llvm::TargetRegistry::lookupTarget(triple, Err);
        if (theTarget == 0)
        {
            error("failed to auto-select target: %s, please use the -march option", Err.c_str());
            fatal();
        }
    }
    else
    {
        for (llvm::TargetRegistry::iterator it = llvm::TargetRegistry::begin(),
             ie = llvm::TargetRegistry::end(); it != ie; ++it)
        {
            if (mArch == it->getName())
            {
                theTarget = &*it;
                break;
            }
        }

        if (!theTarget)
        {
            error("invalid target '%s'", mArch.c_str());
            fatal();
        }
    }

    // Package up features to be passed to target/subtarget
    std::string FeaturesStr;
    if (mCPU.size() || mAttrs.size())
    {
        llvm::SubtargetFeatures Features;
        for (unsigned i = 0; i != mAttrs.size(); ++i)
            Features.AddFeature(mAttrs[i]);
        FeaturesStr = Features.getString();
    }

    // FIXME
    //std::auto_ptr<llvm::TargetMachine> target(theTarget->createTargetMachine(triple, FeaturesStr));
    //assert(target.get() && "Could not allocate target machine!");
    //gTargetMachine = target.get();
    
    llvm::TargetMachine * target = theTarget->createTargetMachine(
        llvm::StringRef(triple),
        llvm::StringRef(mCPU),
        llvm::StringRef(FeaturesStr),
        targetOptions,
        mRelocModel,
        mCodeModel,
        codeGenOptLevel
    );

    gTargetMachine = target;

    gTargetData = target->getTargetData();

    // get final data layout
    std::string datalayout = gTargetData->getStringRepresentation();
    global.params.dataLayout = datalayout.c_str();

    global.params.llvmArch = theTarget->getName();

    if (strcmp(global.params.llvmArch,"x86")==0) {
        VersionCondition::addPredefinedGlobalIdent("X86");
        global.params.isLE = true;
        global.params.is64bit = false;
        global.params.cpu = ARCHx86;
        if (global.params.useInlineAsm) {
            VersionCondition::addPredefinedGlobalIdent("D_InlineAsm_X86");
        }
    }
    else if (strcmp(global.params.llvmArch,"x86-64")==0) {
        VersionCondition::addPredefinedGlobalIdent("X86_64");
        global.params.isLE = true;
        global.params.is64bit = true;
        global.params.cpu = ARCHx86_64;
        if (global.params.useInlineAsm) {
            VersionCondition::addPredefinedGlobalIdent("D_InlineAsm_X86_64");
        }
    }
    else if (strcmp(global.params.llvmArch,"ppc32")==0) {
        VersionCondition::addPredefinedGlobalIdent("PPC");
        global.params.isLE = false;
        global.params.is64bit = false;
        global.params.cpu = ARCHppc;
    }
    else if (strcmp(global.params.llvmArch,"ppc64")==0) {
        VersionCondition::addPredefinedGlobalIdent("PPC64");
        global.params.isLE = false;
        global.params.is64bit = true;
        global.params.cpu = ARCHppc_64;
    }
    else if (strcmp(global.params.llvmArch,"arm")==0) {
        VersionCondition::addPredefinedGlobalIdent("ARM");
        global.params.isLE = true;
        global.params.is64bit = false;
        global.params.cpu = ARCHarm;
    }
    else if (strcmp(global.params.llvmArch,"thumb")==0) {
        VersionCondition::addPredefinedGlobalIdent("Thumb");
        global.params.isLE = true;
        global.params.is64bit = false;
        global.params.cpu = ARCHthumb;
    }
    else {
        error("invalid cpu architecture specified: %s", global.params.llvmArch);
        fatal();
    }

    // endianness
    if (global.params.isLE) {
        VersionCondition::addPredefinedGlobalIdent("LittleEndian");
    }
    else {
        VersionCondition::addPredefinedGlobalIdent("BigEndian");
    }

    // a generic 64bit version
    if (global.params.is64bit) {
        VersionCondition::addPredefinedGlobalIdent("LLVM64");
        // FIXME: is this always correct?
        VersionCondition::addPredefinedGlobalIdent("D_LP64");
    }

    // parse the OS out of the target triple
    // see http://gcc.gnu.org/install/specific.html for details
    // also llvm's different SubTargets have useful information
    size_t npos = std::string::npos;

    // windows
    // FIXME: win64
    if (triple.find("windows") != npos || triple.find("win32") != npos || triple.find("mingw") != npos)
    {
        global.params.os = OSWindows;
        VersionCondition::addPredefinedGlobalIdent("Windows");
        VersionCondition::addPredefinedGlobalIdent("Win32");
        VersionCondition::addPredefinedGlobalIdent("mingw32");
        VersionCondition::addPredefinedGlobalIdent("MinGW");
    }
    // FIXME: cygwin
    else if (triple.find("cygwin") != npos)
    {
        error("CygWin is not yet supported");
        fatal();
    }
    // linux
    else if (triple.find("linux") != npos)
    {
        global.params.os = OSLinux;
        VersionCondition::addPredefinedGlobalIdent("linux");
        VersionCondition::addPredefinedGlobalIdent("Posix");
    }
    // haiku
    else if (triple.find("haiku") != npos)
    {
        global.params.os = OSHaiku;
        VersionCondition::addPredefinedGlobalIdent("Haiku");
        VersionCondition::addPredefinedGlobalIdent("Posix");
    }
    // darwin
    else if (triple.find("-darwin") != npos)
    {
        global.params.os = OSMacOSX;
        VersionCondition::addPredefinedGlobalIdent("OSX");
        VersionCondition::addPredefinedGlobalIdent("darwin");
        VersionCondition::addPredefinedGlobalIdent("Posix");
    }
    // freebsd
    else if (triple.find("-freebsd") != npos)
    {
        global.params.os = OSFreeBSD;
        VersionCondition::addPredefinedGlobalIdent("freebsd");
        VersionCondition::addPredefinedGlobalIdent("FreeBSD");
        VersionCondition::addPredefinedGlobalIdent("Posix");
    }
    // solaris
    else if (triple.find("-solaris") != npos)
    {
        global.params.os = OSSolaris;
        VersionCondition::addPredefinedGlobalIdent("solaris");
        VersionCondition::addPredefinedGlobalIdent("Solaris");
        VersionCondition::addPredefinedGlobalIdent("Posix");
    }
    // unsupported
    else
    {
        error("target '%s' is not yet supported", global.params.targetTriple);
        fatal();
    }

    if (global.params.os == OSWindows) {
        global.dll_ext = "dll";
        global.lib_ext = "lib";
    } else {
        global.dll_ext = "so";
        global.lib_ext = "a";
    }

    // added in 1.039
    if (global.params.doDocComments)
        VersionCondition::addPredefinedGlobalIdent("D_Ddoc");

#if DMDV2
    // unittests?
    if (global.params.useUnitTests)
        VersionCondition::addPredefinedGlobalIdent("unittest");
#endif

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
            char *path = (char *)global.params.imppath->data[i];
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
            char *path = (char *)global.params.fileImppath->data[i];
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
        char *ext;
        char *name;

        p = (char *) files.data[i];

        p = FileName::name(p);      // strip path
        ext = FileName::ext(p);
        if (ext)
        {
#if POSIX
            if (strcmp(ext, global.obj_ext) == 0 ||
            strcmp(ext, global.bc_ext) == 0)
#else
            if (stricmp(ext, global.obj_ext) == 0 ||
            stricmp(ext, global.bc_ext) == 0)
#endif
            {
                global.params.objfiles->push((char *)files.data[i]);
                continue;
            }

#if POSIX
            if (strcmp(ext, "a") == 0)
#elif __MINGW32__
            if (stricmp(ext, "a") == 0)
#else
            if (stricmp(ext, "lib") == 0)
#endif
            {
                global.params.libfiles->push((char *)files.data[i]);
                continue;
            }

            if (strcmp(ext, global.ddoc_ext) == 0)
            {
                global.params.ddocfiles->push((char *)files.data[i]);
                continue;
            }

            if (FileName::equals(ext, global.json_ext))
            {
                global.params.doXGeneration = 1;
                global.params.xfilename = (char *)files.data[i];
                continue;
            }

#if !POSIX
            if (stricmp(ext, "res") == 0)
            {
                global.params.resfile = (char *)files.data[i];
                continue;
            }

            if (stricmp(ext, "def") == 0)
            {
                global.params.deffile = (char *)files.data[i];
                continue;
            }

            if (stricmp(ext, "exe") == 0)
            {
                global.params.exefile = (char *)files.data[i];
                continue;
            }
#endif

            if (stricmp(ext, global.mars_ext) == 0 ||
            stricmp(ext, global.hdr_ext) == 0 ||
            stricmp(ext, "htm") == 0 ||
            stricmp(ext, "html") == 0 ||
            stricmp(ext, "xhtml") == 0)
            {
                ext--;          // skip onto '.'
                assert(*ext == '.');
                name = (char *)mem.malloc((ext - p) + 1);
                memcpy(name, p, ext - p);
                name[ext - p] = 0;      // strip extension

                if (name[0] == 0 ||
                    strcmp(name, "..") == 0 ||
                    strcmp(name, ".") == 0)
                {
            Linvalid:
                    error("invalid file name '%s'", (char *)files.data[i]);
                    fatal();
                }
            }
            else
            {   error("unrecognized file extension %s\n", ext);
                fatal();
            }
        }
        else
        {   name = p;
            if (!*name)
            goto Linvalid;
        }

        id = Lexer::idPool(name);
        m = new Module((char *) files.data[i], id, global.params.doDocComments, global.params.doHdrGeneration);
        m->isRoot = true;
        modules.push(m);
    }

    // Read files, parse them
    for (unsigned i = 0; i < modules.dim; i++)
    {
        m = (Module *)modules.data[i];
        if (global.params.verbose)
            printf("parse     %s\n", m->toChars());
        if (!Module::rootModule)
            Module::rootModule = m;
        m->importedFrom = m;
        m->read(0);
#ifdef _DH
        m->parse(global.params.doDocComments);
#else
        m->parse();
#endif
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
#ifdef _DH
    if (global.params.doHdrGeneration)
    {
        /* Generate 'header' import files.
         * Since 'header' import files must be independent of command
         * line switches and what else is imported, they are generated
         * before any semantic analysis.
         */
        for (unsigned i = 0; i < modules.dim; i++)
        {
            m = (Module *)modules.data[i];
            if (global.params.verbose)
                printf("import    %s\n", m->toChars());
            m->genhdrfile();
        }
    }
    if (global.errors)
        fatal();
#endif

    // load all unconditional imports for better symbol resolving
    for (unsigned i = 0; i < modules.dim; i++)
    {
       m = (Module *)modules.data[i];
       if (global.params.verbose)
           printf("importall %s\n", m->toChars());
       m->importAll(0);
    }
    if (global.errors)
       fatal();

    // Do semantic analysis
    for (unsigned i = 0; i < modules.dim; i++)
    {
        m = (Module *)modules.data[i];
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
        m = (Module *)modules.data[i];
        if (global.params.verbose)
            printf("semantic2 %s\n", m->toChars());
        m->semantic2();
    }
    if (global.errors)
        fatal();

    // Do pass 3 semantic analysis
    for (unsigned i = 0; i < modules.dim; i++)
    {
        m = (Module *)modules.data[i];
        if (global.params.verbose)
            printf("semantic3 %s\n", m->toChars());
        m->semantic3();
    }
    if (global.errors)
        fatal();

#if !IN_LLVM
    // Scan for functions to inline
    if (global.params.useInline)
    {
        /* The problem with useArrayBounds and useAssert is that the
         * module being linked to may not have generated them, so if
         * we inline functions from those modules, the symbols for them will
         * not be found at link time.
         */
        if (!global.params.useArrayBounds && !global.params.useAssert)
#else
    // This doesn't play nice with debug info at the moment
    if (!global.params.symdebug && willInline())
    {
        global.params.useAvailableExternally = true;
        Logger::println("Running some extra semantic3's for inlining purposes");
#endif
        {
            // Do pass 3 semantic analysis on all imported modules,
            // since otherwise functions in them cannot be inlined
            for (unsigned i = 0; i < Module::amodules.dim; i++)
            {
                m = (Module *)Module::amodules.data[i];
                if (global.params.verbose)
                    printf("semantic3 %s\n", m->toChars());
                m->semantic3();
            }
            if (global.errors)
                fatal();
        }

#if !IN_LLVM
        for (int i = 0; i < modules.dim; i++)
        {
            m = (Module *)modules.data[i];
            if (global.params.verbose)
                printf("inline scan %s\n", m->toChars());
            m->inlineScan();
        }
#endif
    }
    if (global.errors || global.warnings)
        fatal();

    // write module dependencies to file if requested
    if (global.params.moduleDepsFile != NULL)
    {
        assert (global.params.moduleDepsFile != NULL);

        File deps(global.params.moduleDepsFile);
        OutBuffer* ob = global.params.moduleDeps;
        deps.setbuffer((void*)ob->data, ob->offset);
        deps.write();
    }

    // collects llvm modules to be linked if singleobj is passed
    std::vector<llvm::Module*> llvmModules;
    llvm::LLVMContext& context = llvm::getGlobalContext();

    // Generate output files
    for (unsigned i = 0; i < modules.dim; i++)
    {
        m = (Module *)modules.data[i];
        if (global.params.verbose)
            printf("code      %s\n", m->toChars());
        if (global.params.obj)
        {
            llvm::Module* lm = m->genLLVMModule(context, &ir);
            if (!singleObj)
            {
                m->deleteObjFile();
                writeModule(lm, m->objfile->name->str);
                global.params.objfiles->push(m->objfile->name->str);
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
        Module* m = (Module*)modules.data[0];
        char* name = m->toChars();
        char* filename = m->objfile->name->str;

        llvm::Linker linker(name, name, context);

        std::string errormsg;
        for (int i = 0; i < llvmModules.size(); i++)
        {
            if(linker.LinkInModule(llvmModules[i], &errormsg))
                error("%s", errormsg.c_str());
            delete llvmModules[i];
        }

        m->deleteObjFile();
        writeModule(linker.getModule(), filename);
        global.params.objfiles->push(filename);
    }

    // output json file
    if (global.params.doXGeneration)
        json_generate(&modules);

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
                    m = (Module *)modules.data[i];
                    m->deleteObjFile();
                }
                deleteExecutable();
            }
        }
    }

    return status;
}
