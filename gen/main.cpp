// Pulled out of dmd/mars.c

// some things are taken from llvm's llc tool
// which uses the llvm license

#include "gen/llvm.h"
#include "llvm/LinkAllVMCore.h"
#include "llvm/Linker.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"

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

#include "gen/logger.h"
#include "gen/linker.h"
#include "gen/irstate.h"
#include "gen/toobj.h"

#include "gen/cl_options.h"
#include "gen/cl_helpers.h"
using namespace opts;

#include "gen/configfile.h"

extern void getenv_setargv(const char *envvar, int *pargc, char** *pargv);
extern void backend_init();
extern void backend_term();

static cl::opt<bool> noDefaultLib("nodefaultlib",
    cl::desc("Don't add a default library for linking implicitly"),
    cl::ZeroOrMore);

static ArrayAdapter impPathsStore("I", global.params.imppath);
static cl::list<std::string, ArrayAdapter> importPaths("I",
    cl::desc("Where to look for imports"),
    cl::value_desc("path"),
    cl::location(impPathsStore),
    cl::Prefix);

static ArrayAdapter defaultLibStore("defaultlib", global.params.defaultlibnames);
static cl::list<std::string, ArrayAdapter> defaultlibs("defaultlib",
    cl::desc("Set default libraries for non-debug build"),
    cl::value_desc("lib,..."),
    cl::location(defaultLibStore),
    cl::CommaSeparated);

static ArrayAdapter debugLibStore("debuglib", global.params.debuglibnames);
static cl::list<std::string, ArrayAdapter> debuglibs("debuglib",
    cl::desc("Set default libraries for debug build"),
    cl::value_desc("lib,..."),
    cl::location(debugLibStore),
    cl::CommaSeparated);

void printVersion() {
    printf("LLVM D Compiler %s\nbased on DMD %s and %s\n%s\n%s\n",
    global.ldc_version, global.version, global.llvm_version, global.copyright, global.written);
    printf("D Language Documentation: http://www.digitalmars.com/d/1.0/index.html\n"
           "LDC Homepage: http://www.dsource.org/projects/ldc\n");
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

int main(int argc, char** argv)
{
    // stack trace on signals
    llvm::sys::PrintStackTraceOnErrorSignal();

    Array files;
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

    global.params.linkswitches = new Array();
    global.params.libfiles = new Array();
    global.params.objfiles = new Array();
    global.params.ddocfiles = new Array();


    // Set predefined version identifiers
    VersionCondition::addPredefinedGlobalIdent("LLVM");
    VersionCondition::addPredefinedGlobalIdent("LDC");
    VersionCondition::addPredefinedGlobalIdent("all");
#if DMDV2
    VersionCondition::addPredefinedGlobalIdent("D_Version2");
#endif

    // merge DFLAGS environment variable into argc/argv
    getenv_setargv("DFLAGS", &argc, &argv);
#if 0
    for (int i = 0; i < argc; i++)
    {
    printf("argv[%d] = '%s'\n", i, argv[i]);
    }
#endif

    // build complete fixed up list of command line arguments
    std::vector<const char*> final_args;
    final_args.reserve(argc);

    // insert argc + DFLAGS
    final_args.insert(final_args.end(), &argv[0], &argv[argc]);

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

#if 0
    for (size_t i = 0; i < final_args.size(); ++i)
    {
        printf("final_args[%zu] = %s\n", i, final_args[i]);
    }
#endif

    // Handle fixed-up arguments!
    cl::SetVersionPrinter(&printVersion);
    cl::ParseCommandLineOptions(final_args.size(), (char**)&final_args[0], "LLVM-based D Compiler\n", true);

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
    
#ifdef _DH
    initFromString(global.params.hdrdir, hdrDir);
    initFromString(global.params.hdrname, hdrFile);
    global.params.doHdrGeneration |=
        global.params.hdrdir || global.params.hdrname;
#endif

    processVersions(debugArgs, "debug",
        DebugCondition::setGlobalLevel,
        DebugCondition::addGlobalIdent);
    processVersions(versions, "version",
        VersionCondition::setGlobalLevel,
        VersionCondition::addGlobalIdent);

    global.params.output_o =
        opts::output_o == cl::BOU_UNSET
        ? OUTPUTFLAGdefault
        : opts::output_o == cl::BOU_TRUE
            ? OUTPUTFLAGset
            : OUTPUTFLAGno;
    global.params.output_bc = opts::output_bc ? OUTPUTFLAGset : OUTPUTFLAGno;
    global.params.output_ll = opts::output_ll ? OUTPUTFLAGset : OUTPUTFLAGno;
    global.params.output_s  = opts::output_s  ? OUTPUTFLAGset : OUTPUTFLAGno;

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

    Array* libs;
    if (global.params.symdebug)
        libs = global.params.debuglibnames;
    else
        libs = global.params.defaultlibnames;

    if (libs)
    {
        for (int i = 0; i < libs->dim; i++)
        {
            char *arg = (char *)mem.malloc(64);
            strcpy(arg, "-l");
            strncat(arg, (char *)libs->data[i], 64);
            global.params.linkswitches->push(arg);
        }
    }
    else if (!noDefaultLib)
    {
        char *arg;
        arg = (char *)mem.malloc(64);
        strcpy(arg, "-lldc-runtime");
        global.params.linkswitches->push(arg);
        arg = (char *)mem.malloc(64);
        strcpy(arg, "-ltango-cc-tango");
        global.params.linkswitches->push(arg);
        arg = (char *)mem.malloc(64);
        strcpy(arg, "-ltango-gc-basic");
        global.params.linkswitches->push(arg);
        // pass the runtime again to resolve issues
        // with linking order
        arg = (char *)mem.malloc(64);
        strcpy(arg, "-lldc-runtime");
        global.params.linkswitches->push(arg);
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
    if (!global.params.obj || !global.params.output_o)
        global.params.link = 0;

    if (global.params.link)
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
        if (global.params.objname && files.dim > 1)
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
    else if ((m32bits || m64bits) && (mArch != 0 || !mTargetTriple.empty()))
        error("-m32 and -m64 switches cannot be used together with -march and -mtriple switches");
    if (global.errors)
        fatal();

    llvm::Module mod("dummy");

    // override triple if needed
    const char* defaultTriple = DEFAULT_TARGET_TRIPLE;
    if ((sizeof(void*) == 4 && m64bits) || (sizeof(void*) == 8 && m32bits))
    {
        defaultTriple = DEFAULT_ALT_TARGET_TRIPLE;
    }

    // did the user override the target triple?
    if (mTargetTriple.empty())
    {
        if (mArch != 0)
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

    mod.setTargetTriple(global.params.targetTriple);

    // Allocate target machine.  First, check whether the user has
    // explicitly specified an architecture to compile for.
    if (mArch == 0)
    {
        std::string Err;
        mArch = llvm::TargetMachineRegistry::getClosestStaticTargetForModule(mod, Err);
        if (mArch == 0)
        {
            error("failed to auto-select target: %s, please use the -march option", Err.c_str());
            fatal();
        }
    }

    // Package up features to be passed to target/subtarget
    std::string FeaturesStr;
    if (mCPU.size() || mAttrs.size())
    {
        llvm::SubtargetFeatures Features;
        Features.setCPU(mCPU);
        for (unsigned i = 0; i != mAttrs.size(); ++i)
        Features.AddFeature(mAttrs[i]);
        FeaturesStr = Features.getString();
    }

    std::auto_ptr<llvm::TargetMachine> target(mArch->CtorFn(mod, FeaturesStr));
    assert(target.get() && "Could not allocate target machine!");
    gTargetMachine = target.get();
    gTargetData = gTargetMachine->getTargetData();

    // get final data layout
    std::string datalayout = gTargetData->getStringRepresentation();
    global.params.dataLayout = datalayout.c_str();

    global.params.llvmArch = mArch->Name;

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
    std::string triple = global.params.targetTriple;
    size_t npos = std::string::npos;

    // windows
    // FIXME: win64
    if (triple.find("windows") != npos || triple.find("win32") != npos || triple.find("mingw") != npos)
    {
        global.params.os = OSWindows;
        VersionCondition::addPredefinedGlobalIdent("Windows");
        VersionCondition::addPredefinedGlobalIdent("Win32");
        VersionCondition::addPredefinedGlobalIdent("mingw32");
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
        VersionCondition::addPredefinedGlobalIdent("Posix");
    }
    // solaris
    else if (triple.find("-solaris") != npos)
    {
        global.params.os = OSSolaris;
        VersionCondition::addPredefinedGlobalIdent("solaris");
        VersionCondition::addPredefinedGlobalIdent("Posix");
    }
    // unsupported
    else
    {
        error("target triple '%s' is not supported", global.params.targetTriple);
        fatal();
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
        for (int i = 0; i < global.params.imppath->dim; i++)
        {
            char *path = (char *)global.params.imppath->data[i];
            Array *a = FileName::splitPath(path);

            if (a)
            {
                if (!global.path)
                    global.path = new Array();
                global.path->append(a);
            }
        }
    }

    // Build string import search path
    if (global.params.fileImppath)
    {
        for (int i = 0; i < global.params.fileImppath->dim; i++)
        {
            char *path = (char *)global.params.fileImppath->data[i];
            Array *a = FileName::splitPath(path);

            if (a)
            {
                if (!global.filePath)
                    global.filePath = new Array();
                global.filePath->append(a);
            }
        }
    }

    // Create Modules
    Array modules;
    modules.reserve(files.dim);
    for (int i = 0; i < files.dim; i++)
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
                global.params.objfiles->push(files.data[i]);
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
                global.params.libfiles->push(files.data[i]);
                continue;
            }

            if (strcmp(ext, global.ddoc_ext) == 0)
            {
                global.params.ddocfiles->push(files.data[i]);
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

        id = new Identifier(name, 0);
        m = new Module((char *) files.data[i], id, global.params.doDocComments, global.params.doHdrGeneration);
        modules.push(m);
    }

    // Read files, parse them
    for (int i = 0; i < modules.dim; i++)
    {
        m = (Module *)modules.data[i];
        if (global.params.verbose)
            printf("parse     %s\n", m->toChars());
        if (!Module::rootModule)
            Module::rootModule = m;
        m->importedFrom = m;
        m->read(0);
        m->parse();
        m->buildTargetFiles();
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
        for (int i = 0; i < modules.dim; i++)
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

    // Do semantic analysis
    for (int i = 0; i < modules.dim; i++)
    {
        m = (Module *)modules.data[i];
        if (global.params.verbose)
            printf("semantic  %s\n", m->toChars());
        m->semantic();
    }
    if (global.errors)
        fatal();

    // Do pass 2 semantic analysis
    for (int i = 0; i < modules.dim; i++)
    {
        m = (Module *)modules.data[i];
        if (global.params.verbose)
            printf("semantic2 %s\n", m->toChars());
        m->semantic2();
    }
    if (global.errors)
        fatal();

    // Do pass 3 semantic analysis
    for (int i = 0; i < modules.dim; i++)
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
        {
#endif
            // Do pass 3 semantic analysis on all imported modules,
            // since otherwise functions in them cannot be inlined
            for (int i = 0; i < Module::amodules.dim; i++)
            {
                m = (Module *)Module::amodules.data[i];
                if (global.params.verbose)
                    printf("semantic3 %s\n", m->toChars());
                m->semantic3();
            }
            if (global.errors)
                fatal();
#if !IN_LLVM
        }

        for (int i = 0; i < modules.dim; i++)
        {
            m = (Module *)modules.data[i];
            if (global.params.verbose)
                printf("inline scan %s\n", m->toChars());
            m->inlineScan();
        }
    }
#endif
    if (global.errors)
        fatal();

    // collects llvm modules to be linked if singleobj is passed
    std::vector<llvm::Module*> llvmModules;

    // Generate output files
    for (int i = 0; i < modules.dim; i++)
    {
        m = (Module *)modules.data[i];
        if (global.params.verbose)
            printf("code      %s\n", m->toChars());
        if (global.params.obj)
        {
            llvm::Module* lm = m->genLLVMModule(&ir);
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
        
        llvm::Linker linker(name, name);
        std::string errormsg;
        for (int i = 0; i < llvmModules.size(); i++)
        {
            if(linker.LinkInModule(llvmModules[i], &errormsg))
                error(errormsg.c_str());
            delete llvmModules[i];
        }
        
#if LLVM_REV < 66404
        // Workaround for llvm bug #3749
        // Not needed since LLVM r66404 (it no longer checks for this)
        llvm::GlobalVariable* ctors = linker.getModule()->getGlobalVariable("llvm.global_ctors");
        if (ctors) {
            ctors->removeDeadConstantUsers();
            assert(ctors->use_empty());
        }
#endif
        
        m->deleteObjFile();
        writeModule(linker.getModule(), filename);
        global.params.objfiles->push(filename);
    }
    
    backend_term();
    if (global.errors)
        fatal();

    if (!global.params.objfiles->dim)
    {
        if (global.params.link)
            error("no object files to link");
    }
    else
    {
        if (global.params.link)
            //status = runLINK();
            linkObjToExecutable(global.params.argv0);

        if (global.params.run)
        {
            if (!status)
            {
                status = runExectuable();

                /* Delete .obj files and .exe file
                 */
                for (int i = 0; i < modules.dim; i++)
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
