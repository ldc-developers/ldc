// Compiler implementation of the D programming language
// Copyright (c) 1999-2009 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <limits.h>
#include <string>
#include <cstdarg>

#if __DMC__
#include <dos.h>
#endif

#if POSIX
#include <errno.h>
#elif _WIN32
#include <windows.h>
#endif

#include "mem.h"
#include "root.h"

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "id.h"
#include "cond.h"
#include "expression.h"
#include "lexer.h"

#include "gen/logger.h"
#include "gen/linker.h"
#include "revisions.h"

#include "gen/cl_options.h"
#include "gen/cl_helpers.h"
using namespace opts;


static cl::opt<bool> forceBE("forcebe",
    cl::desc("Force big-endian"),
    cl::Hidden,
    cl::ZeroOrMore);

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


void getenv_setargv(const char *envvar, int *pargc, char** *pargv);

Global global;

Global::Global()
{
    mars_ext = "d";
    sym_ext  = "d";
    hdr_ext  = "di";
    doc_ext  = "html";
    ddoc_ext = "ddoc";

// LDC
    ll_ext  = "ll";
    bc_ext  = "bc";
    s_ext   = "s";
    obj_ext = "o";
#if _WIN32
    obj_ext_alt = "obj";
#endif

    copyright = "Copyright (c) 1999-2009 by Digital Mars and Tomas Lindquist Olsen";
    written = "written by Walter Bright and Tomas Lindquist Olsen";
    version = "v1.039";
    ldc_version = LDC_REV;
    llvm_version = LLVM_REV;
    global.structalign = 8;

    // This should only be used as a global, so the other fields are
    // automatically initialized to zero when the program is loaded.
    // In particular, DO NOT zero-initialize .params here (like DMD
    // does) because command-line options initialize some of those
    // fields to non-zero defaults, and do so from constructors that
    // may run before this one.
}

char *Loc::toChars() const
{
    OutBuffer buf;

    if (filename)
    {
	buf.printf("%s", filename);
    }

    if (linnum)
	buf.printf("(%d)", linnum);
    buf.writeByte(0);
    return (char *)buf.extractData();
}

Loc::Loc(Module *mod, unsigned linnum)
{
    this->linnum = linnum;
    this->filename = mod ? mod->srcfile->toChars() : NULL;
}

/**************************************
 * Print error message and exit.
 */

void error(Loc loc, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    verror(loc, format, ap);
    va_end( ap );
}

void verror(Loc loc, const char *format, va_list ap)
{
    if (!global.gag)
    {
	char *p = loc.toChars();

	if (*p)
	    fprintf(stdmsg, "%s: ", p);
	mem.free(p);

	fprintf(stdmsg, "Error: ");
	vfprintf(stdmsg, format, ap);
	fprintf(stdmsg, "\n");
	fflush(stdmsg);
    }
    global.errors++;
}

/***************************************
 * Call this after printing out fatal error messages to clean up and exit
 * the compiler.
 */

void fatal()
{
#if 0
    halt();
#endif
    exit(EXIT_FAILURE);
}

/**************************************
 * Try to stop forgetting to remove the breakpoints from
 * release builds.
 */
void halt()
{
#ifdef DEBUG
    *(char*)0=0;
#endif
}

extern void backend_init();
extern void backend_term();

void printVersion() {
    printf("LLVM D Compiler %s\nbased on DMD %s and %s\n%s\n%s\n",
    global.ldc_version, global.version, global.llvm_version, global.copyright, global.written);
    printf("D Language Documentation: http://www.digitalmars.com/d/1.0/index.html\n"
           "LDC Homepage: http://www.dsource.org/projects/ldc\n");
}

// Helper function to handle -d-debug=* and -d-version=*
static void processVersions(std::vector<std::string>& list, char* type,
        void (*setLevel)(unsigned), void (*addIdent)(char*)) {
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

int main(int argc, char *argv[])
{
    int i;
    Array files;
    char *p, *ext;
    Module *m;
    int status = EXIT_SUCCESS;

    // Set default values
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

    global.params.is64bit = sizeof(void*) == 8 ? 1 : 0;

    uint16_t endiantest = 0xFF00;
    uint8_t endianres = ((uint8_t*)&endiantest)[0];
    if (endianres == 0x00)
        global.params.isLE = true;
    else if (endianres == 0xFF)
        global.params.isLE = false;
    else {
        error("Endian test is broken");
        fatal();
    }

    // Predefine version identifiers
#if IN_LLVM
    VersionCondition::addPredefinedGlobalIdent("LLVM");
    VersionCondition::addPredefinedGlobalIdent("LDC");
#endif

    // setup default target os to be build os
#if _WIN32
    global.params.os = OSWindows;
#elif linux
    global.params.os = OSLinux;
#elif __APPLE__
    global.params.os = OSMacOSX;
#elif __FreeBSD__
    global.params.os = OSFreeBSD;
#elif defined (__SVR4) && defined (__sun)
    global.params.os = OSSolaris;
#else
 #error Unsupported OS
#endif /* linux */

    assert(global.params.os != OSinvalid);

    //VersionCondition::addPredefinedGlobalIdent("D_Bits");
    VersionCondition::addPredefinedGlobalIdent("all");

//#if _WIN32
//    inifile(global.params.argv0, "ldc.ini");
//#elif POSIX
    inifile(global.params.argv0, "ldc.conf");
//#else
//#error
//#endif
    getenv_setargv("DFLAGS", &argc, &argv);

#if 0
    for (i = 0; i < argc; i++)
    {
	printf("argv[%d] = '%s'\n", i, argv[i]);
    }
#endif

    cl::SetVersionPrinter(&printVersion);
    cl::ParseCommandLineOptions(argc, argv, "LLVM-based D Compiler\n");

    global.params.optimize = (global.params.optimizeLevel >= 0);

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

    if (mArch)
        global.params.llvmArch = mArch->Name;

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

    bool allowForceEndianness = false;

    if (global.params.llvmArch == 0) {
    #if defined(__x86_64__) || defined(_M_X64)
        global.params.llvmArch = "x86-64";
    #elif defined(__i386__) || defined(_M_IX86)
        global.params.llvmArch = "x86";
    #elif defined(__ppc__) || defined(_M_PPC)
        if (global.params.is64bit)
            global.params.llvmArch = "ppc64";
        else
            global.params.llvmArch = "ppc32";
    #elif defined(__arm__)
        global.params.llvmArch = "arm";
    #elif defined(__thumb__)
        global.params.llvmArch = "thumb";
    #else
    #error
    #endif
    }

    if (strcmp(global.params.llvmArch,"x86")==0) {
        VersionCondition::addPredefinedGlobalIdent("X86");
        global.params.isLE = true;
        global.params.is64bit = false;
        global.params.cpu = ARCHx86;
        if (global.params.useInlineAsm) {
            VersionCondition::addPredefinedGlobalIdent("LLVM_InlineAsm_X86");
        }
    }
    else if (strcmp(global.params.llvmArch,"x86-64")==0) {
        VersionCondition::addPredefinedGlobalIdent("X86_64");
        global.params.isLE = true;
        global.params.is64bit = true;
        global.params.cpu = ARCHx86_64;
        if (global.params.useInlineAsm) {
            VersionCondition::addPredefinedGlobalIdent("LLVM_InlineAsm_X86_64");
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
    }

    assert(global.params.cpu != ARCHinvalid);

    if (allowForceEndianness && forceBE) {
        VersionCondition::addPredefinedGlobalIdent("BigEndian");
        global.params.isLE = false;
    }
    else if (global.params.isLE) {
        VersionCondition::addPredefinedGlobalIdent("LittleEndian");
    }
    else {
        VersionCondition::addPredefinedGlobalIdent("BigEndian");
    }

    if (global.params.is64bit) {
        VersionCondition::addPredefinedGlobalIdent("LLVM64");
    }


    // setup version idents and tt_os for chosen target os
    switch(global.params.os)
    {
    case OSWindows:
    // TODO Win64 stuff!
	VersionCondition::addPredefinedGlobalIdent("Windows");
	VersionCondition::addPredefinedGlobalIdent("Win32");
	VersionCondition::addPredefinedGlobalIdent("mingw32");
	break;

    case OSLinux:
	VersionCondition::addPredefinedGlobalIdent("linux");
	VersionCondition::addPredefinedGlobalIdent("Posix");
	break;

    case OSMacOSX:
	VersionCondition::addPredefinedGlobalIdent("OSX");
	VersionCondition::addPredefinedGlobalIdent("darwin");
	VersionCondition::addPredefinedGlobalIdent("Posix");
	break;

    case OSFreeBSD:
	VersionCondition::addPredefinedGlobalIdent("freebsd");
	VersionCondition::addPredefinedGlobalIdent("Posix");
	break;

    case OSSolaris:
	VersionCondition::addPredefinedGlobalIdent("solaris");
	VersionCondition::addPredefinedGlobalIdent("Posix");
	break;

    default:
	assert(false && "Target OS not supported");
    }

    if (!global.params.targetTriple)
        global.params.targetTriple = DEFAULT_TARGET_TRIPLE;

    Logger::println("Target triple: %s", global.params.targetTriple);

    // build a minimal data layout so llvm can find the target
    global.params.dataLayout = global.params.isLE
        ? (char*)(global.params.is64bit ? "e-p:64:64" : "e-p:32:32")
        : (char*)(global.params.is64bit ? "E-p:64:64" : "E-p:32:32");
    Logger::println("Layout: %s", global.params.dataLayout);

    // added in 1.039
    if (global.params.doDocComments)
        VersionCondition::addPredefinedGlobalIdent("D_Ddoc");

    // Initialization
    Type::init();
    Id::initialize();
    Module::init();
    initPrecedence();

    backend_init();

    //printf("%d source files\n",files.dim);

    // Build import search path
    if (global.params.imppath)
    {
	for (i = 0; i < global.params.imppath->dim; i++)
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
	for (i = 0; i < global.params.fileImppath->dim; i++)
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
    for (i = 0; i < files.dim; i++)
    {	Identifier *id;
	char *ext;
	char *name;

	p = (char *) files.data[i];

	p = FileName::name(p);		// strip path
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
		ext--;			// skip onto '.'
		assert(*ext == '.');
		name = (char *)mem.malloc((ext - p) + 1);
		memcpy(name, p, ext - p);
		name[ext - p] = 0;		// strip extension

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
	    {	error("unrecognized file extension %s\n", ext);
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
    for (i = 0; i < modules.dim; i++)
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
	for (i = 0; i < modules.dim; i++)
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
    for (i = 0; i < modules.dim; i++)
    {
	m = (Module *)modules.data[i];
	if (global.params.verbose)
	    printf("semantic  %s\n", m->toChars());
	m->semantic();
    }
    if (global.errors)
	fatal();

    // Do pass 2 semantic analysis
    for (i = 0; i < modules.dim; i++)
    {
	m = (Module *)modules.data[i];
	if (global.params.verbose)
	    printf("semantic2 %s\n", m->toChars());
	m->semantic2();
    }
    if (global.errors)
	fatal();

    // Do pass 3 semantic analysis
    for (i = 0; i < modules.dim; i++)
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
	    for (i = 0; i < Module::amodules.dim; i++)
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

	for (i = 0; i < modules.dim; i++)
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

    // Generate output files
    for (i = 0; i < modules.dim; i++)
    {
	m = (Module *)modules.data[i];
	if (global.params.verbose)
	    printf("code      %s\n", m->toChars());
	if (global.params.obj)
	{
	    m->genobjfile(0);
	    global.params.objfiles->push(m->objfile->name->str);
	}
	if (global.errors)
	    m->deleteObjFile();
	else
	{
	    if (global.params.doDocComments)
		m->gendocfile();
	}
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
		for (i = 0; i < modules.dim; i++)
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



/***********************************
 * Parse and append contents of environment variable envvar
 * to argc and argv[].
 * The string is separated into arguments, processing \ and ".
 */

void getenv_setargv(const char *envvar, int *pargc, char** *pargv)
{
    char *env;
    char *p;
    Array *argv;
    int argc;

    int wildcard;		// do wildcard expansion
    int instring;
    int slash;
    char c;
    int j;

    env = getenv(envvar);
    if (!env)
	return;

    env = mem.strdup(env);	// create our own writable copy

    argc = *pargc;
    argv = new Array();
    argv->setDim(argc);

    int argc_left = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp((*pargv)[i], "-run") || !strcmp((*pargv)[i], "--run")) {
            // HACK: set flag to indicate we saw '-run' here
            global.params.run = true;
            // Don't eat -run yet so the program arguments don't get changed
            argc_left = argc - i;
            argc = i;
            *pargv = &(*pargv)[i];
            argv->setDim(i);
            break;
        } else {
            argv->data[i] = (void *)(*pargv)[i];
        }
    }
    // HACK to stop required values from command line being drawn from DFLAGS
    argv->push((char*)"");
    argc++;

    j = 1;			// leave argv[0] alone
    while (1)
    {
	wildcard = 1;
	switch (*env)
	{
	    case ' ':
	    case '\t':
		env++;
		break;

	    case 0:
		goto Ldone;

	    case '"':
		wildcard = 0;
	    default:
		argv->push(env);		// append
		//argv->insert(j, env);		// insert at position j
		j++;
		argc++;
		p = env;
		slash = 0;
		instring = 0;
		c = 0;

		while (1)
		{
		    c = *env++;
		    switch (c)
		    {
			case '"':
			    p -= (slash >> 1);
			    if (slash & 1)
			    {	p--;
				goto Laddc;
			    }
			    instring ^= 1;
			    slash = 0;
			    continue;

			case ' ':
			case '\t':
			    if (instring)
				goto Laddc;
			    *p = 0;
			    //if (wildcard)
				//wildcardexpand();	// not implemented
			    break;

			case '\\':
			    slash++;
			    *p++ = c;
			    continue;

			case 0:
			    *p = 0;
			    //if (wildcard)
				//wildcardexpand();	// not implemented
			    goto Ldone;

			default:
			Laddc:
			    slash = 0;
			    *p++ = c;
			    continue;
		    }
		    break;
		}
	}
    }

Ldone:
    assert(argc == argv->dim);
    argv->reserve(argc_left);
    for (int i = 0; i < argc_left; i++)
        argv->data[argc++] = (void *)(*pargv)[i];

    *pargc = argc;
    *pargv = (char **)argv->data;
}
