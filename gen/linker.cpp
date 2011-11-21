#include "gen/linker.h"
#include "gen/llvm.h"
#include "llvm/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#if _WIN32
#include "llvm/Support/SystemUtils.h"
#endif

#include "root.h"
#include "mars.h"
#include "module.h"

#define NO_COUT_LOGGER
#include "gen/logger.h"
#include "gen/cl_options.h"
#include "gen/optimizer.h"
#include "gen/programs.h"

//////////////////////////////////////////////////////////////////////////////

// Is this useful?
llvm::cl::opt<bool> quiet("quiet",
    llvm::cl::desc("Suppress output of link command (unless -v is also passed)"),
    llvm::cl::Hidden,
    llvm::cl::ZeroOrMore,
    llvm::cl::init(true));

//////////////////////////////////////////////////////////////////////////////

typedef std::vector<llvm::Module*> Module_vector;

void linkModules(llvm::Module* dst, const Module_vector& MV)
{
    if (MV.empty())
        return;

    llvm::Linker linker("ldc", dst);

    std::string err;
    for (Module_vector::const_iterator i=MV.begin(); i!=MV.end(); ++i)
    {
        if (!linker.LinkInModule(*i, &err))
        {
            error("%s", err.c_str());
            fatal();
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

static llvm::sys::Path gExePath;

int linkExecutable(const char* argv0)
{
    Logger::println("*** Linking executable ***");

    // error string
    std::string errstr;

    // find the llvm-ld program
	llvm::sys::Path ldpath = llvm::sys::Program::FindProgramByName("llvm-ld");
    if (ldpath.isEmpty())
    {
		ldpath.set("llvm-ld");
    }

    // build arguments
    std::vector<const char*> args;

    // first the program name ??
    args.push_back("llvm-ld");

    // output filename
    std::string exestr;
    if (global.params.exefile)
    {   // explicit
        exestr = global.params.exefile;
    }
    else
    {   // inferred
        // try root module name
        if (Module::rootModule)
            exestr = Module::rootModule->toChars();
        else
            exestr = "a.out";
    }
    if (global.params.os == OSWindows && !(exestr.substr(exestr.length()-4) == ".exe"))
        exestr.append(".exe");

    std::string outopt = "-o=" + exestr;
    args.push_back(outopt.c_str());

    // set the global gExePath
    gExePath.set(exestr);
    assert(gExePath.isValid());

    // create path to exe
    llvm::sys::Path exedir(llvm::sys::path::parent_path(gExePath.str()));
    if (!llvm::sys::fs::exists(exedir.str()))
    {
        exedir.createDirectoryOnDisk(true, &errstr);
        if (!errstr.empty())
        {
            error("failed to create path to linking output: %s\n%s", exedir.c_str(), errstr.c_str());
            fatal();
        }
    }

    // strip debug info
    if (!global.params.symdebug)
        args.push_back("-strip-debug");

    // optimization level
    if (!optimize())
        args.push_back("-disable-opt");
    else
    {
        switch(optLevel())
        {
        case 0:
            args.push_back("-disable-opt");
            break;
        case 1:
            args.push_back("-globaldce");
            args.push_back("-disable-opt");
            args.push_back("-globaldce");
            args.push_back("-mem2reg");
        case 2:
        case 3:
        case 4:
        case 5:
            // use default optimization
            break;
        default:
            assert(0);
        }
    }

    // inlining
    if (!(global.params.useInline || doInline()))
    {
        args.push_back("-disable-inlining");
    }

    // additional linker switches
    for (unsigned i = 0; i < global.params.linkswitches->dim; i++)
    {
        char *p = (char *)global.params.linkswitches->data[i];
        args.push_back(p);
    }

    // native please
    args.push_back("-native");


    // user libs
    for (unsigned i = 0; i < global.params.libfiles->dim; i++)
    {
        char *p = (char *)global.params.libfiles->data[i];
        args.push_back(p);
    }

    // default libs
    switch(global.params.os) {
    case OSLinux:
    case OSMacOSX:
        args.push_back("-ldl");
    case OSFreeBSD:
        args.push_back("-lpthread");
        args.push_back("-lm");
        break;
    case OSHaiku:
        args.push_back("-lroot");
        break;
    case OSWindows:
        // FIXME: I'd assume kernel32 etc
        break;
    }

    // object files
    for (unsigned i = 0; i < global.params.objfiles->dim; i++)
    {
        char *p = (char *)global.params.objfiles->data[i];
        args.push_back(p);
    }

    // print link command?
    if (!quiet || global.params.verbose)
    {
        // Print it
        for (int i = 0; i < args.size(); i++)
            printf("%s ", args[i]);
        printf("\n");
        fflush(stdout);
    }

    // terminate args list
    args.push_back(NULL);

    // try to call linker!!!
    if (int status = llvm::sys::Program::ExecuteAndWait(ldpath, &args[0], NULL, NULL, 0,0, &errstr))
    {
        error("linking failed:\nstatus: %d", status);
        if (!errstr.empty())
            error("message: %s", errstr.c_str());
        return status;
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////

int linkObjToExecutable(const char* argv0)
{
    Logger::println("*** Linking executable ***");

    // error string
    std::string errstr;

    // find gcc for linking
    llvm::sys::Path gcc = getGcc();
    // get a string version for argv[0]
    const char* gccStr = gcc.c_str();

    // build arguments
    std::vector<const char*> args;

    // first the program name ??
    args.push_back(gccStr);

    // object files
    for (unsigned i = 0; i < global.params.objfiles->dim; i++)
    {
        char *p = (char *)global.params.objfiles->data[i];
        args.push_back(p);
    }

    // output filename
    std::string exestr;
    if (global.params.exefile)
    {   // explicit
        exestr = global.params.exefile;
    }
    else
    {   // inferred
        // try root module name
        if (Module::rootModule)
            exestr = Module::rootModule->toChars();
        else if (global.params.objfiles->dim)
            exestr = FileName::removeExt((char*)global.params.objfiles->data[0]);
        else
            exestr = "a.out";
    }
    if (global.params.os == OSWindows && !(exestr.rfind(".exe") == exestr.length()-4))
        exestr.append(".exe");

    args.push_back("-o");
    args.push_back(exestr.c_str());

    // set the global gExePath
    gExePath.set(exestr);
    assert(gExePath.isValid());

    // create path to exe
    llvm::sys::Path exedir(llvm::sys::path::parent_path(gExePath.str()));
    if (!llvm::sys::fs::exists(exedir.str()))
    {
        exedir.createDirectoryOnDisk(true, &errstr);
        if (!errstr.empty())
        {
            error("failed to create path to linking output: %s\n%s", exedir.c_str(), errstr.c_str());
            fatal();
        }
    }

    // additional linker switches
    for (unsigned i = 0; i < global.params.linkswitches->dim; i++)
    {
        char *p = (char *)global.params.linkswitches->data[i];
        args.push_back(p);
    }

    // user libs
    for (unsigned i = 0; i < global.params.libfiles->dim; i++)
    {
        char *p = (char *)global.params.libfiles->data[i];
        args.push_back(p);
    }

    // default libs
    switch(global.params.os) {
    case OSLinux:
        args.push_back("-lrt");
        // fallthrough
    case OSMacOSX:
        args.push_back("-ldl");
        // fallthrough
    case OSFreeBSD:
        args.push_back("-lpthread");
        args.push_back("-lm");
        break;

    case OSSolaris:
        args.push_back("-lm");
        args.push_back("-lumem");
        // solaris TODO
        break;

    case OSWindows:
        // FIXME: I'd assume kernel32 etc
        break;
    }

    //FIXME: enforce 64 bit
    if (global.params.is64bit)
        args.push_back("-m64");
    else
        // Assume 32-bit?
        args.push_back("-m32");

    // print link command?
    if (!quiet || global.params.verbose)
    {
        // Print it
        for (int i = 0; i < args.size(); i++)
            printf("%s ", args[i]);
        printf("\n");
        fflush(stdout);
    }

    Logger::println("Linking with: ");
    std::vector<const char*>::const_iterator I = args.begin(), E = args.end();
    Stream logstr = Logger::cout();
    for (; I != E; ++I)
        if (*I)
            logstr << "'" << *I << "'" << " ";
    logstr << "\n"; // FIXME where's flush ?


    // terminate args list
    args.push_back(NULL);

    // try to call linker
    if (int status = llvm::sys::Program::ExecuteAndWait(gcc, &args[0], NULL, NULL, 0,0, &errstr))
    {
        error("linking failed:\nstatus: %d", status);
        if (!errstr.empty())
            error("message: %s", errstr.c_str());
        return status;
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////

void deleteExecutable()
{
    if (!gExePath.isEmpty())
    {
        assert(gExePath.isValid());
        bool is_directory;
        assert(!(!llvm::sys::fs::is_directory(gExePath.str(), is_directory) && is_directory));
        gExePath.eraseFromDisk(false);
    }
}

//////////////////////////////////////////////////////////////////////////////

int runExecutable()
{
    assert(!gExePath.isEmpty());
    assert(gExePath.isValid());

    // build arguments
    std::vector<const char*> args;
    // args[0] should be the name of the executable
    args.push_back(gExePath.c_str());
    // Skip first argument to -run; it's a D source file.
    for (size_t i = 1, length = opts::runargs.size(); i < length; i++)
    {
        args.push_back(opts::runargs[i].c_str());
    }
    // terminate args list
    args.push_back(NULL);

    // try to call linker!!!
    std::string errstr;
    int status = llvm::sys::Program::ExecuteAndWait(gExePath, &args[0], NULL, NULL, 0,0, &errstr);
    if (status < 0)
    {
#if defined(_MSC_VER)
        error("program received signal %d", -status);
#else
        error("program received signal %d (%s)", -status, strsignal(-status));
#endif
        return -status;
    }

    if (!errstr.empty())
    {
        error("failed to execute program");
        if (!errstr.empty())
            error("error message: %s", errstr.c_str());
        fatal();
    }
    return status;
}
