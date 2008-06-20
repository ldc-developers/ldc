#include "gen/llvm.h"
#include "llvm/Linker.h"
#include "llvm/System/Program.h"

#include "root.h"
#include "mars.h"
#include "module.h"

#define NO_COUT_LOGGER
#include "gen/logger.h"

//////////////////////////////////////////////////////////////////////////////

typedef std::vector<llvm::Module*> Module_vector;

void linkModules(llvm::Module* dst, const Module_vector& MV)
{
    if (MV.empty())
        return;

    llvm::Linker linker("llvmdc", dst);

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

int linkExecutable()
{
    Logger::println("*** Linking executable ***");

    // error string
    std::string errstr;

    // find the llvm-ld program
    llvm::sys::Path ldpath = llvm::sys::Program::FindProgramByName("llvm-ld");
    if (ldpath.isEmpty())
    {
        error("linker program not found");
        fatal();
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
    if (global.params.isWindows)
        exestr.append(".exe");

    std::string outopt = "-o=" + exestr;
    args.push_back(outopt.c_str());

    // set the global gExePath
    gExePath.set(exestr);
    assert(gExePath.isValid());

    // create path to exe
    llvm::sys::Path exedir(gExePath);
    exedir.set(gExePath.getDirname());
    exedir.createDirectoryOnDisk(true, &errstr);
    if (!errstr.empty())
    {
        error("failed to create path to linking output\n%s", errstr.c_str());
        fatal();
    }

    // strip debug info
    if (!global.params.symdebug)
        args.push_back("-strip-debug");

    // optimization level
    if (!global.params.optimize)
        args.push_back("-disable-opt");
    else
    {
        const char* s = 0;
        switch(global.params.optimizeLevel)
        {
        case 0:
            s = "-O0"; break;
        case 1:
            s = "-O1"; break;
        case 2:
            s = "-O2"; break;
        case 3:
            s = "-O3"; break;
        case 4:
            s = "-O4"; break;
        case 5:
            s = "-O5"; break;
        default:
            assert(0);
        }
        args.push_back(s);
    }

    // inlining
    if (!(global.params.useInline || global.params.llvmInline))
    {
        args.push_back("-disable-inlining");
    }

    // additional linker switches
    for (int i = 0; i < global.params.linkswitches->dim; i++)
    {
        char *p = (char *)global.params.linkswitches->data[i];
        args.push_back(p);
    }

    // native please
    args.push_back("-native");


    // user libs
    for (int i = 0; i < global.params.libfiles->dim; i++)
    {
        char *p = (char *)global.params.libfiles->data[i];
        args.push_back(p);
    }

    // default libs
    args.push_back("-ltango-base-c-llvmdc");
    args.push_back("-lpthread");
    args.push_back("-ldl");
    args.push_back("-lm");

    // object files
    for (int i = 0; i < global.params.objfiles->dim; i++)
    {
        char *p = (char *)global.params.objfiles->data[i];
        args.push_back(p);
    }

    // runtime library
    // must be linked in last to null terminate the moduleinfo appending list
    std::string runtime_path(global.params.runtimePath);
    if (*runtime_path.rbegin() != '/')
        runtime_path.append("/");
    runtime_path.append("libtango-base-llvmdc.a");
    args.push_back(runtime_path.c_str());

    // print link command?
    if (!global.params.quiet || global.params.verbose)
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
        fatal();
    }
}

//////////////////////////////////////////////////////////////////////////////

void deleteExecutable()
{
    if (!gExePath.isEmpty())
    {
        assert(gExePath.isValid());
        assert(!gExePath.isDirectory());
        gExePath.eraseFromDisk(false);
    }
}

//////////////////////////////////////////////////////////////////////////////

int runExectuable()
{
    assert(!gExePath.isEmpty());
    assert(gExePath.isValid());

    // build arguments
    std::vector<const char*> args;
    for (size_t i = 0; i < global.params.runargs_length; i++)
    {
        char *a = global.params.runargs[i];
        args.push_back(a);
    }
    // terminate args list
    args.push_back(NULL);

    // try to call linker!!!
    std::string errstr;
    int status = llvm::sys::Program::ExecuteAndWait(gExePath, &args[0], NULL, NULL, 0,0, &errstr);
    if (!errstr.empty())
    {
        error("failed to execute program");
        if (!errstr.empty())
            error("error message: %s", errstr.c_str());
        fatal();
    }
    return status;
}
