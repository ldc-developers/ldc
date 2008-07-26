#include "gen/llvm.h"
#include "llvm/Linker.h"
#include "llvm/System/Program.h"
#if _WIN32
#include "llvm/Support/SystemUtils.h"
#endif

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
    if (!exedir.exists())
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
    if (!global.params.optimize)
        args.push_back("-disable-opt");
    else
    {
        const char* s = 0;
        switch(global.params.optimizeLevel)
        {
        case 0:
            args.push_back("-disable-opt");
            args.push_back("-globaldce");
            break;
        case 1:
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
    if(global.params.isLinux)
    {
        args.push_back("-lpthread");
        args.push_back("-ldl");
        args.push_back("-lm");
    }
    else if (global.params.isWindows)
    {
        // FIXME: I'd assume kernel32 etc
    }

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
    runtime_path.append("libtango-base-llvmdc-native.a");
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
