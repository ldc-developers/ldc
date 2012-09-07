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
#include "gen/optimizer.h"
#include "gen/programs.h"

#include "driver/linker.h"
#include "driver/cl_options.h"

//////////////////////////////////////////////////////////////////////////////

// Is this useful?
llvm::cl::opt<bool> quiet("quiet",
    llvm::cl::desc("Suppress output of link command (unless -v is also passed)"),
    llvm::cl::Hidden,
    llvm::cl::ZeroOrMore,
    llvm::cl::init(true));

//////////////////////////////////////////////////////////////////////////////

bool endsWith(const std::string &str, const std::string &end)
{
    return (str.length() >= end.length() && std::equal(end.rbegin(), end.rend(), str.rbegin()));
}

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

int linkObjToBinary(bool sharedLib)
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
        char *p = static_cast<char *>(global.params.objfiles->data[i]);
        args.push_back(p);
    }

    // user libs
    for (unsigned i = 0; i < global.params.libfiles->dim; i++)
    {
        char *p = static_cast<char *>(global.params.libfiles->data[i]);
        args.push_back(p);
    }

    // output filename
    std::string output;
    if (!sharedLib && global.params.exefile)
    {   // explicit
        output = global.params.exefile;
    }
    else if (sharedLib && global.params.objname)
    {   // explicit
        output = global.params.objname;
    }
    else
    {   // inferred
        // try root module name
        if (Module::rootModule)
            output = Module::rootModule->toChars();
        else if (global.params.objfiles->dim)
            output = FileName::removeExt(static_cast<char*>(global.params.objfiles->data[0]));
        else
            output = "a.out";

        if (sharedLib) {
            std::string libExt = std::string(".") + global.dll_ext;
            if (!endsWith(output, libExt))
            {
                if (global.params.os != OSWindows)
                    output = "lib" + output + libExt;
                else
                    output.append(libExt);
            }
        } else if (global.params.os == OSWindows && !endsWith(output, ".exe")) {
            output.append(".exe");
        }
    }

    if (sharedLib)
        args.push_back("-shared");

    args.push_back("-o");
    args.push_back(output.c_str());

    // set the global gExePath
    gExePath.set(output);
    assert(gExePath.isValid());

    // create path to exe
    llvm::sys::Path exedir(llvm::sys::path::parent_path(gExePath.str()));
    if (!exedir.empty() && !llvm::sys::fs::exists(exedir.str()))
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
        char *p = static_cast<char *>(global.params.linkswitches->data[i]);
        args.push_back("-Xlinker");
        args.push_back(p);
    }

    // default libs
    bool addSoname = false;
    switch(global.params.os) {
    case OSLinux:
        addSoname = true;
        args.push_back("-lrt");
        // fallthrough
    case OSMacOSX:
        args.push_back("-ldl");
        // fallthrough
    case OSFreeBSD:
        addSoname = true;
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

    OutBuffer buf;
    if (opts::createSharedLib && addSoname) {
        std::string soname = opts::soname;
        if (!soname.empty()) {
            buf.writestring("-Wl,-soname,");
            buf.writestring(soname.c_str());
            args.push_back(buf.toChars());
        }
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

void createStaticLibrary()
{
    Logger::println("*** Creating static library ***");

    // error string
    std::string errstr;

    // find archiver
    llvm::sys::Path ar = getArchiver();

    // build arguments
    std::vector<const char*> args;

    // first the program name ??
    args.push_back(ar.c_str());

    // ask ar to create a new library
    args.push_back("rcs");

    // output filename
    std::string libName;
    if (global.params.objname)
    {   // explicit
        libName = global.params.objname;
    }
    else
    {   // inferred
        // try root module name
        if (Module::rootModule)
            libName = Module::rootModule->toChars();
        else if (global.params.objfiles->dim)
            libName = FileName::removeExt(static_cast<char*>(global.params.objfiles->data[0]));
        else
            libName = "a.out";
    }
    std::string libExt = std::string(".") + global.lib_ext;
    if (!endsWith(libName, libExt))
    {
        if (global.params.os != OSWindows)
            libName = "lib" + libName + libExt;
        else
            libName.append(libExt);
    }
    args.push_back(libName.c_str());

    // object files
    for (unsigned i = 0; i < global.params.objfiles->dim; i++)
    {
        char *p = static_cast<char *>(global.params.objfiles->data[i]);
        args.push_back(p);
    }

    // create path to the library
    llvm::sys::Path libdir(llvm::sys::path::parent_path(libName.c_str()));
    if (!libdir.empty() && !llvm::sys::fs::exists(libdir.str()))
    {
        libdir.createDirectoryOnDisk(true, &errstr);
        if (!errstr.empty())
        {
            error("failed to create path to linking output: %s\n%s", libdir.c_str(), errstr.c_str());
            fatal();
        }
    }

    // print the command?
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

    // try to call archiver
    if (int status = llvm::sys::Program::ExecuteAndWait(ar, &args[0], NULL, NULL, 0,0, &errstr))
    {
        error("archiver failed:\nstatus: %d", status);
        if (!errstr.empty())
            error("message: %s", errstr.c_str());
    }
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
