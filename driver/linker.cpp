//===-- linker.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "llvm/Linker.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#if _WIN32
#include "llvm/Support/SystemUtils.h"
#endif

#include "root.h"
#include "mars.h"
#include "module.h"

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

static bool endsWith(const std::string &str, const std::string &end)
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

static int ExecuteToolAndWait(llvm::sys::Path tool, std::vector<std::string> args, bool verbose = false)
{
    // Construct real argument list.
    // First entry is the tool itself, last entry must be NULL.
    std::vector<const char *> realargs;
    realargs.reserve(args.size() + 2);
    realargs.push_back(tool.c_str());
    for (std::vector<std::string>::const_iterator it = args.begin(); it != args.end(); ++it)
    {
        realargs.push_back((*it).c_str());
    }
    realargs.push_back(NULL);

    // Print command line if requested
    if (verbose)
    {
        // Print it
        for (size_t i = 0; i < realargs.size()-1; i++)
            printf("%s ", realargs[i]);
        printf("\n");
        fflush(stdout);
    }

    // Execute tool.
    std::string errstr;
    if (int status = llvm::sys::Program::ExecuteAndWait(tool, &realargs[0], NULL, NULL, 0, 0, &errstr))
    {
        error("%s failed with status: %d", tool.c_str(), status);
        if (!errstr.empty())
            error("message: %s", errstr.c_str());
        return status;
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////

static void CreateDirectoryOnDisk(llvm::StringRef fileName)
{
    llvm::sys::Path dir(llvm::sys::path::parent_path(fileName));
    if (!dir.empty() && !llvm::sys::fs::exists(dir.str()))
    {
        std::string errstr;
        dir.createDirectoryOnDisk(true, &errstr);
        if (!errstr.empty())
        {
            error("failed to create path to file: %s\n%s", dir.c_str(), errstr.c_str());
            fatal();
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

static std::string getOutputName(bool const sharedLib)
{
    if (!sharedLib && global.params.exefile)
    {
        return global.params.exefile;
    }

    if (sharedLib && global.params.objname)
    {
        return global.params.objname;
    }

    // Output name is inferred.
    std::string result;

    // try root module name
    if (Module::rootModule)
        result = Module::rootModule->toChars();
    else if (global.params.objfiles->dim)
        result = FileName::removeExt(static_cast<char*>(global.params.objfiles->data[0]));
    else
        result = "a.out";

    if (sharedLib)
    {
        std::string libExt = std::string(".") + global.dll_ext;
        if (!endsWith(result, libExt))
        {
            if (global.params.targetTriple.getOS() != llvm::Triple::Win32)
                result = "lib" + result + libExt;
            else
                result.append(libExt);
        }
    }
    else if (global.params.targetTriple.isOSWindows() && !endsWith(result, ".exe"))
    {
        result.append(".exe");
    }

    return result;
}

//////////////////////////////////////////////////////////////////////////////

static llvm::sys::Path gExePath;

static int linkObjToBinaryGcc(bool sharedLib)
{
    Logger::println("*** Linking executable ***");

    // find gcc for linking
    llvm::sys::Path gcc = getGcc();

    // build arguments
    std::vector<std::string> args;

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
    std::string output = getOutputName(sharedLib);

    if (sharedLib)
        args.push_back("-shared");

    args.push_back("-o");
    args.push_back(output);

    // set the global gExePath
    gExePath.set(output);
    assert(gExePath.isValid());

    // create path to exe
    CreateDirectoryOnDisk(gExePath.str());

    // additional linker switches
    for (unsigned i = 0; i < global.params.linkswitches->dim; i++)
    {
        char *p = static_cast<char *>(global.params.linkswitches->data[i]);
        // Don't push -l and -L switches using -Xlinker, but pass them directly
        // to GCC. This makes sure user-defined paths take precedence over
        // GCC's builtin LIBRARY_PATHs.
        if (!p[0] || !(p[0] == '-' && (p[1] == 'l' || p[1] == 'L')))
            args.push_back("-Xlinker");
        args.push_back(p);
    }

    // default libs
    bool addSoname = false;
    switch (global.params.targetTriple.getOS()) {
    case llvm::Triple::Linux:
        addSoname = true;
        args.push_back("-lrt");
        // fallthrough
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
        args.push_back("-ldl");
        // fallthrough
    case llvm::Triple::FreeBSD:
        addSoname = true;
        args.push_back("-lpthread");
        args.push_back("-lm");
        break;

    case llvm::Triple::Solaris:
        args.push_back("-lm");
        args.push_back("-lumem");
        // solaris TODO
        break;

    default:
        // OS not yet handled, will probably lead to linker errors.
        // FIXME: Win32, MinGW.
        break;
    }

    //FIXME: enforce 64 bit
    if (global.params.is64bit)
        args.push_back("-m64");
    else
        // Assume 32-bit?
        args.push_back("-m32");

    if (opts::createSharedLib && addSoname) {
        std::string soname = opts::soname;
        if (!soname.empty()) {
            args.push_back("-Wl,-soname," + soname);
        }
    }

    Logger::println("Linking with: ");
    std::vector<std::string>::const_iterator I = args.begin(), E = args.end();
    Stream logstr = Logger::cout();
    for (; I != E; ++I)
        if (!(*I).empty())
            logstr << "'" << *I << "'" << " ";
    logstr << "\n"; // FIXME where's flush ?

    // try to call linker
    return ExecuteToolAndWait(gcc, args, !quiet || global.params.verbose);
}

//////////////////////////////////////////////////////////////////////////////

static int linkObjToBinaryWin(bool sharedLib)
{
    Logger::println("*** Linking executable ***");

    // find link.exe for linking
    llvm::sys::Path tool = getLink();

    // build arguments
    std::vector<std::string> args;

    // be verbose if requested
    if (!global.params.verbose)
        args.push_back("/NOLOGO");

    // specify that the image will contain a table of safe exception handlers (32bit only)
    if (!global.params.is64bit)
        args.push_back("/SAFESEH");

    // mark executable to be compatible with Windows Data Execution Prevention feature
    args.push_back("/NXCOMPAT");

    // use address space layout randomization (ASLR) feature
    args.push_back("/DYNAMICBASE");

    // because of a LLVM bug
    args.push_back("/LARGEADDRESSAWARE:NO");

    // output debug information
    if (global.params.symdebug)
        args.push_back("/DEBUG");

    // specify creation of DLL
    if (sharedLib)
        args.push_back("/DLL");

    // output filename
    std::string output = getOutputName(sharedLib);

    args.push_back("/OUT:" + output);

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

    // set the global gExePath
    gExePath.set(output);
    assert(gExePath.isValid());

    // create path to exe
    CreateDirectoryOnDisk(gExePath.str());

    // additional linker switches
    for (unsigned i = 0; i < global.params.linkswitches->dim; i++)
    {
        static const std::string LIBPATH("-L");
        static const std::string LIB("-l");
        std::string str(static_cast<char *>(global.params.linkswitches->data[i]));
        if (str.length() > 2)
        {
            if (std::equal(LIBPATH.begin(), LIBPATH.end(), str.begin()))
                str = "/LIBPATH:" + str.substr(2);
            else if (std::equal(LIB.begin(), LIB.end(), str.begin()))
            {
                str = str.substr(2) + ".lib";
            }
        }
        args.push_back(str);
    }

    // default libs
    // TODO check which libaries are necessary
    args.push_back("kernel32.lib");
    args.push_back("user32.lib");
    args.push_back("gdi32.lib");
    args.push_back("winspool.lib");
    args.push_back("shell32.lib"); // required for dmain2.d
    args.push_back("ole32.lib");
    args.push_back("oleaut32.lib");
    args.push_back("uuid.lib");
    args.push_back("comdlg32.lib");
    args.push_back("advapi32.lib");

    Logger::println("Linking with: ");
    std::vector<std::string>::const_iterator I = args.begin(), E = args.end();
    Stream logstr = Logger::cout();
    for (; I != E; ++I)
        if (!(*I).empty())
            logstr << "'" << *I << "'" << " ";
    logstr << "\n"; // FIXME where's flush ?

    // try to call linker
    return ExecuteToolAndWait(tool, args, !quiet || global.params.verbose);
}

//////////////////////////////////////////////////////////////////////////////

int linkObjToBinary(bool sharedLib)
{
    int status;
    if (global.params.targetTriple.getOS() == llvm::Triple::Win32)
        status = linkObjToBinaryWin(sharedLib);
    else
        status = linkObjToBinaryGcc(sharedLib);
    return status;
}

//////////////////////////////////////////////////////////////////////////////

void createStaticLibrary()
{
    Logger::println("*** Creating static library ***");

    const bool isTargetWindows = global.params.targetTriple.getOS() == llvm::Triple::Win32;

    // find archiver
    llvm::sys::Path tool = isTargetWindows ? getLib() : getArchiver();

    // build arguments
    std::vector<std::string> args;

    // ask ar to create a new library
    if (!isTargetWindows)
        args.push_back("rcs");

    // ask lib to be quiet
    if (isTargetWindows)
        args.push_back("/NOLOGO");

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
        if (!isTargetWindows)
            libName = "lib" + libName + libExt;
        else
            libName.append(libExt);
    }
    if (isTargetWindows)
        args.push_back("/OUT:" + libName);
    else
        args.push_back(libName);

    // object files
    for (unsigned i = 0; i < global.params.objfiles->dim; i++)
    {
        char *p = static_cast<char *>(global.params.objfiles->data[i]);
        args.push_back(p);
    }

    // create path to the library
    CreateDirectoryOnDisk(libName);

    // try to call archiver
    ExecuteToolAndWait(tool, args, !quiet || global.params.verbose);
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

    // Run executable
    int status = ExecuteToolAndWait(gExePath, opts::runargs, !quiet || global.params.verbose);
    if (status < 0)
    {
#if defined(_MSC_VER)
        error("program received signal %d", -status);
#else
        error("program received signal %d (%s)", -status, strsignal(-status));
#endif
        return -status;
    }
    return status;
}
