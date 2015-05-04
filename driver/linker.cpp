//===-- linker.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/linker.h"
#include "mars.h"
#include "module.h"
#include "root.h"
#include "driver/cl_options.h"
#include "driver/tool.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/programs.h"
#include "llvm/ADT/Triple.h"
#if LDC_LLVM_VER >= 305
#include "llvm/Linker/Linker.h"
#else
#include "llvm/Linker.h"
#endif
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#if _WIN32
#include "llvm/Support/SystemUtils.h"
#endif

//////////////////////////////////////////////////////////////////////////////

static bool endsWith(const std::string &str, const std::string &end)
{
    return (str.length() >= end.length() && std::equal(end.rbegin(), end.rend(), str.rbegin()));
}

//////////////////////////////////////////////////////////////////////////////

static void CreateDirectoryOnDisk(llvm::StringRef fileName)
{
    llvm::StringRef dir(llvm::sys::path::parent_path(fileName));
    if (!dir.empty() && !llvm::sys::fs::exists(dir))
    {
#if LDC_LLVM_VER >= 305
        std::error_code ec = llvm::sys::fs::create_directory(dir);
#else
        bool existed;
        llvm::error_code ec = llvm::sys::fs::create_directory(dir, existed);
#endif
        if (ec)
        {
            error(Loc(), "failed to create path to file: %s\n%s", dir.data(), ec.message().c_str());
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
        result = FileName::removeExt(static_cast<const char*>(global.params.objfiles->data[0]));
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

static std::string gExePath;

static int linkObjToBinaryGcc(bool sharedLib)
{
    Logger::println("*** Linking executable ***");

    // find gcc for linking
    std::string gcc(getGcc());

    // build arguments
    std::vector<std::string> args;

    // object files
    for (unsigned i = 0; i < global.params.objfiles->dim; i++)
    {
        const char *p = static_cast<const char *>(global.params.objfiles->data[i]);
        args.push_back(p);
    }

    // user libs
    for (unsigned i = 0; i < global.params.libfiles->dim; i++)
    {
        const char *p = static_cast<const char *>(global.params.libfiles->data[i]);
        args.push_back(p);
    }

    // output filename
    std::string output = getOutputName(sharedLib);

    if (sharedLib)
        args.push_back("-shared");

    args.push_back("-o");
    args.push_back(output);

    // set the global gExePath
    gExePath = output;
    //assert(gExePath.isValid());

    // create path to exe
    CreateDirectoryOnDisk(gExePath);

#if LDC_LLVM_VER >= 303
    // Pass sanitizer arguments to linker. Requires clang.
    if (opts::sanitize == opts::AddressSanitizer) {
        args.push_back("-fsanitize=address");
    }

    if (opts::sanitize == opts::MemorySanitizer) {
        args.push_back("-fsanitize=memory");
    }

    if (opts::sanitize == opts::ThreadSanitizer) {
        args.push_back("-fsanitize=thread");
    }
#endif

    // additional linker switches
    for (unsigned i = 0; i < global.params.linkswitches->dim; i++)
    {
        const char *p = static_cast<const char *>(global.params.linkswitches->data[i]);
        // Don't push -l and -L switches using -Xlinker, but pass them indirectly
        // via GCC. This makes sure user-defined paths take precedence over
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
        if (!opts::disableLinkerStripDead) {
            args.push_back("-Wl,--gc-sections");
        }
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

#if LDC_LLVM_VER < 305
    case llvm::Triple::MinGW32:
        // This is really more of a kludge, as linking in the Winsock functions
        // should be handled by the pragma(lib, ...) in std.socket, but it
        // makes LDC behave as expected for now.
        args.push_back("-lws2_32");
        break;
#endif

    default:
        // OS not yet handled, will probably lead to linker errors.
        // FIXME: Win32.
        break;
    }

#if LDC_LLVM_VER >= 305
    if (global.params.targetTriple.isWindowsGNUEnvironment())
    {
        // This is really more of a kludge, as linking in the Winsock functions
        // should be handled by the pragma(lib, ...) in std.socket, but it
        // makes LDC behave as expected for now.
        args.push_back("-lws2_32");
    }
#endif

    // Only specify -m32/-m64 for architectures where the two variants actually
    // exist (as e.g. the GCC ARM toolchain doesn't recognize the switches).
    if (global.params.targetTriple.get64BitArchVariant().getArch() !=
        llvm::Triple::UnknownArch &&
        global.params.targetTriple.get32BitArchVariant().getArch() !=
        llvm::Triple::UnknownArch
    ) {
        if (global.params.is64bit)
            args.push_back("-m64");
        else
            args.push_back("-m32");
    }

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
    return executeToolAndWait(gcc, args, global.params.verbose);
}

//////////////////////////////////////////////////////////////////////////////

static int linkObjToBinaryWin(bool sharedLib)
{
    Logger::println("*** Linking executable ***");

    // find link.exe for linking
    std::string tool(getLink());

    // build arguments
    std::vector<std::string> args;

    args.push_back("/NOLOGO");

    // specify that the image will contain a table of safe exception handlers (32bit only)
    if (!global.params.is64bit)
        args.push_back("/SAFESEH");

    // mark executable to be compatible with Windows Data Execution Prevention feature
    args.push_back("/NXCOMPAT");

    // use address space layout randomization (ASLR) feature
    args.push_back("/DYNAMICBASE");

    // because of a LLVM bug
    // most of the bug is fixed in LLVM 3.4
#if LDC_LLVM_VER < 304
    if (global.params.symdebug)
        args.push_back("/LARGEADDRESSAWARE:NO");
    else
#endif
    args.push_back("/LARGEADDRESSAWARE");

    // output debug information
    if (global.params.symdebug)
    {
        args.push_back("/DEBUG");
    }

    // remove dead code and fold identical COMDATs
    if (opts::disableLinkerStripDead)
        args.push_back("/OPT:NOREF");
    else
    {
        args.push_back("/OPT:REF");
        args.push_back("/OPT:ICF");
    }

    // specify creation of DLL
    if (sharedLib)
        args.push_back("/DLL");

    // output filename
    std::string output = getOutputName(sharedLib);

    args.push_back("/OUT:" + output);

    // object files
    for (unsigned i = 0; i < global.params.objfiles->dim; i++)
    {
        const char *p = static_cast<const char *>(global.params.objfiles->data[i]);
        args.push_back(p);
    }

    // user libs
    for (unsigned i = 0; i < global.params.libfiles->dim; i++)
    {
        const char *p = static_cast<const char *>(global.params.libfiles->data[i]);
        args.push_back(p);
    }

    // set the global gExePath
    gExePath = output;
    //assert(gExePath.isValid());

    // create path to exe
    CreateDirectoryOnDisk(gExePath);

    // additional linker switches
    for (unsigned i = 0; i < global.params.linkswitches->dim; i++)
    {
        std::string str(static_cast<const char *>(global.params.linkswitches->data[i]));
        if (str.length() > 2)
        {
            // rewrite common -L and -l switches
            if (str[0] == '-' && str[1] == 'L')
                str = "/LIBPATH:" + str.substr(2);
            else if (str[0] == '-' && str[1] == 'l')
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
    return executeToolAndWait(tool, args, global.params.verbose);
}

//////////////////////////////////////////////////////////////////////////////

int linkObjToBinary(bool sharedLib)
{
    int status;
#if LDC_LLVM_VER >= 305
    if (global.params.targetTriple.isWindowsMSVCEnvironment())
#else
    if (global.params.targetTriple.getOS() == llvm::Triple::Win32)
#endif
        status = linkObjToBinaryWin(sharedLib);
    else
        status = linkObjToBinaryGcc(sharedLib);
    return status;
}

//////////////////////////////////////////////////////////////////////////////

void createStaticLibrary()
{
    Logger::println("*** Creating static library ***");

#if LDC_LLVM_VER >= 305
    const bool isTargetWindows = global.params.targetTriple.isWindowsMSVCEnvironment();
#else
    const bool isTargetWindows = global.params.targetTriple.getOS() == llvm::Triple::Win32;
#endif

    // find archiver
    std::string tool(isTargetWindows ? getLib() : getArchiver());

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
            libName = FileName::removeExt(static_cast<const char*>(global.params.objfiles->data[0]));
        else
            libName = "a.out";
    }
    std::string libExt = std::string(".") + global.lib_ext;
    if (!endsWith(libName, libExt))
    {
        if (!isTargetWindows)
        {
            libName = llvm::sys::path::parent_path(libName).str()
                    + "/lib"
                    + llvm::sys::path::filename(libName).str()
                    + libExt;
        }
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
        const char *p = static_cast<const char *>(global.params.objfiles->data[i]);
        args.push_back(p);
    }

    // create path to the library
    CreateDirectoryOnDisk(libName);

    // try to call archiver
    executeToolAndWait(tool, args, global.params.verbose);
}

//////////////////////////////////////////////////////////////////////////////

void deleteExecutable()
{
    if (!gExePath.empty())
    {
        //assert(gExePath.isValid());
        bool is_directory;
        assert(!(!llvm::sys::fs::is_directory(gExePath, is_directory) && is_directory));
#if LDC_LLVM_VER < 305
        bool Existed;
        llvm::sys::fs::remove(gExePath, Existed);
#else
        llvm::sys::fs::remove(gExePath);
#endif
    }
}

//////////////////////////////////////////////////////////////////////////////

int runExecutable()
{
    assert(!gExePath.empty());
    //assert(gExePath.isValid());

    // Run executable
    int status = executeToolAndWait(gExePath, opts::runargs, global.params.verbose);
    if (status < 0)
    {
#if defined(_MSC_VER) || defined(__MINGW32__)
        error(Loc(), "program received signal %d", -status);
#else
        error(Loc(), "program received signal %d (%s)", -status, strsignal(-status));
#endif
        return -status;
    }
    return status;
}
