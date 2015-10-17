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
#include "driver/exe_path.h"
#include "driver/tool.h"
#include "gen/llvm.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "gen/programs.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SystemUtils.h"

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
    // MIPS does not have -m32/-m64 but requires -mabi=.
    if (global.params.targetTriple.get64BitArchVariant().getArch() !=
        llvm::Triple::UnknownArch &&
        global.params.targetTriple.get32BitArchVariant().getArch() !=
        llvm::Triple::UnknownArch) {
        if (global.params.targetTriple.get64BitArchVariant().getArch() ==
            llvm::Triple::mips64 ||
            global.params.targetTriple.get64BitArchVariant().getArch() ==
            llvm::Triple::mips64el) {
            switch (getMipsABI())
            {
                case MipsABI::EABI:
                    args.push_back("-mabi=eabi");
                    break;
                case MipsABI::O32:
                    args.push_back("-mabi=32");
                    break;
                case MipsABI::N32:
                    args.push_back("-mabi=n32");
                    break;
                case MipsABI::N64:
                    args.push_back("-mabi=64");
                    break;
                case MipsABI::Unknown:
                    break;
            }
        }
        else {
            if (global.params.is64bit)
                args.push_back("-m64");
            else
                args.push_back("-m32");
        }
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

static bool setupMSVCEnvironment(std::string& tool, std::vector<std::string>& args)
{
    // if the VSINSTALLDIR environment variable is NOT set,
    // the environment is most likely not set up properly
    bool setup = !getenv("VSINSTALLDIR");

    if (setup)
    {
        // use a helper batch file to let MSVC set up the environment and
        // then invoke the tool
        args.push_back(tool); // tool is first arg for batch file
        tool = exe_path::prependBinDir(
            global.params.targetTriple.isArch64Bit() ? "amd64.bat" : "x86.bat");
    }
    else
        tool = getProgram(tool.c_str());

    return setup;
}

//////////////////////////////////////////////////////////////////////////////

// stupid copy from Program.inc, but inaccessible there, because of "static"

/// ArgNeedsQuotes - Check whether argument needs to be quoted when calling
/// CreateProcess.
static bool ArgNeedsQuotes(const char *Str) {
    return Str[0] == '\0' || strpbrk(Str, "\t \"&\'()*<>\\`^|") != 0;
}

/// CountPrecedingBackslashes - Returns the number of backslashes preceding Cur
/// in the C string Start.
static unsigned int CountPrecedingBackslashes(const char *Start,
    const char *Cur) {
    unsigned int Count = 0;
    --Cur;
    while (Cur >= Start && *Cur == '\\') {
        ++Count;
        --Cur;
    }
    return Count;
}

/// EscapePrecedingEscapes - Append a backslash to Dst for every backslash
/// preceding Cur in the Start string.  Assumes Dst has enough space.
static char *EscapePrecedingEscapes(char *Dst, const char *Start,
    const char *Cur) {
    unsigned PrecedingEscapes = CountPrecedingBackslashes(Start, Cur);
    while (PrecedingEscapes > 0) {
        *Dst++ = '\\';
        --PrecedingEscapes;
    }
    return Dst;
}

/// ArgLenWithQuotes - Check whether argument needs to be quoted when calling
/// CreateProcess and returns length of quoted arg with escaped quotes
static unsigned int ArgLenWithQuotes(const char *Str) {
    const char *Start = Str;
    bool Quoted = ArgNeedsQuotes(Str);
    unsigned int len = Quoted ? 2 : 0;

    while (*Str != '\0') {
        if (*Str == '\"') {
            // We need to add a backslash, but ensure that it isn't escaped.
            unsigned PrecedingEscapes = CountPrecedingBackslashes(Start, Str);
            len += PrecedingEscapes + 1;
        }
        // Note that we *don't* need to escape runs of backslashes that don't
        // precede a double quote!  See MSDN:
        // http://msdn.microsoft.com/en-us/library/17w5ykft%28v=vs.85%29.aspx

        ++len;
        ++Str;
    }

    if (Quoted) {
        // Make sure the closing quote doesn't get escaped by a trailing backslash.
        unsigned PrecedingEscapes = CountPrecedingBackslashes(Start, Str);
        len += PrecedingEscapes + 1;
    }

    return len;
}

static std::string flattenArgs(llvm::ArrayRef<std::string> args) {
    // First, determine the length of the command line.
    unsigned len = 0;
    for (size_t i = 0; i < args.size(); i++) {
        len += ArgLenWithQuotes(args[i].c_str()) + 1;
    }

    // Now build the command line.
    std::unique_ptr<char[]> command(new char[len + 1]);
    char *p = command.get();

    for (size_t i = 0; i < args.size(); i++) {
        const char *arg = args[i].c_str();
        const char *start = arg;

        bool needsQuoting = ArgNeedsQuotes(arg);
        if (needsQuoting)
            *p++ = '"';

        while (*arg != '\0') {
            if (*arg == '\"') {
                // Escape all preceding escapes (if any), and then escape the quote.
                p = EscapePrecedingEscapes(p, start, arg);
                *p++ = '\\';
            }

            *p++ = *arg++;
        }

        if (needsQuoting) {
            // Make sure our quote doesn't get escaped by a trailing backslash.
            p = EscapePrecedingEscapes(p, start, arg);
            *p++ = '"';
        }
        *p++ = ' ';
    }

    *p = 0;
    return command.get();
}

//////////////////////////////////////////////////////////////////////////////

static int linkObjToBinaryWin(bool sharedLib)
{
    Logger::println("*** Linking executable ***");

    std::string tool = "link.exe";

    // build arguments
    std::vector<std::string> args;

    const bool setupMSVC = setupMSVCEnvironment(tool, args);

    args.push_back("/NOLOGO");

    // specify that the image will contain a table of safe exception handlers (32bit only)
    if (!global.params.is64bit)
        args.push_back("/SAFESEH");

    // because of a LLVM bug, see LDC issue 442
    if (global.params.symdebug)
        args.push_back("/LARGEADDRESSAWARE:NO");
    else
        args.push_back("/LARGEADDRESSAWARE");

    // output debug information
    if (global.params.symdebug)
        args.push_back("/DEBUG");

    // enable Link-time Code Generation (aka. whole program optimization)
    if (global.params.optimize)
        args.push_back("/LTCG");

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
        std::string str = global.params.linkswitches->data[i];
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
    if (setupMSVC) ++I; // skip link.exe, the first arg for the batch file
    Stream logstr = Logger::cout();
    for (; I != E; ++I)
        if (!(*I).empty())
            logstr << "'" << *I << "'" << " ";
    logstr << "\n"; // FIXME where's flush ?

    llvm::SmallString<128> rsppath;
    llvm::ArrayRef<std::string> realArgs(&args[setupMSVC], args.size() - setupMSVC);
    std::string argsFlat = flattenArgs(realArgs);
    if (argsFlat.length() > 2047) {
        llvm::sys::fs::createUniqueFile("ldc-%%%%%%%.lnk", rsppath);
        llvm::sys::writeFileWithEncoding(rsppath, argsFlat, llvm::sys::WEM_CurrentCodePage);
        args.erase(args.begin() + setupMSVC, args.end());
        args.push_back(("@" + rsppath).str());
    }
    // try to call linker
    int rc = executeToolAndWait(tool, args, global.params.verbose);
    if (!rsppath.empty())
        llvm::sys::fs::remove(rsppath);
    return rc;
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
    std::string tool(isTargetWindows ? "lib.exe" : getArchiver());

    // build arguments
    std::vector<std::string> args;

    bool setupMSVC = isTargetWindows && setupMSVCEnvironment(tool, args);

    // ask ar to create a new library
    if (!isTargetWindows)
        args.push_back("rcs");

    // ask lib to be quiet
    if (isTargetWindows)
        args.push_back("/NOLOGO");

    // enable Link-time Code Generation (aka. whole program optimization)
    if (isTargetWindows && global.params.optimize)
        args.push_back("/LTCG");

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
// KN: The following lines were added to fix a test case failure (runnable/test13774.sh).
//     Root cause is that dmd handles it in this why.
//     As a side effect this change broke compiling with dub.
//    if (!FileName::absolute(libName.c_str()))
//        libName = FileName::combine(global.params.objdir, libName.c_str());
    if (llvm::sys::path::extension(libName).empty())
        libName.append(std::string(".") + global.lib_ext);
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

    llvm::SmallString<128> rsppath;
    if (isTargetWindows) {
        llvm::ArrayRef<std::string> realArgs(&args[setupMSVC], args.size() - setupMSVC);
        std::string argsFlat = flattenArgs(realArgs);
        if (argsFlat.length() > 2047) {
            llvm::sys::fs::createUniqueFile("ldc-%%%%%%%.lnk", rsppath);
            llvm::sys::writeFileWithEncoding(rsppath, argsFlat, llvm::sys::WEM_CurrentCodePage);
            args.erase(args.begin() + setupMSVC, args.end());
            args.push_back(("@" + rsppath).str());
        }
    }
    // try to call archiver
    executeToolAndWait(tool, args, global.params.verbose);
    if (!rsppath.empty())
        llvm::sys::fs::remove(rsppath);
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
