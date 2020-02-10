/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 * Entry point for DMD.
 *
 * This modules defines the entry point (main) for DMD, as well as related
 * utilities needed for arguments parsing, path manipulation, etc...
 * This file is not shared with other compilers which use the DMD front-end.
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/mars.d, _mars.d)
 * Documentation:  https://dlang.org/phobos/dmd_mars.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/mars.d
 */

module dmd.mars;

import core.stdc.ctype;
import core.stdc.limits;
import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.string;

import dmd.arraytypes;
import dmd.astcodegen;
import dmd.gluelayer;
import dmd.builtin;
import dmd.cond;
import dmd.console;
import dmd.compiler;
// IN_LLVM import dmd.dinifile;
import dmd.dinterpret;
import dmd.dmodule;
import dmd.doc;
import dmd.dsymbol;
import dmd.dsymbolsem;
import dmd.errors;
import dmd.expression;
import dmd.globals;
import dmd.hdrgen;
import dmd.id;
import dmd.identifier;
import dmd.inline;
import dmd.json;
version (IN_LLVM) {} else
version (NoMain) {} else
{
    import dmd.lib;
    import dmd.link;
}
import dmd.mtype;
import dmd.objc;
import dmd.root.array;
import dmd.root.file;
import dmd.root.filename;
import dmd.root.man;
import dmd.root.outbuffer;
// IN_LLVM import dmd.root.response;
import dmd.root.rmem;
import dmd.root.stringtable;
import dmd.semantic2;
import dmd.semantic3;
import dmd.target;
import dmd.utils;

version (IN_LLVM)
{
    import gen.semantic : extraLDCSpecificSemanticAnalysis;
    extern (C++):

    // in driver/main.cpp
    void registerPredefinedVersions();
    void codegenModules(ref Modules modules);
    // in driver/archiver.cpp
    int createStaticLibrary();
    // in driver/linker.cpp
    int linkObjToBinary();
    void deleteExeFile();
    int runProgram();
}
else
{

/**
 * Print DMD's logo on stdout
 */
private void logo()
{
    printf("DMD%llu D Compiler %.*s\n%.*s %.*s\n",
        cast(ulong)size_t.sizeof * 8,
        cast(int) global._version.length - 1, global._version.ptr,
        cast(int)global.copyright.length, global.copyright.ptr,
        cast(int)global.written.length, global.written.ptr
    );
}

/**
Print DMD's logo with more debug information and error-reporting pointers.

Params:
    stream = output stream to print the information on
*/
extern(C) void printInternalFailure(FILE* stream)
{
    fputs(("---\n" ~
    "ERROR: This is a compiler bug.\n" ~
            "Please report it via https://issues.dlang.org/enter_bug.cgi\n" ~
            "with, preferably, a reduced, reproducible example and the information below.\n" ~
    "DustMite (https://github.com/CyberShadow/DustMite/wiki) can help with the reduction.\n" ~
    "---\n").ptr, stream);
    stream.fprintf("DMD %.*s\n", cast(int) global._version.length - 1, global._version.ptr);
    stream.printPredefinedVersions;
    stream.printGlobalConfigs();
    fputs("---\n".ptr, stream);
}

/**
 * Print DMD's usage message on stdout
 */
private void usage()
{
    import dmd.cli : CLIUsage;
    logo();
    auto help = CLIUsage.usage;
    const inifileCanon = FileName.canonicalName(global.inifilename);
    printf("
Documentation: https://dlang.org/
Config file: %.*s
Usage:
  dmd [<option>...] <file>...
  dmd [<option>...] -run <file> [<arg>...]

Where:
  <file>           D source file
  <arg>            Argument to pass when running the resulting program

<option>:
  @<cmdfile>       read arguments from cmdfile
%.*s", cast(int)inifileCanon.length, inifileCanon.ptr, cast(int)help.length, &help[0]);
}

} // !IN_LLVM

/**
 * Remove generated .di files on error and exit
 */
private void removeHdrFilesAndFail(ref Param params, ref Modules modules)
{
    if (params.doHdrGeneration)
    {
        foreach (m; modules)
        {
            if (m.isHdrFile)
                continue;
            File.remove(m.hdrfile.toChars());
        }
    }

    fatal();
}

version (IN_LLVM) {} else
{

/**
 * DMD's real entry point
 *
 * Parses command line arguments and config file, open and read all
 * provided source file and do semantic analysis on them.
 *
 * Params:
 *   argc = Number of arguments passed via command line
 *   argv = Array of string arguments passed via command line
 *
 * Returns:
 *   Application return code
 */
version (NoMain) {} else
private int tryMain(size_t argc, const(char)** argv, ref Param params)
{
    Strings files;
    Strings libmodules;
    global._init();
    // Check for malformed input
    if (argc < 1 || !argv)
    {
    Largs:
        error(Loc.initial, "missing or null command line arguments");
        fatal();
    }
    // Convert argc/argv into arguments[] for easier handling
    Strings arguments = Strings(argc);
    for (size_t i = 0; i < argc; i++)
    {
        if (!argv[i])
            goto Largs;
        arguments[i] = argv[i];
    }
    if (!responseExpand(arguments)) // expand response files
        error(Loc.initial, "can't open response file");
    //for (size_t i = 0; i < arguments.dim; ++i) printf("arguments[%d] = '%s'\n", i, arguments[i]);
    files.reserve(arguments.dim - 1);
    // Set default values
    params.argv0 = arguments[0].toDString;

    // Temporary: Use 32 bits OMF as the default on Windows, for config parsing
    static if (TARGET.Windows)
    {
        params.is64bit = false;
        params.mscoff = false;
    }

    global.inifilename = parse_conf_arg(&arguments);
    if (global.inifilename)
    {
        // can be empty as in -conf=
        if (global.inifilename.length && !FileName.exists(global.inifilename))
            error(Loc.initial, "Config file '%.*s' does not exist.",
                  cast(int)global.inifilename.length, global.inifilename.ptr);
    }
    else
    {
        version (Windows)
        {
            global.inifilename = findConfFile(params.argv0, "sc.ini");
        }
        else version (Posix)
        {
            global.inifilename = findConfFile(params.argv0, "dmd.conf");
        }
        else
        {
            static assert(0, "fix this");
        }
    }
    // Read the configuration file
    const iniReadResult = global.inifilename.toCStringThen!(fn => File.read(fn.ptr));
    const inifileBuffer = iniReadResult.buffer.data;
    /* Need path of configuration file, for use in expanding @P macro
     */
    const(char)[] inifilepath = FileName.path(global.inifilename);
    Strings sections;
    StringTable!(char*) environment;
    environment._init(7);
    /* Read the [Environment] section, so we can later
     * pick up any DFLAGS settings.
     */
    sections.push("Environment");
    parseConfFile(environment, global.inifilename, inifilepath, inifileBuffer, &sections);

    const(char)* arch = params.is64bit ? "64" : "32"; // use default
    arch = parse_arch_arg(&arguments, arch);

    // parse architecture from DFLAGS read from [Environment] section
    {
        Strings dflags;
        getenv_setargv(readFromEnv(environment, "DFLAGS"), &dflags);
        environment.reset(7); // erase cached environment updates
        arch = parse_arch_arg(&dflags, arch);
    }

    bool is64bit = arch[0] == '6';

    version(Windows) // delete LIB entry in [Environment] (necessary for optlink) to allow inheriting environment for MS-COFF
        if (is64bit || strcmp(arch, "32mscoff") == 0)
            environment.update("LIB", 3).value = null;

    // read from DFLAGS in [Environment{arch}] section
    char[80] envsection = void;
    sprintf(envsection.ptr, "Environment%s", arch);
    sections.push(envsection.ptr);
    parseConfFile(environment, global.inifilename, inifilepath, inifileBuffer, &sections);
    getenv_setargv(readFromEnv(environment, "DFLAGS"), &arguments);
    updateRealEnvironment(environment);
    environment.reset(1); // don't need environment cache any more

    if (parseCommandLine(arguments, argc, params, files))
    {
        Loc loc;
        errorSupplemental(loc, "run `dmd` to print the compiler manual");
        errorSupplemental(loc, "run `dmd -man` to open browser on manual");
        return EXIT_FAILURE;
    }

    if (params.usage)
    {
        usage();
        return EXIT_SUCCESS;
    }

    if (params.logo)
    {
        logo();
        return EXIT_SUCCESS;
    }

    return mars_mainBody(params, files, libmodules);
}

} // !IN_LLVM

extern (C++) int mars_mainBody(ref Param params, ref Strings files, ref Strings libmodules)
{
    /*
    Prints a supplied usage text to the console and
    returns the exit code for the help usage page.

    Returns:
        `EXIT_SUCCESS` if no errors occurred, `EXIT_FAILURE` otherwise
    */
    static int printHelpUsage(string help)
    {
        printf("%.*s", cast(int)help.length, &help[0]);
        return global.errors ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    /*
    Generates code to check for all `params` whether any usage page
    has been requested.
    If so, the generated code will print the help page of the flag
    and return with an exit code.

    Params:
        params = parameters with `Usage` suffices in `params` for which
        their truthness should be checked.

    Returns: generated code for checking the usage pages of the provided `params`.
    */
    static string generateUsageChecks(string[] params)
    {
        string s;
        foreach (n; params)
        {
            s ~= q{
                if (params.}~n~q{Usage)
                    return printHelpUsage(CLIUsage.}~n~q{Usage);
            };
        }
        return s;
    }
    import dmd.cli : CLIUsage;
version (IN_LLVM)
{
    mixin(generateUsageChecks(["transition", "preview", "revert"]));
}
else
{
    mixin(generateUsageChecks(["mcpu", "transition", "check", "checkAction",
        "preview", "revert", "externStd"]));
}

version (IN_LLVM) {} else
{
    if (params.manual)
    {
        version (Windows)
        {
            browse("http://dlang.org/dmd-windows.html");
        }
        version (linux)
        {
            browse("http://dlang.org/dmd-linux.html");
        }
        version (OSX)
        {
            browse("http://dlang.org/dmd-osx.html");
        }
        version (FreeBSD)
        {
            browse("http://dlang.org/dmd-freebsd.html");
        }
        /*NOTE: No regular builds for openbsd/dragonflybsd (yet) */
        /*
        version (OpenBSD)
        {
            browse("http://dlang.org/dmd-openbsd.html");
        }
        version (DragonFlyBSD)
        {
            browse("http://dlang.org/dmd-dragonflybsd.html");
        }
        */
        return EXIT_SUCCESS;
    }
} // !IN_LLVM

    if (params.color)
        global.console = Console.create(core.stdc.stdio.stderr);

version (IN_LLVM) {} else
{
    setTarget(params);           // set target operating system
    setTargetCPU(params);
    if (params.is64bit != is64bit)
        error(Loc.initial, "the architecture must not be changed in the %s section of %.*s",
              envsection.ptr, cast(int)global.inifilename.length, global.inifilename.ptr);
}

    if (global.errors)
    {
        fatal();
    }
    if (files.dim == 0)
    {
        if (params.jsonFieldFlags)
        {
            generateJson(null);
            return EXIT_SUCCESS;
        }
version (IN_LLVM)
{
        error(Loc.initial, "No source files");
}
else
{
        usage();
}
        return EXIT_FAILURE;
    }

    reconcileCommands(params, files.dim);

    // Add in command line versions
    if (params.versionids)
        foreach (charz; *params.versionids)
            VersionCondition.addGlobalIdent(charz.toDString());
    if (params.debugids)
        foreach (charz; *params.debugids)
            DebugCondition.addGlobalIdent(charz.toDString());

version (IN_LLVM)
{
    registerPredefinedVersions();
}
else
{
    setTarget(params);

    // Predefined version identifiers
    addDefaultVersionIdentifiers(params);

    setDefaultLibrary();
}

    // Initialization
    Type._init();
    Id.initialize();
    Module._init();
    target._init(params);
    Expression._init();
    Objc._init();
    builtin_init();
    import dmd.filecache : FileCache;
    FileCache._init();

    version(CRuntime_Microsoft)
    {
        import dmd.root.longdouble;
        initFPU();
    }
    import dmd.root.ctfloat : CTFloat;
    CTFloat.initialize();

    if (params.verbose)
    {
        stdout.printPredefinedVersions();
version (IN_LLVM)
{
        // LDC prints binary/version/config before entering this function.
}
else
{
        stdout.printGlobalConfigs();
}
    }
    //printf("%d source files\n",files.dim);

    // Build import search path

    static Strings* buildPath(Strings* imppath)
    {
        Strings* result = null;
        if (imppath)
        {
            foreach (const path; *imppath)
            {
                Strings* a = FileName.splitPath(path);
                if (a)
                {
                    if (!result)
                        result = new Strings();
                    result.append(a);
                }
            }
        }
        return result;
    }

    if (params.mixinFile)
    {
        params.mixinOut = cast(OutBuffer*)Mem.check(calloc(1, OutBuffer.sizeof));
        atexit(&flushMixins); // see comment for flushMixins
    }
    scope(exit) flushMixins();
    global.path = buildPath(params.imppath);
    global.filePath = buildPath(params.fileImppath);

    if (params.addMain)
        files.push("__main.d");
    // Create Modules
    Modules modules = createModules(files, libmodules);
    // Read files
    // Start by "reading" the special files (__main.d, __stdin.d)
    foreach (m; modules)
    {
        if (params.addMain && m.srcfile.toString() == "__main.d")
        {
            auto data = arraydup("int main(){return 0;}\0\0"); // need 2 trailing nulls for sentinel
            m.srcBuffer = new FileBuffer(cast(ubyte[]) data[0 .. $-2]);
        }
        else if (m.srcfile.toString() == "__stdin.d")
        {
            auto buffer = readFromStdin();
            m.srcBuffer = new FileBuffer(buffer.extractSlice());
        }
    }

    foreach (m; modules)
    {
        m.read(Loc.initial);
    }

    // Parse files
    bool anydocfiles = false;
    size_t filecount = modules.dim;
    for (size_t filei = 0, modi = 0; filei < filecount; filei++, modi++)
    {
        Module m = modules[modi];
        if (params.verbose)
            message("parse     %s", m.toChars());
        if (!Module.rootModule)
            Module.rootModule = m;
        m.importedFrom = m; // m.isRoot() == true
version (IN_LLVM) {} else
{
        if (!params.oneobj || modi == 0 || m.isDocFile)
            m.deleteObjFile();
}

        m.parse();

version (IN_LLVM)
{
        // Finalize output filenames. Update if `-oq` was specified (only feasible after parsing).
        if (params.fullyQualifiedObjectFiles && m.md)
        {
            m.objfile = m.setOutfilename(params.objname, params.objdir, m.arg, FileName.ext(m.objfile.toString()));
            if (m.docfile)
                m.setDocfile();
            if (m.hdrfile)
                m.hdrfile = m.setOutfilename(params.hdrname, params.hdrdir, m.arg, global.hdr_ext);
        }

        // If `-run` is passed, the obj file is temporary and is removed after execution.
        // Make sure the name does not collide with other files from other processes by
        // creating a unique filename.
        if (params.run)
            m.makeObjectFilenameUnique();

        // Set object filename in params.objfiles.
        for (size_t j = 0; j < params.objfiles.dim; j++)
        {
            if (params.objfiles[j] == cast(const(char)*)m)
            {
                params.objfiles[j] = m.objfile.toChars();
                if (!m.isHdrFile && !m.isDocFile && params.obj)
                    m.checkAndAddOutputFile(m.objfile);
                break;
            }
        }

        if (!params.oneobj || modi == 0 || m.isDocFile)
            m.deleteObjFile();
} // IN_LLVM

        if (m.isHdrFile)
        {
            // Remove m's object file from list of object files
            for (size_t j = 0; j < params.objfiles.dim; j++)
            {
                if (m.objfile.toChars() == params.objfiles[j])
                {
                    params.objfiles.remove(j);
                    break;
                }
            }
            if (params.objfiles.dim == 0)
                params.link = false;
        }
        if (m.isDocFile)
        {
            anydocfiles = true;
            gendocfile(m);
            // Remove m from list of modules
            modules.remove(modi);
            modi--;
            // Remove m's object file from list of object files
            for (size_t j = 0; j < params.objfiles.dim; j++)
            {
                if (m.objfile.toChars() == params.objfiles[j])
                {
                    params.objfiles.remove(j);
                    break;
                }
            }
            if (params.objfiles.dim == 0)
                params.link = false;
        }
    }

    if (anydocfiles && modules.dim && (params.oneobj || params.objname))
    {
        error(Loc.initial, "conflicting Ddoc and obj generation options");
        fatal();
    }
    if (global.errors)
        fatal();

    if (params.doHdrGeneration)
    {
        /* Generate 'header' import files.
         * Since 'header' import files must be independent of command
         * line switches and what else is imported, they are generated
         * before any semantic analysis.
         */
        foreach (m; modules)
        {
            if (m.isHdrFile)
                continue;
            if (params.verbose)
                message("import    %s", m.toChars());
            genhdrfile(m);
        }
    }
    if (global.errors)
        removeHdrFilesAndFail(params, modules);

    // load all unconditional imports for better symbol resolving
    foreach (m; modules)
    {
        if (params.verbose)
            message("importall %s", m.toChars());
        m.importAll(null);
    }
    if (global.errors)
        removeHdrFilesAndFail(params, modules);

version (IN_LLVM) {} else
{
    backend_init();
}

    // Do semantic analysis
    foreach (m; modules)
    {
        if (params.verbose)
            message("semantic  %s", m.toChars());
        m.dsymbolSemantic(null);
    }
    //if (global.errors)
    //    fatal();
    Module.dprogress = 1;
    Module.runDeferredSemantic();
    if (Module.deferred.dim)
    {
        for (size_t i = 0; i < Module.deferred.dim; i++)
        {
            Dsymbol sd = Module.deferred[i];
            sd.error("unable to resolve forward reference in definition");
        }
        //fatal();
    }

    // Do pass 2 semantic analysis
    foreach (m; modules)
    {
        if (params.verbose)
            message("semantic2 %s", m.toChars());
        m.semantic2(null);
    }
    Module.runDeferredSemantic2();
    if (global.errors)
        removeHdrFilesAndFail(params, modules);

    // Do pass 3 semantic analysis
    foreach (m; modules)
    {
        if (params.verbose)
            message("semantic3 %s", m.toChars());
        m.semantic3(null);
    }
    if (includeImports)
    {
        // Note: DO NOT USE foreach here because Module.amodules.dim can
        //       change on each iteration of the loop
        for (size_t i = 0; i < compiledImports.dim; i++)
        {
            auto m = compiledImports[i];
            assert(m.isRoot);
            if (params.verbose)
                message("semantic3 %s", m.toChars());
            m.semantic3(null);
            modules.push(m);
        }
    }
    Module.runDeferredSemantic3();
    if (global.errors)
        removeHdrFilesAndFail(params, modules);

version (IN_LLVM)
{
    extraLDCSpecificSemanticAnalysis(modules);
}
else
{
    // Scan for functions to inline
    if (params.useInline)
    {
        foreach (m; modules)
        {
            if (params.verbose)
                message("inline scan %s", m.toChars());
            inlineScanModule(m);
        }
    }
}
    // Do not attempt to generate output files if errors or warnings occurred
    if (global.errors || global.warnings)
        removeHdrFilesAndFail(params, modules);

    // inlineScan incrementally run semantic3 of each expanded functions.
    // So deps file generation should be moved after the inlining stage.
    if (OutBuffer* ob = params.moduleDeps)
    {
        foreach (i; 1 .. modules[0].aimports.dim)
            semantic3OnDependencies(modules[0].aimports[i]);

        const data = (*ob)[];
        if (params.moduleDepsFile)
        {
            writeFile(Loc.initial, params.moduleDepsFile, data);
version (IN_LLVM)
{
            // fix LDC issue #1625
            params.moduleDeps = null;
            params.moduleDepsFile = null;
}
        }
        else
            printf("%.*s", cast(int)data.length, data.ptr);
    }

    printCtfePerformanceStats();

version (IN_LLVM) {} else
{
    Library library = null;
    if (params.lib)
    {
        if (params.objfiles.dim == 0)
        {
            error(Loc.initial, "no input files");
            return EXIT_FAILURE;
        }
        library = Library.factory();
        library.setFilename(params.objdir, params.libname);
        // Add input object and input library files to output library
        foreach (p; libmodules)
            library.addObject(p, null);
    }
}
    // Generate output files
    if (params.doJsonGeneration)
    {
        generateJson(&modules);
    }
    if (!global.errors && params.doDocComments)
    {
        foreach (m; modules)
        {
            gendocfile(m);
        }
    }
    if (params.vcg_ast)
    {
        import dmd.hdrgen;
        foreach (mod; modules)
        {
            auto buf = OutBuffer();
            buf.doindent = 1;
            moduleToBuffer(&buf, mod);

            // write the output to $(filename).cg
            auto cgFilename = FileName.addExt(mod.srcfile.toString(), "cg");
            File.write(cgFilename.ptr, buf[]);
        }
    }
version (IN_LLVM)
{
    import core.memory : GC;

    static if (__traits(compiles, GC.stats))
    {
        if (global.params.verbose)
        {
            static int toMB(ulong size) { return cast(int) (size / 1048576.0 + 0.5); }

            const stats = GC.stats;
            const used = toMB(stats.usedSize);
            const free = toMB(stats.freeSize);
            const total = toMB(stats.usedSize + stats.freeSize);
            message("GC stats  %dM used, %dM free, %dM total", used, free, total);
        }
    }

    codegenModules(modules);
}
else
{
    if (!params.obj)
    {
    }
    else if (params.oneobj)
    {
        Module firstm;    // first module we generate code for
        foreach (m; modules)
        {
            if (m.isHdrFile)
                continue;
            if (!firstm)
            {
                firstm = m;
                obj_start(m.srcfile.toChars());
            }
            if (params.verbose)
                message("code      %s", m.toChars());
            genObjFile(m, false);
        }
        if (!global.errors && firstm)
        {
            obj_end(library, firstm.objfile.toChars());
        }
    }
    else
    {
        foreach (m; modules)
        {
            if (m.isHdrFile)
                continue;
            if (params.verbose)
                message("code      %s", m.toChars());
            obj_start(m.srcfile.toChars());
            genObjFile(m, params.multiobj);
            obj_end(library, m.objfile.toChars());
            obj_write_deferred(library);
            if (global.errors && !params.lib)
                m.deleteObjFile();
        }
    }
    if (params.lib && !global.errors)
        library.write();
    backend_term();
} // !IN_LLVM
    if (global.errors)
        fatal();
    int status = EXIT_SUCCESS;
    if (!params.objfiles.dim)
    {
        if (params.link)
            error(Loc.initial, "no object files to link");
        if (IN_LLVM && !params.link && params.lib)
            error(Loc.initial, "no object files");
    }
    else
    {
version (IN_LLVM)
{
        if (params.link)
            status = linkObjToBinary();
        else if (params.lib)
            status = createStaticLibrary();

        if (status == EXIT_SUCCESS &&
            (params.cleanupObjectFiles || params.run))
        {
            for (size_t i = 0; i < modules.dim; i++)
            {
                modules[i].deleteObjFile();
                if (params.oneobj)
                    break;
            }
        }
}
else // !IN_LLVM
{
        if (params.link)
            status = runLINK();
}
        if (params.run)
        {
            if (!status)
            {
                status = runProgram();
                /* Delete .obj files and .exe file
                 */
version (IN_LLVM) {} else
{
                foreach (m; modules)
                {
                    m.deleteObjFile();
                    if (params.oneobj)
                        break;
                }
}
                params.exefile.toCStringThen!(ef => File.remove(ef.ptr));
            }
        }
    }
    if (global.errors || global.warnings)
        removeHdrFilesAndFail(params, modules);

    return status;
}

private FileBuffer readFromStdin()
{
    enum bufIncrement = 128 * 1024;
    size_t pos = 0;
    size_t sz = bufIncrement;

    ubyte* buffer = null;
    for (;;)
    {
        buffer = cast(ubyte*)mem.xrealloc(buffer, sz + 2); // +2 for sentinel

        // Fill up buffer
        do
        {
            assert(sz > pos);
            size_t rlen = fread(buffer + pos, 1, sz - pos, stdin);
            pos += rlen;
            if (ferror(stdin))
            {
                import core.stdc.errno;
                error(Loc.initial, "cannot read from stdin, errno = %d", errno);
                fatal();
            }
            if (feof(stdin))
            {
                // We're done
                assert(pos < sz + 2);
                buffer[pos] = '\0';
                buffer[pos + 1] = '\0';
                return FileBuffer(buffer[0 .. pos]);
            }
        } while (pos < sz);

        // Buffer full, expand
        sz += bufIncrement;
    }

    assert(0);
}

extern (C++) void generateJson(Modules* modules)
{
    OutBuffer buf;
    json_generate(&buf, modules);

    // Write buf to file
    const(char)[] name = global.params.jsonfilename;
    if (name == "-")
    {
        // Write to stdout; assume it succeeds
        size_t n = fwrite(buf[].ptr, 1, buf.length, stdout);
        assert(n == buf.length); // keep gcc happy about return values
    }
    else
    {
        /* The filename generation code here should be harmonized with Module.setOutfilename()
         */
        const(char)[] jsonfilename;
        if (name)
        {
            jsonfilename = FileName.defaultExt(name, global.json_ext);
        }
        else
        {
            if (global.params.objfiles.dim == 0)
            {
                error(Loc.initial, "cannot determine JSON filename, use `-Xf=<file>` or provide a source file");
                fatal();
            }
            // Generate json file name from first obj name
            const(char)[] n = global.params.objfiles[0].toDString;
            n = FileName.name(n);
            //if (!FileName::absolute(name))
            //    name = FileName::combine(dir, name);
            jsonfilename = FileName.forceExt(n, global.json_ext);
        }
        writeFile(Loc.initial, jsonfilename, buf[]);
    }
}


version (IN_LLVM) {} else
{

version (NoMain) {} else
{
    // in druntime:
    alias MainFunc = extern(C) int function(char[][] args);
    extern (C) int _d_run_main(int argc, char** argv, MainFunc dMain);

    // When using a C main, host DMD may not link against host druntime by default.
    version (DigitalMars)
    {
        version (Win64)
        {
            pragma(lib, "phobos64");
        }
        else version (Win32)
        {
            version (CRuntime_Microsoft)
            {
                pragma(lib, "phobos32mscoff");
            }
            else
            {
                pragma(lib, "phobos");
            }
        }
    }

    /**
     * DMD's entry point, C main.
     *
     * Without `-lowmem`, we need to switch to the bump-pointer allocation scheme
     * right from the start, before any module ctors are run, so we need this hook
     * before druntime is initialized and `_Dmain` is called.
     *
     * Returns:
     *   Return code of the application
     */
    extern (C) int main(int argc, char** argv)
    {
        static if (isGCAvailable)
        {
            bool lowmem = false;
            foreach (i; 1 .. argc)
            {
                if (strcmp(argv[i], "-lowmem") == 0)
                {
                    lowmem = true;
                    break;
                }
            }
            if (!lowmem)
                mem.disableGC();
        }

        // initialize druntime and call _Dmain() below
        return _d_run_main(argc, argv, &_Dmain);
    }

    /**
     * Manual D main (for druntime initialization), which forwards to `tryMain`.
     *
     * Returns:
     *   Return code of the application
     */
    extern (C) int _Dmain(char[][])
    {
        import core.memory;
        import core.runtime;

        // Older host druntime versions need druntime to be initialized before
        // disabling the GC, so we cannot disable it in C main above.
        static if (isGCAvailable)
        {
            if (!mem.isGCEnabled)
                GC.disable();
        }
        else
        {
            GC.disable();
        }

        version(D_Coverage)
        {
            // for now we need to manually set the source path
            string dirName(string path, char separator)
            {
                for (size_t i = path.length - 1; i > 0; i--)
                {
                    if (path[i] == separator)
                        return path[0..i];
                }
                return path;
            }
            version (Windows)
                enum sourcePath = dirName(dirName(__FILE_FULL_PATH__, '\\'), '\\');
            else
                enum sourcePath = dirName(dirName(__FILE_FULL_PATH__, '/'), '/');

            dmd_coverSourcePath(sourcePath);
            dmd_coverDestPath(sourcePath);
            dmd_coverSetMerge(true);
        }

        scope(failure) stderr.printInternalFailure;

        auto args = Runtime.cArgs();
        return tryMain(args.argc, cast(const(char)**)args.argv, global.params);
    }
} // !NoMain

/**
 * Parses an environment variable containing command-line flags
 * and append them to `args`.
 *
 * This function is used to read the content of DFLAGS.
 * Flags are separated based on spaces and tabs.
 *
 * Params:
 *   envvalue = The content of an environment variable
 *   args     = Array to append the flags to, if any.
 */
void getenv_setargv(const(char)* envvalue, Strings* args)
{
    if (!envvalue)
        return;

    char* env = mem.xstrdup(envvalue); // create our own writable copy
    //printf("env = '%s'\n", env);
    while (1)
    {
        switch (*env)
        {
        case ' ':
        case '\t':
            env++;
            break;

        case 0:
            return;

        default:
        {
            args.push(env); // append
            auto p = env;
            auto slash = 0;
            bool instring = false;
            while (1)
            {
                auto c = *env++;
                switch (c)
                {
                case '"':
                    p -= (slash >> 1);
                    if (slash & 1)
                    {
                        p--;
                        goto default;
                    }
                    instring ^= true;
                    slash = 0;
                    continue;

                case ' ':
                case '\t':
                    if (instring)
                        goto default;
                    *p = 0;
                    //if (wildcard)
                    //    wildcardexpand();     // not implemented
                    break;

                case '\\':
                    slash++;
                    *p++ = c;
                    continue;

                case 0:
                    *p = 0;
                    //if (wildcard)
                    //    wildcardexpand();     // not implemented
                    return;

                default:
                    slash = 0;
                    *p++ = c;
                    continue;
                }
                break;
            }
            break;
        }
        }
    }
}

/**
 * Parse command line arguments for the last instance of -m32, -m64 or -m32mscoff
 * to detect the desired architecture.
 *
 * Params:
 *   args = Command line arguments
 *   arch = Default value to use for architecture.
 *          Should be "32" or "64"
 *
 * Returns:
 *   "32", "64" or "32mscoff" if the "-m32", "-m64", "-m32mscoff" flags were passed,
 *   respectively. If they weren't, return `arch`.
 */
const(char)* parse_arch_arg(Strings* args, const(char)* arch)
{
    foreach (const p; *args)
    {
        if (p[0] == '-')
        {
            if (strcmp(p + 1, "m32") == 0 || strcmp(p + 1, "m32mscoff") == 0 || strcmp(p + 1, "m64") == 0)
                arch = p + 2;
            else if (strcmp(p + 1, "run") == 0)
                break;
        }
    }
    return arch;
}


/**
 * Parse command line arguments for the last instance of -conf=path.
 *
 * Params:
 *   args = Command line arguments
 *
 * Returns:
 *   The 'path' in -conf=path, which is the path to the config file to use
 */
const(char)[] parse_conf_arg(Strings* args)
{
    const(char)[] conf;
    foreach (const p; *args)
    {
        const(char)[] arg = p.toDString;
        if (arg.length && arg[0] == '-')
        {
            if(arg.length >= 6 && arg[1 .. 6] == "conf="){
                conf = arg[6 .. $];
            }
            else if (arg[1 .. $] == "run")
                break;
        }
    }
    return conf;
}


/**
 * Set the default and debug libraries to link against, if not already set
 *
 * Must be called after argument parsing is done, as it won't
 * override any value.
 * Note that if `-defaultlib=` or `-debuglib=` was used,
 * we don't override that either.
 */
private void setDefaultLibrary()
{
    if (global.params.defaultlibname is null)
    {
        static if (TARGET.Windows)
        {
            if (global.params.is64bit)
                global.params.defaultlibname = "phobos64";
            else if (global.params.mscoff)
                global.params.defaultlibname = "phobos32mscoff";
            else
                global.params.defaultlibname = "phobos";
        }
        else static if (TARGET.Linux || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
        {
            global.params.defaultlibname = "libphobos2.a";
        }
        else static if (TARGET.OSX)
        {
            global.params.defaultlibname = "phobos2";
        }
        else
        {
            static assert(0, "fix this");
        }
    }
    else if (!global.params.defaultlibname.length)  // if `-defaultlib=` (i.e. an empty defaultlib)
        global.params.defaultlibname = null;

    if (global.params.debuglibname is null)
        global.params.debuglibname = global.params.defaultlibname;
}

/*************************************
 * Set the `is` target fields of `params` according
 * to the TARGET value.
 * Params:
 *      params = where the `is` fields are
 */
void setTarget(ref Param params)
{
    static if (TARGET.Windows)
        params.isWindows = true;
    else static if (TARGET.Linux)
        params.isLinux = true;
    else static if (TARGET.OSX)
        params.isOSX = true;
    else static if (TARGET.FreeBSD)
        params.isFreeBSD = true;
    else static if (TARGET.OpenBSD)
        params.isOpenBSD = true;
    else static if (TARGET.Solaris)
        params.isSolaris = true;
    else static if (TARGET.DragonFlyBSD)
        params.isDragonFlyBSD = true;
    else
        static assert(0, "unknown TARGET");
}

/**
 * Add default `version` identifier for dmd, and set the
 * target platform in `params`.
 * https://dlang.org/spec/version.html#predefined-versions
 *
 * Needs to be run after all arguments parsing (command line, DFLAGS environment
 * variable and config file) in order to add final flags (such as `X86_64` or
 * the `CRuntime` used).
 *
 * Params:
 *      params = which target to compile for (set by `setTarget()`)
 */
void addDefaultVersionIdentifiers(const ref Param params)
{
    VersionCondition.addPredefinedGlobalIdent("DigitalMars");
    if (params.isWindows)
    {
        VersionCondition.addPredefinedGlobalIdent("Windows");
        if (global.params.mscoff)
        {
            VersionCondition.addPredefinedGlobalIdent("CRuntime_Microsoft");
            VersionCondition.addPredefinedGlobalIdent("CppRuntime_Microsoft");
        }
        else
        {
            VersionCondition.addPredefinedGlobalIdent("CRuntime_DigitalMars");
            VersionCondition.addPredefinedGlobalIdent("CppRuntime_DigitalMars");
        }
    }
    else if (params.isLinux)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("linux");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        // Note: This should be done with a target triplet, to support cross compilation.
        // However DMD currently does not support it, so this is a simple
        // fix to make DMD compile on Musl-based systems such as Alpine.
        // See https://github.com/dlang/dmd/pull/8020
        // And https://wiki.osdev.org/Target_Triplet
        version (CRuntime_Musl)
            VersionCondition.addPredefinedGlobalIdent("CRuntime_Musl");
        else
            VersionCondition.addPredefinedGlobalIdent("CRuntime_Glibc");
        VersionCondition.addPredefinedGlobalIdent("CppRuntime_Gcc");
    }
    else if (params.isOSX)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("OSX");
        VersionCondition.addPredefinedGlobalIdent("CppRuntime_Clang");
        // For legacy compatibility
        VersionCondition.addPredefinedGlobalIdent("darwin");
    }
    else if (params.isFreeBSD)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("FreeBSD");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        VersionCondition.addPredefinedGlobalIdent("CppRuntime_Clang");
    }
    else if (params.isOpenBSD)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("OpenBSD");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        VersionCondition.addPredefinedGlobalIdent("CppRuntime_Gcc");
    }
    else if (params.isDragonFlyBSD)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("DragonFlyBSD");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        VersionCondition.addPredefinedGlobalIdent("CppRuntime_Gcc");
    }
    else if (params.isSolaris)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("Solaris");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        VersionCondition.addPredefinedGlobalIdent("CppRuntime_Sun");
    }
    else
    {
        assert(0);
    }
    VersionCondition.addPredefinedGlobalIdent("LittleEndian");
    VersionCondition.addPredefinedGlobalIdent("D_Version2");
    VersionCondition.addPredefinedGlobalIdent("all");

    if (params.cpu >= CPU.sse2)
    {
        VersionCondition.addPredefinedGlobalIdent("D_SIMD");
        if (params.cpu >= CPU.avx)
            VersionCondition.addPredefinedGlobalIdent("D_AVX");
        if (params.cpu >= CPU.avx2)
            VersionCondition.addPredefinedGlobalIdent("D_AVX2");
    }

    if (params.is64bit)
    {
        VersionCondition.addPredefinedGlobalIdent("D_InlineAsm_X86_64");
        VersionCondition.addPredefinedGlobalIdent("X86_64");
        if (params.isWindows)
        {
            VersionCondition.addPredefinedGlobalIdent("Win64");
        }
    }
    else
    {
        VersionCondition.addPredefinedGlobalIdent("D_InlineAsm"); //legacy
        VersionCondition.addPredefinedGlobalIdent("D_InlineAsm_X86");
        VersionCondition.addPredefinedGlobalIdent("X86");
        if (params.isWindows)
        {
            VersionCondition.addPredefinedGlobalIdent("Win32");
        }
    }

    if (params.isLP64)
        VersionCondition.addPredefinedGlobalIdent("D_LP64");
    if (params.doDocComments)
        VersionCondition.addPredefinedGlobalIdent("D_Ddoc");
    if (params.cov)
        VersionCondition.addPredefinedGlobalIdent("D_Coverage");
    if (params.pic != PIC.fixed)
        VersionCondition.addPredefinedGlobalIdent(params.pic == PIC.pic ? "D_PIC" : "D_PIE");
    if (params.useUnitTests)
        VersionCondition.addPredefinedGlobalIdent("unittest");
    if (params.useAssert == CHECKENABLE.on)
        VersionCondition.addPredefinedGlobalIdent("assert");
    if (params.useArrayBounds == CHECKENABLE.off)
        VersionCondition.addPredefinedGlobalIdent("D_NoBoundsChecks");
    if (params.betterC)
    {
        VersionCondition.addPredefinedGlobalIdent("D_BetterC");
    }
    else
    {
        VersionCondition.addPredefinedGlobalIdent("D_ModuleInfo");
        VersionCondition.addPredefinedGlobalIdent("D_Exceptions");
        VersionCondition.addPredefinedGlobalIdent("D_TypeInfo");
    }

    VersionCondition.addPredefinedGlobalIdent("D_HardFloat");
}

} // !IN_LLVM

private void printPredefinedVersions(FILE* stream)
{
    if (global.versionids)
    {
        OutBuffer buf;
        foreach (const str; *global.versionids)
        {
            buf.writeByte(' ');
            buf.writestring(str.toChars());
        }
        stream.fprintf("predefs  %s\n", buf.peekChars());
    }
}

version (IN_LLVM) {} else
{

extern(C) void printGlobalConfigs(FILE* stream)
{
    stream.fprintf("binary    %.*s\n", cast(int)global.params.argv0.length, global.params.argv0.ptr);
    stream.fprintf("version   %.*s\n", cast(int) global._version.length - 1, global._version.ptr);
    const iniOutput = global.inifilename ? global.inifilename : "(none)";
    stream.fprintf("config    %.*s\n", cast(int)iniOutput.length, iniOutput.ptr);
    // Print DFLAGS environment variable
    {
        StringTable!(char*) environment;
        environment._init(0);
        Strings dflags;
        getenv_setargv(readFromEnv(environment, "DFLAGS"), &dflags);
        environment.reset(1);
        OutBuffer buf;
        foreach (flag; dflags[])
        {
            bool needsQuoting;
            foreach (c; flag.toDString())
            {
                if (!(isalnum(c) || c == '_'))
                {
                    needsQuoting = true;
                    break;
                }
            }

            if (flag.strchr(' '))
                buf.printf("'%s' ", flag);
            else
                buf.printf("%s ", flag);
        }

        auto res = buf[] ? buf[][0 .. $ - 1] : "(none)";
        stream.fprintf("DFLAGS    %.*s\n", cast(int)res.length, res.ptr);
    }
}

/****************************************
 * Determine the instruction set to be used, i.e. set params.cpu
 * by combining the command line setting of
 * params.cpu with the target operating system.
 * Params:
 *      params = parameters set by command line switch
 */

private void setTargetCPU(ref Param params)
{
    if (target.isXmmSupported())
    {
        switch (params.cpu)
        {
            case CPU.baseline:
                params.cpu = CPU.sse2;
                break;

            case CPU.native:
            {
                import core.cpuid;
                params.cpu = core.cpuid.avx2 ? CPU.avx2 :
                             core.cpuid.avx  ? CPU.avx  :
                                               CPU.sse2;
                break;
            }

            default:
                break;
        }
    }
    else
        params.cpu = CPU.x87;   // cannot support other instruction sets
}

} // !IN_LLVM

/**************************************
 * we want to write the mixin expansion file also on error, but there
 * are too many ways to terminate dmd (e.g. fatal() which calls exit(EXIT_FAILURE)),
 * so we can't rely on scope(exit) ... in tryMain() actually being executed
 * so we add atexit(&flushMixins); for those fatal exits (with the GC still valid)
 */
extern(C) void flushMixins()
{
    if (!global.params.mixinOut)
        return;

    assert(global.params.mixinFile);
    File.write(global.params.mixinFile, (*global.params.mixinOut)[]);

    global.params.mixinOut.destroy();
    global.params.mixinOut = null;
}

version (IN_LLVM)
{
    import dmd.cli : Usage;

    private bool parseCLIOption(string groupName, Usage.Feature[] features)(ref Param params, const(char)* name)
    {
        string generateCases()
        {
            string buf = `case "all":`;
            foreach (t; features)
                buf ~= `params.`~t.paramName~` = true;`;
            buf ~= "break;";

            foreach (t; features)
                buf ~= `case "`~t.name~`": params.`~t.paramName~` = true; return true;`;

            return buf;
        }

        switch (name[0 .. strlen(name)])
        {
            mixin(generateCases());
            case "?":
            case "h":
            case "help":
                mixin(`params.`~groupName~`Usage = true;`);
                return true;
            default:
                break;
        }

        return false;
    }

    extern(C++) void parseTransitionOption(ref Param params, const(char)* name)
    {
        if (parseCLIOption!("transition", Usage.transitions)(params, name))
            return;

        // undocumented legacy -transition flags (before 2.085)
        const dname = name[0 .. strlen(name)];
        switch (dname)
        {
            case "3449":
                params.vfield = true;
                break;
            case "10378":
            case "import":
                params.bug10378 = true;
                break;
            case "14246":
            case "dtorfields":
                params.dtorFields = true;
                break;
            case "14488":
                params.vcomplex = true;
                break;
            case "16997":
            case "intpromote":
                params.fix16997 = true;
                break;
            case "markdown":
                params.markdown = true;
                break;
            default:
                error(Loc.initial, "Transition `%s` is invalid", name);
                params.transitionUsage = true;
                break;
        }
    }

    extern(C++) void parsePreviewOption(ref Param params, const(char)* name)
    {
        if (!parseCLIOption!("preview", Usage.previews)(params, name))
        {
            error(Loc.initial, "Preview `%s` is invalid", name);
            params.previewUsage = true;
        }
    }

    extern(C++) void parseRevertOption(ref Param params, const(char)* name)
    {
        if (!parseCLIOption!("revert", Usage.reverts)(params, name))
        {
            error(Loc.initial, "Revert `%s` is invalid", name);
            params.revertUsage = true;
        }
    }
}
else // !IN_LLVM
{

/****************************************************
 * Parse command line arguments.
 *
 * Prints message(s) if there are errors.
 *
 * Params:
 *      arguments = command line arguments
 *      argc = argument count
 *      params = set to result of parsing `arguments`
 *      files = set to files pulled from `arguments`
 * Returns:
 *      true if errors in command line
 */

bool parseCommandLine(const ref Strings arguments, const size_t argc, ref Param params, ref Strings files)
{
    bool errors;

    void error(const(char)* format, const(char*) arg = null)
    {
        dmd.errors.error(Loc.initial, format, arg);
        errors = true;
    }

    /************************************
     * Convert string to integer.
     * Params:
     *  p = pointer to start of string digits, ending with 0
     *  max = max allowable value (inclusive)
     * Returns:
     *  uint.max on error, otherwise converted integer
     */
    static pure uint parseDigits(const(char)*p, const uint max)
    {
        uint value;
        bool overflow;
        for (uint d; (d = uint(*p) - uint('0')) < 10; ++p)
        {
            import core.checkedint : mulu, addu;
            value = mulu(value, 10, overflow);
            value = addu(value, d, overflow);
        }
        return (overflow || value > max || *p) ? uint.max : value;
    }

    /********************************
     * Params:
     *  p = 0 terminated string
     *  s = string
     * Returns:
     *  true if `p` starts with `s`
     */
    static pure bool startsWith(const(char)* p, string s)
    {
        foreach (const c; s)
        {
            if (c != *p)
                return false;
            ++p;
        }
        return true;
    }

    /**
     * Print an error messsage about an invalid switch.
     * If an optional supplemental message has been provided,
     * it will be printed too.
     *
     * Params:
     *  p = 0 terminated string
     *  availableOptions = supplemental help message listing the available options
     */
    void errorInvalidSwitch(const(char)* p, string availableOptions = null)
    {
        error("Switch `%s` is invalid", p);
        if (availableOptions !is null)
            errorSupplemental(Loc.initial, "%.*s", cast(int)availableOptions.length, availableOptions.ptr);
    }

    enum CheckOptions { success, error, help }

    /*
    Checks whether the CLI options contains a valid argument or a help argument.
    If a help argument has been used, it will set the `usageFlag`.

    Params:
        p = 0 terminated string
        usageFlag = parameter for the usage help page to set (by `ref`)
        missingMsg = error message to use when no argument has been provided

    Returns:
        `success` if a valid argument has been passed and it's not a help page
        `error` if an error occurred (e.g. `-foobar`)
        `help` if a help page has been request (e.g. `-flag` or `-flag=h`)
    */
    CheckOptions checkOptions(const(char)* p, ref bool usageFlag, string missingMsg)
    {
        // Checks whether a flag has no options (e.g. -foo or -foo=)
        if (*p == 0 || *p == '=' && !p[1])
        {
            .error(Loc.initial, "%.*s", cast(int)missingMsg.length, missingMsg.ptr);
            errors = true;
            usageFlag = true;
            return CheckOptions.help;
        }
        if (*p != '=')
            return CheckOptions.error;
        p++;
        /* Checks whether the option pointer supplied is a request
           for the help page, e.g. -foo=j */
        if (((*p == 'h' || *p == '?') && !p[1]) || // -flag=h || -flag=?
            strcmp(p, "help") == 0)
        {
            usageFlag = true;
            return CheckOptions.help;
        }
        return CheckOptions.success;
    }

    static string checkOptionsMixin(string usageFlag, string missingMsg)
    {
        return q{
            final switch (checkOptions(p + len - 1, params.}~usageFlag~","~
                          `"`~missingMsg~`"`~q{))
            {
                case CheckOptions.error:
                    goto Lerror;
                case CheckOptions.help:
                    return false;
                case CheckOptions.success:
                    break;
            }
        };
    }

    import dmd.cli : Usage;
    bool parseCLIOption(string name, Usage.Feature[] features)(ref Param params, const(char)* p)
    {
        // Parse:
        //      -<name>=<feature>
        const ps = p + name.length + 1;
        if (Identifier.isValidIdentifier(ps + 1))
        {
            string generateTransitionsText()
            {
                import dmd.cli : Usage;
                string buf = `case "all":`;
                foreach (t; features)
                {
                    if (t.deprecated_)
                        continue;

                    buf ~= `params.`~t.paramName~` = true;`;
                }
                buf ~= "break;\n";

                foreach (t; features)
                {
                    buf ~= `case "`~t.name~`":`;
                    if (t.deprecated_)
                        buf ~= "deprecation(Loc.initial, \"`-"~name~"="~t.name~"` no longer has any effect.\"); ";
                    buf ~= `params.`~t.paramName~` = true; return true;`;
                }
                return buf;
            }
            const ident = ps + 1;
            switch (ident.toDString())
            {
                mixin(generateTransitionsText());
            default:
                return false;
            }
        }
        return false;
    }

    version (none)
    {
        for (size_t i = 0; i < arguments.dim; i++)
        {
            printf("arguments[%d] = '%s'\n", i, arguments[i]);
        }
    }
    for (size_t i = 1; i < arguments.dim; i++)
    {
        const(char)* p = arguments[i];
        const(char)[] arg = p.toDString();
        if (*p != '-')
        {
            static if (TARGET.Windows)
            {
                const ext = FileName.ext(arg);
                if (ext.length && FileName.equals(ext, "exe"))
                {
                    params.objname = arg;
                    continue;
                }
                if (arg == "/?")
                {
                    params.usage = true;
                    return false;
                }
            }
            files.push(p);
            continue;
        }

        if (arg == "-allinst")               // https://dlang.org/dmd.html#switch-allinst
            params.allInst = true;
        else if (arg == "-de")               // https://dlang.org/dmd.html#switch-de
            params.useDeprecated = DiagnosticReporting.error;
        else if (arg == "-d")                // https://dlang.org/dmd.html#switch-d
            params.useDeprecated = DiagnosticReporting.off;
        else if (arg == "-dw")               // https://dlang.org/dmd.html#switch-dw
            params.useDeprecated = DiagnosticReporting.inform;
        else if (arg == "-c")                // https://dlang.org/dmd.html#switch-c
            params.link = false;
        else if (startsWith(p + 1, "checkaction")) // https://dlang.org/dmd.html#switch-checkaction
        {
            /* Parse:
             *    -checkaction=D|C|halt|context
             */
            enum len = "-checkaction=".length;
            mixin(checkOptionsMixin("checkActionUsage",
                "`-check=<behavior>` requires a behavior"));
            if (strcmp(p + len, "D") == 0)
                params.checkAction = CHECKACTION.D;
            else if (strcmp(p + len, "C") == 0)
                params.checkAction = CHECKACTION.C;
            else if (strcmp(p + len, "halt") == 0)
                params.checkAction = CHECKACTION.halt;
            else if (strcmp(p + len, "context") == 0)
                params.checkAction = CHECKACTION.context;
            else
            {
                errorInvalidSwitch(p);
                params.checkActionUsage = true;
                return false;
            }
        }
        else if (startsWith(p + 1, "check")) // https://dlang.org/dmd.html#switch-check
        {
            enum len = "-check=".length;
            mixin(checkOptionsMixin("checkUsage",
                "`-check=<action>` requires an action"));
            /* Parse:
             *    -check=[assert|bounds|in|invariant|out|switch][=[on|off]]
             */

            // Check for legal option string; return true if so
            static bool check(const(char)* p, string name, ref CHECKENABLE ce)
            {
                p += len;
                if (startsWith(p, name))
                {
                    p += name.length;
                    if (*p == 0 ||
                        strcmp(p, "=on") == 0)
                    {
                        ce = CHECKENABLE.on;
                        return true;
                    }
                    else if (strcmp(p, "=off") == 0)
                    {
                        ce = CHECKENABLE.off;
                        return true;
                    }
                }
                return false;
            }

            if (!(check(p, "assert",    params.useAssert     ) ||
                  check(p, "bounds",    params.useArrayBounds) ||
                  check(p, "in",        params.useIn         ) ||
                  check(p, "invariant", params.useInvariants ) ||
                  check(p, "out",       params.useOut        ) ||
                  check(p, "switch",    params.useSwitchError)))
            {
                errorInvalidSwitch(p);
                params.checkUsage = true;
                return false;
            }
        }
        else if (startsWith(p + 1, "color")) // https://dlang.org/dmd.html#switch-color
        {
            // Parse:
            //      -color
            //      -color=auto|on|off
            if (p[6] == '=')
            {
                if (strcmp(p + 7, "on") == 0)
                    params.color = true;
                else if (strcmp(p + 7, "off") == 0)
                    params.color = false;
                else if (strcmp(p + 7, "auto") != 0)
                {
                    errorInvalidSwitch(p, "Available options for `-color` are `on`, `off` and `auto`");
                    return true;
                }
            }
            else if (p[6])
                goto Lerror;
            else
                params.color = true;
        }
        else if (startsWith(p + 1, "conf=")) // https://dlang.org/dmd.html#switch-conf
        {
            // ignore, already handled above
        }
        else if (startsWith(p + 1, "cov")) // https://dlang.org/dmd.html#switch-cov
        {
            params.cov = true;
            // Parse:
            //      -cov
            //      -cov=nnn
            if (p[4] == '=')
            {
                if (isdigit(cast(char)p[5]))
                {
                    const percent = parseDigits(p + 5, 100);
                    if (percent == uint.max)
                        goto Lerror;
                    params.covPercent = cast(ubyte)percent;
                }
                else
                {
                    errorInvalidSwitch(p, "Only a number can be passed to `-cov=<num>`");
                    return true;
                }
            }
            else if (p[4])
                goto Lerror;
        }
        else if (arg == "-shared")
            params.dll = true;
        else if (arg == "-fPIC")
        {
            static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
            {
                params.pic = PIC.pic;
            }
            else
            {
                goto Lerror;
            }
        }
        else if (arg == "-fPIE")
        {
            static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
            {
                params.pic = PIC.pie;
            }
            else
            {
                goto Lerror;
            }
        }
        else if (arg == "-map") // https://dlang.org/dmd.html#switch-map
            params.map = true;
        else if (arg == "-multiobj")
            params.multiobj = true;
        else if (startsWith(p + 1, "mixin="))
        {
            auto tmp = p + 6 + 1;
            if (!tmp[0])
                goto Lnoarg;
            params.mixinFile = mem.xstrdup(tmp);
        }
        else if (arg == "-g") // https://dlang.org/dmd.html#switch-g
            params.symdebug = 1;
        else if (arg == "-gf")
        {
            if (!params.symdebug)
                params.symdebug = 1;
            params.symdebugref = true;
        }
        else if (arg == "-gs")  // https://dlang.org/dmd.html#switch-gs
            params.alwaysframe = true;
        else if (arg == "-gx")  // https://dlang.org/dmd.html#switch-gx
            params.stackstomp = true;
        else if (arg == "-lowmem") // https://dlang.org/dmd.html#switch-lowmem
        {
            static if (isGCAvailable)
            {
                // ignore, already handled in C main
            }
            else
            {
                error("switch '%s' requires DMD to be built with '-version=GC'", arg.ptr);
                continue;
            }
        }
        else if (arg.length > 6 && arg[0..6] == "--DRT-")
        {
            continue; // skip druntime options, e.g. used to configure the GC
        }
        else if (arg == "-m32") // https://dlang.org/dmd.html#switch-m32
        {
            static if (TARGET.DragonFlyBSD) {
                error("-m32 is not supported on DragonFlyBSD, it is 64-bit only");
            } else {
                params.is64bit = false;
                params.mscoff = false;
            }
        }
        else if (arg == "-m64") // https://dlang.org/dmd.html#switch-m64
        {
            params.is64bit = true;
            static if (TARGET.Windows)
            {
                params.mscoff = true;
            }
        }
        else if (arg == "-m32mscoff") // https://dlang.org/dmd.html#switch-m32mscoff
        {
            static if (TARGET.Windows)
            {
                params.is64bit = 0;
                params.mscoff = true;
            }
            else
            {
                error("-m32mscoff can only be used on windows");
            }
        }
        else if (strncmp(p + 1, "mscrtlib=", 9) == 0)
        {
            static if (TARGET.Windows)
            {
                params.mscrtlib = (p + 10).toDString;
            }
            else
            {
                error("-mscrtlib");
            }
        }
        else if (startsWith(p + 1, "profile")) // https://dlang.org/dmd.html#switch-profile
        {
            // Parse:
            //      -profile
            //      -profile=gc
            if (p[8] == '=')
            {
                if (strcmp(p + 9, "gc") == 0)
                    params.tracegc = true;
                else
                {
                    errorInvalidSwitch(p, "Only `gc` is allowed for `-profile`");
                    return true;
                }
            }
            else if (p[8])
                goto Lerror;
            else
                params.trace = true;
        }
        else if (arg == "-v") // https://dlang.org/dmd.html#switch-v
            params.verbose = true;
        else if (arg == "-vcg-ast")
            params.vcg_ast = true;
        else if (arg == "-vtls") // https://dlang.org/dmd.html#switch-vtls
            params.vtls = true;
        else if (arg == "-vcolumns") // https://dlang.org/dmd.html#switch-vcolumns
            params.showColumns = true;
        else if (arg == "-vgc") // https://dlang.org/dmd.html#switch-vgc
            params.vgc = true;
        else if (startsWith(p + 1, "verrors")) // https://dlang.org/dmd.html#switch-verrors
        {
            if (p[8] == '=' && isdigit(cast(char)p[9]))
            {
                const num = parseDigits(p + 9, int.max);
                if (num == uint.max)
                    goto Lerror;
                params.errorLimit = num;
            }
            else if (startsWith(p + 9, "spec"))
            {
                params.showGaggedErrors = true;
            }
            else if (startsWith(p + 9, "context"))
            {
                params.printErrorContext = true;
            }
            else
            {
                errorInvalidSwitch(p, "Only number, `spec`, or `context` are allowed for `-verrors`");
                return true;
            }
        }
        else if (startsWith(p + 1, "mcpu")) // https://dlang.org/dmd.html#switch-mcpu
        {
            enum len = "-mcpu=".length;
            // Parse:
            //      -mcpu=identifier
            mixin(checkOptionsMixin("mcpuUsage",
                "`-mcpu=<architecture>` requires an architecture"));
            if (Identifier.isValidIdentifier(p + len))
            {
                const ident = p + len;
                switch (ident.toDString())
                {
                case "baseline":
                    params.cpu = CPU.baseline;
                    break;
                case "avx":
                    params.cpu = CPU.avx;
                    break;
                case "avx2":
                    params.cpu = CPU.avx2;
                    break;
                case "native":
                    params.cpu = CPU.native;
                    break;
                default:
                    errorInvalidSwitch(p, "Only `baseline`, `avx`, `avx2` or `native` are allowed for `-mcpu`");
                    params.mcpuUsage = true;
                    return false;
                }
            }
            else
            {
                errorInvalidSwitch(p, "Only `baseline`, `avx`, `avx2` or `native` are allowed for `-mcpu`");
                params.mcpuUsage = true;
                return false;
            }
        }
        else if (startsWith(p + 1, "extern-std")) // https://dlang.org/dmd.html#switch-extern-std
        {
            enum len = "-extern-std=".length;
            // Parse:
            //      -extern-std=identifier
            mixin(checkOptionsMixin("externStdUsage",
                "`-extern-std=<standard>` requires a standard"));
            if (strcmp(p + len, "c++98") == 0)
                params.cplusplus = CppStdRevision.cpp98;
            else if (strcmp(p + len, "c++11") == 0)
                params.cplusplus = CppStdRevision.cpp11;
            else if (strcmp(p + len, "c++14") == 0)
                params.cplusplus = CppStdRevision.cpp14;
            else if (strcmp(p + len, "c++17") == 0)
                params.cplusplus = CppStdRevision.cpp17;
            else
            {
                error("Switch `%s` is invalid", p);
                params.externStdUsage = true;
                return false;
            }
        }
        else if (startsWith(p + 1, "transition")) // https://dlang.org/dmd.html#switch-transition
        {
            enum len = "-transition=".length;
            // Parse:
            //      -transition=number
            mixin(checkOptionsMixin("transitionUsage",
                "`-transition=<name>` requires a name"));
            if (!parseCLIOption!("transition", Usage.transitions)(params, p))
            {
                // Legacy -transition flags
                // Before DMD 2.085, DMD `-transition` was used for all language flags
                // These are kept for backwards compatibility, but no longer documented
                if (isdigit(cast(char)p[len]))
                {
                    const num = parseDigits(p + len, int.max);
                    if (num == uint.max)
                        goto Lerror;

                    // Bugzilla issue number
                    switch (num)
                    {
                        case 3449:
                            params.vfield = true;
                            break;
                        case 10378:
                            params.bug10378 = true;
                            break;
                        case 14246:
                            params.dtorFields = true;
                            break;
                        case 14488:
                            params.vcomplex = true;
                            break;
                        case 16997:
                            params.fix16997 = true;
                            break;
                        default:
                            error("Transition `%s` is invalid", p);
                            params.transitionUsage = true;
                            return false;
                    }
                }
                else if (Identifier.isValidIdentifier(p + len))
                {
                    const ident = p + len;
                    switch (ident.toDString())
                    {
                        case "import":
                            params.bug10378 = true;
                            break;
                        case "dtorfields":
                            params.dtorFields = true;
                            break;
                        case "intpromote":
                            params.fix16997 = true;
                            break;
                        case "markdown":
                            params.markdown = true;
                            break;
                        default:
                            error("Transition `%s` is invalid", p);
                            params.transitionUsage = true;
                            return false;
                    }
                }
                errorInvalidSwitch(p);
                params.transitionUsage = true;
                return false;
            }
        }
        else if (startsWith(p + 1, "preview") ) // https://dlang.org/dmd.html#switch-preview
        {
            enum len = "-preview=".length;
            // Parse:
            //      -preview=name
            mixin(checkOptionsMixin("previewUsage",
                "`-preview=<name>` requires a name"));

            if (!parseCLIOption!("preview", Usage.previews)(params, p))
            {
                error("Preview `%s` is invalid", p);
                params.previewUsage = true;
                return false;
            }

            if (params.useDIP1021)
                params.vsafe = true;    // dip1021 implies dip1000

            // copy previously standalone flags from -transition
            // -preview=dip1000 implies -preview=dip25 too
            if (params.vsafe)
                params.useDIP25 = true;
        }
        else if (startsWith(p + 1, "revert") ) // https://dlang.org/dmd.html#switch-revert
        {
            enum len = "-revert=".length;
            // Parse:
            //      -revert=name
            mixin(checkOptionsMixin("revertUsage",
                "`-revert=<name>` requires a name"));

            if (!parseCLIOption!("revert", Usage.reverts)(params, p))
            {
                error("Revert `%s` is invalid", p);
                params.revertUsage = true;
                return false;
            }

            if (params.noDIP25)
                params.useDIP25 = false;
        }
        else if (arg == "-w")   // https://dlang.org/dmd.html#switch-w
            params.warnings = DiagnosticReporting.error;
        else if (arg == "-wi")  // https://dlang.org/dmd.html#switch-wi
            params.warnings = DiagnosticReporting.inform;
        else if (arg == "-O")   // https://dlang.org/dmd.html#switch-O
            params.optimize = true;
        else if (p[1] == 'o')
        {
            const(char)* path;
            switch (p[2])
            {
            case '-':                       // https://dlang.org/dmd.html#switch-o-
                params.obj = false;
                break;
            case 'd':                       // https://dlang.org/dmd.html#switch-od
                if (!p[3])
                    goto Lnoarg;
                path = p + 3 + (p[3] == '=');
                version (Windows)
                {
                    path = toWinPath(path);
                }
                params.objdir = path.toDString;
                break;
            case 'f':                       // https://dlang.org/dmd.html#switch-of
                if (!p[3])
                    goto Lnoarg;
                path = p + 3 + (p[3] == '=');
                version (Windows)
                {
                    path = toWinPath(path);
                }
                params.objname = path.toDString;
                break;
            case 'p':                       // https://dlang.org/dmd.html#switch-op
                if (p[3])
                    goto Lerror;
                params.preservePaths = true;
                break;
            case 0:
                error("-o no longer supported, use -of or -od");
                break;
            default:
                goto Lerror;
            }
        }
        else if (p[1] == 'D')       // https://dlang.org/dmd.html#switch-D
        {
            params.doDocComments = true;
            switch (p[2])
            {
            case 'd':               // https://dlang.org/dmd.html#switch-Dd
                if (!p[3])
                    goto Lnoarg;
                params.docdir = p + 3 + (p[3] == '=');
                break;
            case 'f':               // https://dlang.org/dmd.html#switch-Df
                if (!p[3])
                    goto Lnoarg;
                params.docname = p + 3 + (p[3] == '=');
                break;
            case 0:
                break;
            default:
                goto Lerror;
            }
        }
        else if (p[1] == 'H')       // https://dlang.org/dmd.html#switch-H
        {
            params.doHdrGeneration = true;
            switch (p[2])
            {
            case 'd':               // https://dlang.org/dmd.html#switch-Hd
                if (!p[3])
                    goto Lnoarg;
                params.hdrdir = (p + 3 + (p[3] == '=')).toDString;
                break;
            case 'f':               // https://dlang.org/dmd.html#switch-Hf
                if (!p[3])
                    goto Lnoarg;
                params.hdrname = (p + 3 + (p[3] == '=')).toDString;
                break;
            case 0:
                break;
            default:
                goto Lerror;
            }
        }
        else if (startsWith(p + 1, "Xcc="))
        {
            // Linking code is guarded by version (Posix):
            static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
            {
                params.linkswitches.push(p + 5);
                params.linkswitchIsForCC.push(true);
            }
            else
            {
                goto Lerror;
            }
        }
        else if (p[1] == 'X')       // https://dlang.org/dmd.html#switch-X
        {
            params.doJsonGeneration = true;
            switch (p[2])
            {
            case 'f':               // https://dlang.org/dmd.html#switch-Xf
                if (!p[3])
                    goto Lnoarg;
                params.jsonfilename = (p + 3 + (p[3] == '=')).toDString;
                break;
            case 'i':
                if (!p[3])
                    goto Lnoarg;
                if (p[3] != '=')
                    goto Lerror;
                if (!p[4])
                    goto Lnoarg;

                {
                    auto flag = tryParseJsonField(p + 4);
                    if (!flag)
                    {
                        error("unknown JSON field `-Xi=%s`, expected one of " ~ jsonFieldNames, p + 4);
                        continue;
                    }
                    global.params.jsonFieldFlags |= flag;
                }
                break;
            case 0:
                break;
            default:
                goto Lerror;
            }
        }
        else if (arg == "-ignore")      // https://dlang.org/dmd.html#switch-ignore
            params.ignoreUnsupportedPragmas = true;
        else if (arg == "-inline")      // https://dlang.org/dmd.html#switch-inline
        {
            params.useInline = true;
            params.hdrStripPlainFunctions = false;
        }
        else if (arg == "-i")
            includeImports = true;
        else if (startsWith(p + 1, "i="))
        {
            includeImports = true;
            if (!p[3])
            {
                error("invalid option '%s', module patterns cannot be empty", p);
            }
            else
            {
                // NOTE: we could check that the argument only contains valid "module-pattern" characters.
                //       Invalid characters doesn't break anything but an error message to the user might
                //       be nice.
                includeModulePatterns.push(p + 3);
            }
        }
        else if (arg == "-dip25")       // https://dlang.org/dmd.html#switch-dip25
            params.useDIP25 = true;
        else if (arg == "-dip1000")
        {
            params.useDIP25 = true;
            params.vsafe = true;
        }
        else if (arg == "-dip1008")
        {
            params.ehnogc = true;
        }
        else if (arg == "-lib")         // https://dlang.org/dmd.html#switch-lib
            params.lib = true;
        else if (arg == "-nofloat")
            params.nofloat = true;
        else if (arg == "-quiet")
        {
            // Ignore
        }
        else if (arg == "-release")     // https://dlang.org/dmd.html#switch-release
            params.release = true;
        else if (arg == "-betterC")     // https://dlang.org/dmd.html#switch-betterC
            params.betterC = true;
        else if (arg == "-noboundscheck") // https://dlang.org/dmd.html#switch-noboundscheck
        {
            params.boundscheck = CHECKENABLE.off;
        }
        else if (startsWith(p + 1, "boundscheck")) // https://dlang.org/dmd.html#switch-boundscheck
        {
            // Parse:
            //      -boundscheck=[on|safeonly|off]
            if (p[12] == '=')
            {
                if (strcmp(p + 13, "on") == 0)
                {
                    params.boundscheck = CHECKENABLE.on;
                }
                else if (strcmp(p + 13, "safeonly") == 0)
                {
                    params.boundscheck = CHECKENABLE.safeonly;
                }
                else if (strcmp(p + 13, "off") == 0)
                {
                    params.boundscheck = CHECKENABLE.off;
                }
                else
                    goto Lerror;
            }
            else
                goto Lerror;
        }
        else if (arg == "-unittest")
            params.useUnitTests = true;
        else if (p[1] == 'I')              // https://dlang.org/dmd.html#switch-I
        {
            if (!params.imppath)
                params.imppath = new Strings();
            params.imppath.push(p + 2 + (p[2] == '='));
        }
        else if (p[1] == 'm' && p[2] == 'v' && p[3] == '=') // https://dlang.org/dmd.html#switch-mv
        {
            if (p[4] && strchr(p + 5, '='))
            {
                params.modFileAliasStrings.push(p + 4);
            }
            else
                goto Lerror;
        }
        else if (p[1] == 'J')             // https://dlang.org/dmd.html#switch-J
        {
            if (!params.fileImppath)
                params.fileImppath = new Strings();
            params.fileImppath.push(p + 2 + (p[2] == '='));
        }
        else if (startsWith(p + 1, "debug") && p[6] != 'l') // https://dlang.org/dmd.html#switch-debug
        {
            // Parse:
            //      -debug
            //      -debug=number
            //      -debug=identifier
            if (p[6] == '=')
            {
                if (isdigit(cast(char)p[7]))
                {
                    const level = parseDigits(p + 7, int.max);
                    if (level == uint.max)
                        goto Lerror;

                    params.debuglevel = level;
                }
                else if (Identifier.isValidIdentifier(p + 7))
                {
                    if (!params.debugids)
                        params.debugids = new Array!(const(char)*);
                    params.debugids.push(p + 7);
                }
                else
                    goto Lerror;
            }
            else if (p[6])
                goto Lerror;
            else
                params.debuglevel = 1;
        }
        else if (startsWith(p + 1, "version")) // https://dlang.org/dmd.html#switch-version
        {
            // Parse:
            //      -version=number
            //      -version=identifier
            if (p[8] == '=')
            {
                if (isdigit(cast(char)p[9]))
                {
                    const level = parseDigits(p + 9, int.max);
                    if (level == uint.max)
                        goto Lerror;
                    params.versionlevel = level;
                }
                else if (Identifier.isValidIdentifier(p + 9))
                {
                    if (!params.versionids)
                        params.versionids = new Array!(const(char)*);
                    params.versionids.push(p + 9);
                }
                else
                    goto Lerror;
            }
            else
                goto Lerror;
        }
        else if (arg == "--b")
            params.debugb = true;
        else if (arg == "--c")
            params.debugc = true;
        else if (arg == "--f")
            params.debugf = true;
        else if (arg == "--help" ||
                 arg == "-h")
        {
            params.usage = true;
            return false;
        }
        else if (arg == "--r")
            params.debugr = true;
        else if (arg == "--version")
        {
            params.logo = true;
            return false;
        }
        else if (arg == "--x")
            params.debugx = true;
        else if (arg == "--y")
            params.debugy = true;
        else if (p[1] == 'L')                        // https://dlang.org/dmd.html#switch-L
        {
            params.linkswitches.push(p + 2 + (p[2] == '='));
            params.linkswitchIsForCC.push(false);
        }
        else if (startsWith(p + 1, "defaultlib="))   // https://dlang.org/dmd.html#switch-defaultlib
        {
            params.defaultlibname = (p + 1 + 11).toDString;
        }
        else if (startsWith(p + 1, "debuglib="))     // https://dlang.org/dmd.html#switch-debuglib
        {
            params.debuglibname = (p + 1 + 9).toDString;
        }
        else if (startsWith(p + 1, "deps"))          // https://dlang.org/dmd.html#switch-deps
        {
            if (params.moduleDeps)
            {
                error("-deps[=file] can only be provided once!");
                break;
            }
            if (p[5] == '=')
            {
                params.moduleDepsFile = (p + 1 + 5).toDString;
                if (!params.moduleDepsFile[0])
                    goto Lnoarg;
            }
            else if (p[5] != '\0')
            {
                // Else output to stdout.
                goto Lerror;
            }
            params.moduleDeps = new OutBuffer();
        }
        else if (arg == "-main")             // https://dlang.org/dmd.html#switch-main
        {
            params.addMain = true;
        }
        else if (startsWith(p + 1, "man"))   // https://dlang.org/dmd.html#switch-man
        {
            params.manual = true;
            return false;
        }
        else if (arg == "-run")              // https://dlang.org/dmd.html#switch-run
        {
            params.run = true;
            size_t length = argc - i - 1;
            if (length)
            {
                const(char)* ext = FileName.ext(arguments[i + 1]);
                if (ext && FileName.equals(ext, "d") == 0 && FileName.equals(ext, "di") == 0)
                {
                    error("-run must be followed by a source file, not '%s'", arguments[i + 1]);
                    break;
                }
                if (strcmp(arguments[i + 1], "-") == 0)
                    files.push("__stdin.d");
                else
                    files.push(arguments[i + 1]);
                params.runargs.setDim(length - 1);
                for (size_t j = 0; j < length - 1; ++j)
                {
                    params.runargs[j] = arguments[i + 2 + j];
                }
                i += length;
            }
            else
            {
                params.run = false;
                goto Lnoarg;
            }
        }
        else if (p[1] == '\0')
            files.push("__stdin.d");
        else
        {
        Lerror:
            error("unrecognized switch '%s'", arguments[i]);
            continue;
        Lnoarg:
            error("argument expected for switch '%s'", arguments[i]);
            continue;
        }
    }
    return errors;
}

} // !IN_LLVM

/***********************************************
 * Adjust gathered command line switches and reconcile them.
 * Params:
 *      params = switches gathered from command line,
 *               and update in place
 *      numSrcFiles = number of source files
 */
version (NoMain) {} else
private void reconcileCommands(ref Param params, size_t numSrcFiles)
{
version (IN_LLVM)
{
    if (params.lib && params.dll)
        error(Loc.initial, "cannot mix -lib and -shared");
}
else
{
    static if (TARGET.OSX)
    {
        params.pic = PIC.pic;
    }
    static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
    {
        if (params.lib && params.dll)
            error(Loc.initial, "cannot mix -lib and -shared");
    }
    static if (TARGET.Windows)
    {
        if (params.mscoff && !params.mscrtlib)
        {
            VSOptions vsopt;
            vsopt.initialize();
            params.mscrtlib = vsopt.defaultRuntimeLibrary(params.is64bit).toDString;
        }
    }

    // Target uses 64bit pointers.
    params.isLP64 = params.is64bit;
} // !IN_LLVM

    if (params.boundscheck != CHECKENABLE._default)
    {
        if (params.useArrayBounds == CHECKENABLE._default)
            params.useArrayBounds = params.boundscheck;
    }

    if (params.useUnitTests)
    {
        if (params.useAssert == CHECKENABLE._default)
            params.useAssert = CHECKENABLE.on;
    }

    if (params.release)
    {
        if (params.useInvariants == CHECKENABLE._default)
            params.useInvariants = CHECKENABLE.off;

        if (params.useIn == CHECKENABLE._default)
            params.useIn = CHECKENABLE.off;

        if (params.useOut == CHECKENABLE._default)
            params.useOut = CHECKENABLE.off;

        if (params.useArrayBounds == CHECKENABLE._default)
            params.useArrayBounds = CHECKENABLE.safeonly;

        if (params.useAssert == CHECKENABLE._default)
            params.useAssert = CHECKENABLE.off;

        if (params.useSwitchError == CHECKENABLE._default)
            params.useSwitchError = CHECKENABLE.off;
    }
    else
    {
        if (params.useInvariants == CHECKENABLE._default)
            params.useInvariants = CHECKENABLE.on;

        if (params.useIn == CHECKENABLE._default)
            params.useIn = CHECKENABLE.on;

        if (params.useOut == CHECKENABLE._default)
            params.useOut = CHECKENABLE.on;

        if (params.useArrayBounds == CHECKENABLE._default)
            params.useArrayBounds = CHECKENABLE.on;

        if (params.useAssert == CHECKENABLE._default)
            params.useAssert = CHECKENABLE.on;

        if (params.useSwitchError == CHECKENABLE._default)
            params.useSwitchError = CHECKENABLE.on;
    }

    if (params.betterC)
    {
        params.checkAction = CHECKACTION.C;
        params.useModuleInfo = false;
        params.useTypeInfo = false;
        params.useExceptions = false;
    }


    if (!params.obj || params.lib || (IN_LLVM && params.output_o == OUTPUTFLAGno))
        params.link = false;
    if (params.link)
    {
        params.exefile = params.objname;
        params.oneobj = true;
        if (params.objname)
        {
            /* Use this to name the one object file with the same
             * name as the exe file.
             */
            params.objname = FileName.forceExt(params.objname, global.obj_ext);
            /* If output directory is given, use that path rather than
             * the exe file path.
             */
            if (params.objdir)
            {
                const(char)[] name = FileName.name(params.objname);
                params.objname = FileName.combine(params.objdir, name);
            }
        }
    }
    else if (params.run)
    {
        error(Loc.initial, "flags conflict with -run");
        fatal();
    }
    else if (params.lib)
    {
        params.libname = params.objname;
        params.objname = null;
        // Haven't investigated handling these options with multiobj
        if (!IN_LLVM && !params.cov && !params.trace)
            params.multiobj = true;
    }
    else
    {
version (IN_LLVM)
{
        if (params.objname && numSrcFiles + (params.addMain ? 1 : 0) > 1)
            params.oneobj = true;
}
else
{
        if (params.objname && numSrcFiles)
        {
            params.oneobj = true;
            //error("multiple source files, but only one .obj name");
            //fatal();
        }
}
    }

    if (params.noDIP25)
        params.useDIP25 = false;
}

/**
Creates the list of modules based on the files provided

Files are dispatched in the various arrays
(global.params.{ddocfiles,dllfiles,jsonfiles,etc...})
according to their extension.
Binary files are added to libmodules.

Params:
  files = File names to dispatch
  libmodules = Array to which binaries (shared/static libs and object files)
               will be appended

Returns:
  An array of path to D modules
*/
Modules createModules(ref Strings files, ref Strings libmodules)
{
    Modules modules;
    modules.reserve(files.dim);
version (IN_LLVM)
{
    size_t firstModuleObjectFileIndex = size_t.max;
}
else
{
    bool firstmodule = true;
}
    for (size_t i = 0; i < files.dim; i++)
    {
        const(char)[] name;
version (IN_LLVM) {} else
{
        version (Windows)
        {
            files[i] = toWinPath(files[i]);
        }
}
        const(char)[] p = files[i].toDString();
        p = FileName.name(p); // strip path
        const(char)[] ext = FileName.ext(p);
        if (ext)
        {
            /* Deduce what to do with a file based on its extension
             */
            if (FileName.equals(ext, global.obj_ext))
            {
                global.params.objfiles.push(files[i]);
                libmodules.push(files[i]);
                continue;
            }
            // Detect LLVM bitcode files on commandline
            if (IN_LLVM && FileName.equals(ext, global.bc_ext))
            {
                global.params.bitcodeFiles.push(files[i]);
                continue;
            }
            if (FileName.equals(ext, global.lib_ext))
            {
                global.params.libfiles.push(files[i]);
                libmodules.push(files[i]);
                continue;
            }
            // IN_LLVM replaced: static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
            if (!global.params.isWindows)
            {
                if (FileName.equals(ext, global.dll_ext))
                {
                    global.params.dllfiles.push(files[i]);
                    libmodules.push(files[i]);
                    continue;
                }
            }
            if (ext == global.ddoc_ext)
            {
                global.params.ddocfiles.push(files[i]);
                continue;
            }
            if (FileName.equals(ext, global.json_ext))
            {
                global.params.doJsonGeneration = true;
                global.params.jsonfilename = files[i].toDString;
                continue;
            }
            if (FileName.equals(ext, global.map_ext))
            {
                global.params.mapfile = files[i].toDString;
                continue;
            }
            // IN_LLVM replaced: static if (TARGET.Windows)
            if (global.params.isWindows)
            {
                if (FileName.equals(ext, "res"))
                {
                    global.params.resfile = files[i].toDString;
                    continue;
                }
                if (FileName.equals(ext, "def"))
                {
                    global.params.deffile = files[i].toDString;
                    continue;
                }
                if (FileName.equals(ext, "exe"))
                {
                    assert(0); // should have already been handled
                }
            }
            /* Examine extension to see if it is a valid
             * D source file extension
             */
            if (FileName.equals(ext, global.mars_ext) || FileName.equals(ext, global.hdr_ext) || FileName.equals(ext, "dd"))
            {
                name = FileName.removeExt(p);
                if (!name.length || name == ".." || name == ".")
                {
                Linvalid:
                    error(Loc.initial, "invalid file name '%s'", files[i]);
                    fatal();
                }
            }
            else
            {
                error(Loc.initial, "unrecognized file extension %.*s", cast(int)ext.length, ext.ptr);
                fatal();
            }
        }
        else
        {
            name = p;
            if (!name.length)
                goto Linvalid;
        }
        /* At this point, name is the D source file name stripped of
         * its path and extension.
         */
        auto id = Identifier.idPool(name);
        auto m = new Module(files[i].toDString, id, global.params.doDocComments, global.params.doHdrGeneration);
        modules.push(m);
version (IN_LLVM)
{
        if (!global.params.oneobj || firstModuleObjectFileIndex == size_t.max)
        {
            global.params.objfiles.push(cast(const(char)*)m); // defer to a later stage after parsing
            if (firstModuleObjectFileIndex == size_t.max)
                firstModuleObjectFileIndex = global.params.objfiles.dim - 1;
        }
}
else
{
        if (firstmodule)
        {
            global.params.objfiles.push(m.objfile.toChars());
            firstmodule = false;
        }
}
    }
version (IN_LLVM)
{
    if (global.params.oneobj && modules.dim < 2 && !includeImports)
        global.params.oneobj = false;
    // global.params.oneobj => move object file for first source file to
    // beginning of object files list
    if (global.params.oneobj && firstModuleObjectFileIndex != 0)
    {
        auto fn = global.params.objfiles[firstModuleObjectFileIndex];
        global.params.objfiles.remove(firstModuleObjectFileIndex);
        global.params.objfiles.insert(0, fn);
    }
}
    return modules;
}
