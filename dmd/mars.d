/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 * Entry point for DMD.
 *
 * This modules defines the entry point (main) for DMD, as well as related
 * utilities needed for arguments parsing, path manipulation, etc...
 * This file is not shared with other compilers which use the DMD front-end.
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
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
// IN_LLVM import dmd.dinifile;
import dmd.dinterpret;
import dmd.dmodule;
import dmd.doc;
import dmd.dscope;
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
// IN_LLVM import dmd.lib;
// IN_LLVM import dmd.link;
import dmd.mtype;
import dmd.objc;
import dmd.parse;
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
import dmd.tokens;
import dmd.utils;

version(IN_LLVM)
{
    import gen.semantic : extraLDCSpecificSemanticAnalysis;
    extern (C++):

    // in driver/main.cpp
    void addDefaultVersionIdentifiers();
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
    printf("DMD%llu D Compiler %s\n%s %s\n", cast(ulong)size_t.sizeof * 8, global._version, global.copyright, global.written);
}


/**
 * Print DMD's usage message on stdout
 */
private void usage()
{
    import dmd.cli : CLIUsage;
    logo();
    auto help = CLIUsage.usage;
    printf("
Documentation: https://dlang.org/
Config file: %s
Usage:
  dmd [<option>...] <file>...
  dmd [<option>...] -run <file> [<arg>...]

Where:
  <file>           D source file
  <arg>            Argument to pass when running the resulting program

<option>:
  @<cmdfile>       read arguments from cmdfile
%.*s", FileName.canonicalName(global.inifilename), help.length, &help[0]);
}

} // !IN_LLVM

/// DMD-generated module `__entrypoint` where the C main resides
extern (C++) __gshared Module entrypoint = null;
/// Module in which the D main is
extern (C++) __gshared Module rootHasMain = null;


/**
 * Generate C main() in response to seeing D main().
 *
 * This function will generate a module called `__entrypoint`,
 * and set the globals `entrypoint` and `rootHasMain`.
 *
 * This used to be in druntime, but contained a reference to _Dmain
 * which didn't work when druntime was made into a dll and was linked
 * to a program, such as a C++ program, that didn't have a _Dmain.
 *
 * Params:
 *   sc = Scope which triggered the generation of the C main,
 *        used to get the module where the D main is.
 */
extern (C++) void genCmain(Scope* sc)
{
    if (entrypoint)
        return;
    /* The D code to be generated is provided as D source code in the form of a string.
     * Note that Solaris, for unknown reasons, requires both a main() and an _main()
     */
    version(IN_LLVM)
    {
        immutable cmaincode =
        q{
            pragma(LDC_profile_instr, false):
            extern(C)
            {
                int _d_run_main(int argc, char **argv, void* mainFunc);
                int _Dmain(char[][] args);
                int main(int argc, char **argv)
                {
                    return _d_run_main(argc, argv, &_Dmain);
                }
                version (Solaris) int _main(int argc, char** argv) { return main(argc, argv); }
            }
            pragma(LDC_no_moduleinfo);
        };
    }
    else
    {
        immutable cmaincode =
        q{
            extern(C)
            {
                int _d_run_main(int argc, char **argv, void* mainFunc);
                int _Dmain(char[][] args);
                int main(int argc, char **argv)
                {
                    return _d_run_main(argc, argv, &_Dmain);
                }
                version (Solaris) int _main(int argc, char** argv) { return main(argc, argv); }
            }
        };
    }
    Identifier id = Id.entrypoint;
    auto m = new Module("__entrypoint.d", id, 0, 0);
    scope p = new Parser!ASTCodegen(m, cmaincode, false);
    p.scanloc = Loc.initial;
    p.nextToken();
    m.members = p.parseModule();
    assert(p.token.value == TOK.endOfFile);
    assert(!p.errors); // shouldn't have failed to parse it
    bool v = global.params.verbose;
    global.params.verbose = false;
    m.importedFrom = m;
    m.importAll(null);
    m.dsymbolSemantic(null);
    m.semantic2(null);
    m.semantic3(null);
    global.params.verbose = v;
    entrypoint = m;
    rootHasMain = sc._module;
}

version(IN_LLVM) {} else
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
private int tryMain(size_t argc, const(char)** argv)
{
    Strings files;
    Strings libmodules;
    global._init();
    debug
    {
        printf("DMD %s DEBUG\n", global._version);
        fflush(stdout); // avoid interleaving with stderr output when redirecting
    }
    // Check for malformed input
    if (argc < 1 || !argv)
    {
    Largs:
        error(Loc.initial, "missing or null command line arguments");
        fatal();
    }
    // Convert argc/argv into arguments[] for easier handling
    Strings arguments;
    arguments.setDim(argc);
    for (size_t i = 0; i < argc; i++)
    {
        if (!argv[i])
            goto Largs;
        arguments[i] = argv[i];
    }
    if (response_expand(&arguments)) // expand response files
        error(Loc.initial, "can't open response file");
    //for (size_t i = 0; i < arguments.dim; ++i) printf("arguments[%d] = '%s'\n", i, arguments[i]);
    files.reserve(arguments.dim - 1);
    // Set default values
    global.params.argv0 = arguments[0];

    // Temporary: Use 32 bits as the default on Windows, for config parsing
    static if (TARGET.Windows)
        global.params.is64bit = false;

    global.inifilename = parse_conf_arg(&arguments);
    if (global.inifilename)
    {
        // can be empty as in -conf=
        if (strlen(global.inifilename) && !FileName.exists(global.inifilename))
            error(Loc.initial, "Config file '%s' does not exist.", global.inifilename);
    }
    else
    {
        version (Windows)
        {
            global.inifilename = findConfFile(global.params.argv0, "sc.ini");
        }
        else version (Posix)
        {
            global.inifilename = findConfFile(global.params.argv0, "dmd.conf");
        }
        else
        {
            static assert(0, "fix this");
        }
    }
    // Read the configurarion file
    auto inifile = File(global.inifilename);
    inifile.read();
    /* Need path of configuration file, for use in expanding @P macro
     */
    const(char)* inifilepath = FileName.path(global.inifilename);
    Strings sections;
    StringTable environment;
    environment._init(7);
    /* Read the [Environment] section, so we can later
     * pick up any DFLAGS settings.
     */
    sections.push("Environment");
    parseConfFile(&environment, global.inifilename, inifilepath, inifile.len, inifile.buffer, &sections);

    const(char)* arch = global.params.is64bit ? "64" : "32"; // use default
    arch = parse_arch_arg(&arguments, arch);

    // parse architecture from DFLAGS read from [Environment] section
    {
        Strings dflags;
        getenv_setargv(readFromEnv(&environment, "DFLAGS"), &dflags);
        environment.reset(7); // erase cached environment updates
        arch = parse_arch_arg(&dflags, arch);
    }

    bool is64bit = arch[0] == '6';

    version(Windows) // delete LIB entry in [Environment] (necessary for optlink) to allow inheriting environment for MS-COFF
        if (is64bit || strcmp(arch, "32mscoff") == 0)
            environment.update("LIB", 3).ptrvalue = null;

    // read from DFLAGS in [Environment{arch}] section
    char[80] envsection = void;
    sprintf(envsection.ptr, "Environment%s", arch);
    sections.push(envsection.ptr);
    parseConfFile(&environment, global.inifilename, inifilepath, inifile.len, inifile.buffer, &sections);
    getenv_setargv(readFromEnv(&environment, "DFLAGS"), &arguments);
    updateRealEnvironment(&environment);
    environment.reset(1); // don't need environment cache any more

    if (parseCommandLine(arguments, argc, global.params, files))
    {
        Loc loc;
        errorSupplemental(loc, "run 'dmd -man' to open browser on manual");
        return EXIT_FAILURE;
    }

    if (global.params.usage)
    {
        usage();
        return EXIT_SUCCESS;
    }

    if (global.params.logo)
    {
        logo();
        return EXIT_SUCCESS;
    }

    if (global.params.mcpuUsage)
    {
        import dmd.cli : CLIUsage;
        auto help = CLIUsage.mcpu;
        printf("%.*s", help.length, &help[0]);
        return EXIT_SUCCESS;
    }

    if (global.params.transitionUsage)
    {
        import dmd.cli : CLIUsage;
        auto help = CLIUsage.transitionUsage;
        printf("%.*s", help.length, &help[0]);
        return EXIT_SUCCESS;
    }

    if (global.params.manual)
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

    if (global.params.color)
        global.console = Console.create(core.stdc.stdio.stderr);

    global.params.cpu = setTargetCPU(global.params.cpu);
    if (global.params.is64bit != is64bit)
        error(Loc.initial, "the architecture must not be changed in the %s section of %s", envsection.ptr, global.inifilename);
    if (global.params.enforcePropertySyntax)
    {
        /*NOTE: -property used to disallow calling non-properties
         without parentheses. This behaviour has fallen from grace.
         Phobos dropped support for it while dmd still recognized it, so
         that the switch has effectively not been supported. Time to
         remove it from dmd.
         Step 1 (2.069): Deprecate -property and ignore it. */
        Loc loc;
        deprecation(loc, "The -property switch is deprecated and has no " ~
            "effect anymore.");
        /* Step 2: Remove -property. Throw an error when it's set.
         Do this by removing global.params.enforcePropertySyntax and the code
         above that sets it. Let it be handled as an unrecognized switch.
         Step 3: Possibly reintroduce -property with different semantics.
         Any new semantics need to be decided on first. */
    }
    // Target uses 64bit pointers.
    global.params.isLP64 = global.params.is64bit;
    if (global.errors)
    {
        fatal();
    }
    if (files.dim == 0)
    {
        if (global.params.jsonFieldFlags)
        {
            generateJson(null);
            return EXIT_SUCCESS;
        }
        usage();
        return EXIT_FAILURE;
    }
    static if (TARGET.OSX)
    {
        global.params.pic = 1;
    }
    static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
    {
        if (global.params.lib && global.params.dll)
            error(Loc.initial, "cannot mix -lib and -shared");
    }
    static if (TARGET.Windows)
    {
        if (global.params.mscoff && !global.params.mscrtlib)
        {
            VSOptions vsopt;
            vsopt.initialize();
            global.params.mscrtlib = vsopt.defaultRuntimeLibrary(global.params.is64bit);
        }
    }
    if (global.params.release)
    {
        global.params.useInvariants = false;
        global.params.useIn = false;
        global.params.useOut = false;

        if (global.params.useArrayBounds == CHECKENABLE._default)
            global.params.useArrayBounds = CHECKENABLE.safeonly;

        if (global.params.useAssert == CHECKENABLE._default)
            global.params.useAssert = CHECKENABLE.off;

        if (global.params.useSwitchError == CHECKENABLE._default)
            global.params.useSwitchError = CHECKENABLE.off;
    }
    if (global.params.betterC)
    {
        global.params.checkAction = CHECKACTION.C;
        global.params.useModuleInfo = false;
        global.params.useTypeInfo = false;
        global.params.useExceptions = false;
    }
    if (global.params.useUnitTests)
        global.params.useAssert = CHECKENABLE.on;

    if (global.params.useArrayBounds == CHECKENABLE._default)
        global.params.useArrayBounds = CHECKENABLE.on;

    if (global.params.useAssert == CHECKENABLE._default)
        global.params.useAssert = CHECKENABLE.on;

    if (global.params.useSwitchError == CHECKENABLE._default)
        global.params.useSwitchError = CHECKENABLE.on;

    if (!global.params.obj || global.params.lib)
        global.params.link = false;

    return mars_mainBody(files, libmodules);
}

} // !IN_LLVM

extern (C++) int mars_mainBody(ref Strings files, ref Strings libmodules)
{
    version(IN_LLVM)
    {
        if (global.params.color)
            global.console = Console.create(core.stdc.stdio.stderr);
    }

    if (global.params.link)
    {
        global.params.exefile = global.params.objname;
        global.params.oneobj = true;
        if (global.params.objname)
        {
            /* Use this to name the one object file with the same
             * name as the exe file.
             */
            global.params.objname = cast(char*)FileName.forceExt(global.params.objname, global.obj_ext);
            /* If output directory is given, use that path rather than
             * the exe file path.
             */
            if (global.params.objdir)
            {
                const(char)* name = FileName.name(global.params.objname);
                global.params.objname = cast(char*)FileName.combine(global.params.objdir, name);
            }
        }
    }
    else if (global.params.run)
    {
        error(Loc.initial, "flags conflict with -run");
        fatal();
    }
    else if (global.params.lib)
    {
        global.params.libname = global.params.objname;
        global.params.objname = null;
      version (IN_LLVM) {} else
      {
        // Haven't investigated handling these options with multiobj
        if (!global.params.cov && !global.params.trace)
            global.params.multiobj = true;
      }
    }
    else
    {
        if (global.params.objname && files.dim + (global.params.addMain ? 1 : 0) > 1)
        {
            global.params.oneobj = true;
            //error("multiple source files, but only one .obj name");
            //fatal();
        }
    }

    // Add in command line versions
    if (global.params.versionids)
        foreach (charz; *global.params.versionids)
            VersionCondition.addGlobalIdent(charz[0 .. strlen(charz)]);
    if (global.params.debugids)
        foreach (charz; *global.params.debugids)
            DebugCondition.addGlobalIdent(charz[0 .. strlen(charz)]);

    // Predefined version identifiers
    addDefaultVersionIdentifiers();

  version (IN_LLVM) {} else
  {
    setDefaultLibrary();
  }

    // Initialization
    Type._init();
    Id.initialize();
    Module._init();
    Module.onImport = &marsOnImport;
    Target._init();
    Expression._init();
    Objc._init();
    builtin_init();

  version (IN_LLVM)
  {
    // LDC prints binary/version/config before entering this function.
    // DMD prints the predefined versions as part of addDefaultVersionIdentifiers().
    // Let's do it here after initialization, as e.g. Objc.init() may add `D_ObjectiveC`.
    printPredefinedVersions();
  }
  else
  {
    printPredefinedVersions();

    if (global.params.verbose)
    {
        message("binary    %s", global.params.argv0);
        message("version   %s", global._version);
        message("config    %s", global.inifilename ? global.inifilename : "(none)");
        // Print DFLAGS environment variable
        {
            Strings dflags;
            getenv_setargv(readFromEnv(&environment, "DFLAGS"), &dflags);
            OutBuffer buf;
            foreach (flag; dflags.asDArray)
            {
                bool needsQuoting;
                for (auto flagp = flag; flagp; flagp++)
                {
                    auto c = flagp[0];
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

            auto res = buf.peekSlice() ? buf.peekSlice()[0 .. $ - 1] : "(none)";
            message("DFLAGS    %.*s", res.length, res.ptr);
        }
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

    global.path = buildPath(global.params.imppath);
    global.filePath = buildPath(global.params.fileImppath);

    if (global.params.addMain)
    {
        files.push(cast(char*)global.main_d); // a dummy name, we never actually look up this file
    }
    // Create Modules
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
        const(char)* name;
      version (IN_LLVM) {} else
      {
        version (Windows)
        {
            files[i] = toWinPath(files[i]);
        }
      }
        const(char)* p = files[i];
        p = FileName.name(p); // strip path
        const(char)* ext = FileName.ext(p);
        char* newname;
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
          version (IN_LLVM)
          {
            // Detect LLVM bitcode files on commandline
            if (FileName.equals(ext, global.bc_ext)) {
              global.params.bitcodeFiles.push(files[i]);
              continue;
            }
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
            if (strcmp(ext, global.ddoc_ext) == 0)
            {
                global.params.ddocfiles.push(files[i]);
                continue;
            }
            if (FileName.equals(ext, global.json_ext))
            {
                global.params.doJsonGeneration = true;
                global.params.jsonfilename = files[i];
                continue;
            }
            if (FileName.equals(ext, global.map_ext))
            {
                global.params.mapfile = files[i];
                continue;
            }
            // IN_LLVM replaced: static if (TARGET.Windows)
            if (global.params.isWindows)
            {
                if (FileName.equals(ext, "res"))
                {
                    global.params.resfile = files[i];
                    continue;
                }
                if (FileName.equals(ext, "def"))
                {
                    global.params.deffile = files[i];
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
                ext--; // skip onto '.'
                assert(*ext == '.');
                newname = cast(char*)mem.xmalloc((ext - p) + 1);
                memcpy(newname, p, ext - p);
                newname[ext - p] = 0; // strip extension
                name = newname;
                if (name[0] == 0 || strcmp(name, "..") == 0 || strcmp(name, ".") == 0)
                {
                Linvalid:
                    error(Loc.initial, "invalid file name '%s'", files[i]);
                    fatal();
                }
            }
            else
            {
                error(Loc.initial, "unrecognized file extension %s", ext);
                fatal();
            }
        }
        else
        {
            name = p;
            if (!*name)
                goto Linvalid;
        }
        /* At this point, name is the D source file name stripped of
         * its path and extension.
         */
        auto id = Identifier.idPool(name, cast(uint)strlen(name));
        auto m = new Module(files[i], id, global.params.doDocComments, global.params.doHdrGeneration);
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
            global.params.objfiles.push(m.objfile.name.str);
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
    // Read files
    /* Start by "reading" the dummy main.d file
     */
    if (global.params.addMain)
    {
        bool added = false;
        foreach (m; modules)
        {
            if (strcmp(m.srcfile.name.str, global.main_d) == 0)
            {
                string buf = "int main(){return 0;}";
                m.srcfile.setbuffer(cast(void*)buf.ptr, buf.length);
                m.srcfile._ref = 1;
                added = true;
                break;
            }
        }
        assert(added);
    }
    enum ASYNCREAD = false;
    static if (ASYNCREAD)
    {
        // Multi threaded
        AsyncRead* aw = AsyncRead.create(modules.dim);
        foreach (m; modules)
        {
            aw.addFile(m.srcfile);
        }
        aw.start();
    }
    else
    {
        // Single threaded
        foreach (m; modules)
        {
            m.read(Loc.initial);
        }
    }
    // Parse files
    bool anydocfiles = false;
    size_t filecount = modules.dim;
    for (size_t filei = 0, modi = 0; filei < filecount; filei++, modi++)
    {
        Module m = modules[modi];
        if (global.params.verbose)
            message("parse     %s", m.toChars());
        if (!Module.rootModule)
            Module.rootModule = m;
        m.importedFrom = m; // m.isRoot() == true
      version (IN_LLVM) {} else
      {
        if (!global.params.oneobj || modi == 0 || m.isDocFile)
            m.deleteObjFile();
      }
        static if (ASYNCREAD)
        {
            if (aw.read(filei))
            {
                error(Loc.initial, "cannot read file %s", m.srcfile.name.toChars());
                fatal();
            }
        }
        m.parse();
      version (IN_LLVM)
      {
        // Finalize output filenames. Update if `-oq` was specified (only feasible after parsing).
        if (global.params.fullyQualifiedObjectFiles && m.md)
        {
            m.objfile = m.setOutfile(global.params.objname, global.params.objdir, m.arg, FileName.ext(m.objfile.name.str));
            if (m.docfile)
                m.setDocfile();
            if (m.hdrfile)
                m.hdrfile = m.setOutfile(global.params.hdrname, global.params.hdrdir, m.arg, global.hdr_ext);
        }

        // If `-run` is passed, the obj file is temporary and is removed after execution.
        // Make sure the name does not collide with other files from other processes by
        // creating a unique filename.
        if (global.params.run)
            m.makeObjectFilenameUnique();

        // Set object filename in global.params.objfiles.
        for (size_t j = 0; j < global.params.objfiles.dim; j++)
        {
            if (global.params.objfiles[j] == cast(const(char)*)m)
            {
                global.params.objfiles[j] = m.objfile.name.str;
                if (!m.isDocFile && global.params.obj)
                    m.checkAndAddOutputFile(m.objfile);
                break;
            }
        }

        if (!global.params.oneobj || modi == 0 || m.isDocFile)
            m.deleteObjFile();
      }
        if (m.isDocFile)
        {
            anydocfiles = true;
            gendocfile(m);
            // Remove m from list of modules
            modules.remove(modi);
            modi--;
            // Remove m's object file from list of object files
            for (size_t j = 0; j < global.params.objfiles.dim; j++)
            {
                if (m.objfile.name.str == global.params.objfiles[j])
                {
                    global.params.objfiles.remove(j);
                    break;
                }
            }
            if (global.params.objfiles.dim == 0)
                global.params.link = false;
        }
    }
    static if (ASYNCREAD)
    {
        AsyncRead.dispose(aw);
    }
    if (anydocfiles && modules.dim && (global.params.oneobj || global.params.objname))
    {
        error(Loc.initial, "conflicting Ddoc and obj generation options");
        fatal();
    }
    if (global.errors)
        fatal();

    if (global.params.doHdrGeneration)
    {
        /* Generate 'header' import files.
         * Since 'header' import files must be independent of command
         * line switches and what else is imported, they are generated
         * before any semantic analysis.
         */
        foreach (m; modules)
        {
            if (global.params.verbose)
                message("import    %s", m.toChars());
            genhdrfile(m);
        }
    }
    if (global.errors)
        fatal();

    // load all unconditional imports for better symbol resolving
    foreach (m; modules)
    {
        if (global.params.verbose)
            message("importall %s", m.toChars());
        m.importAll(null);
    }
    if (global.errors)
        fatal();

  version (IN_LLVM) {} else
  {
    backend_init();
  }

    // Do semantic analysis
    foreach (m; modules)
    {
        if (global.params.verbose)
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
        if (global.params.verbose)
            message("semantic2 %s", m.toChars());
        m.semantic2(null);
    }
    Module.runDeferredSemantic2();
    if (global.errors)
        fatal();

    // Do pass 3 semantic analysis
    foreach (m; modules)
    {
        if (global.params.verbose)
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
            if (global.params.verbose)
                message("semantic3 %s", m.toChars());
            m.semantic3(null);
            modules.push(m);
        }
    }
    Module.runDeferredSemantic3();
    if (global.errors)
        fatal();

  version (IN_LLVM)
  {
    extraLDCSpecificSemanticAnalysis(modules);
  }
  else
  {
    // Scan for functions to inline
    if (global.params.useInline)
    {
        foreach (m; modules)
        {
            if (global.params.verbose)
                message("inline scan %s", m.toChars());
            inlineScanModule(m);
        }
    }
  }
    // Do not attempt to generate output files if errors or warnings occurred
    if (global.errors || global.warnings)
        fatal();

    // inlineScan incrementally run semantic3 of each expanded functions.
    // So deps file generation should be moved after the inlinig stage.
    if (global.params.moduleDeps)
    {
        foreach (i; 1 .. modules[0].aimports.dim)
            semantic3OnDependencies(modules[0].aimports[i]);

        OutBuffer* ob = global.params.moduleDeps;
        if (global.params.moduleDepsFile)
        {
            auto deps = File(global.params.moduleDepsFile);
            deps.setbuffer(cast(void*)ob.data, ob.offset);
            writeFile(Loc.initial, &deps);
          version (IN_LLVM)
          {
            // fix LDC issue #1625
            global.params.moduleDeps = null;
            global.params.moduleDepsFile = null;
          }
        }
        else
            printf("%.*s", cast(int)ob.offset, ob.data);
    }

    printCtfePerformanceStats();

  version (IN_LLVM) {} else
  {
    Library library = null;
    if (global.params.lib)
    {
        if (global.params.objfiles.dim == 0)
        {
            error(Loc.initial, "no input files");
            return EXIT_FAILURE;
        }
        library = Library.factory();
        library.setFilename(global.params.objdir, global.params.libname);
        // Add input object and input library files to output library
        for (size_t i = 0; i < libmodules.dim; i++)
        {
            const(char)* p = libmodules[i];
            library.addObject(p, null);
        }
    }
  }
    // Generate output files
    if (global.params.doJsonGeneration)
    {
        generateJson(&modules);
    }
    if (!global.errors && global.params.doDocComments)
    {
        foreach (m; modules)
        {
            gendocfile(m);
        }
    }
    if (global.params.vcg_ast)
    {
        import dmd.hdrgen;
        foreach (mod; modules)
        {
            auto buf = OutBuffer();
            buf.doindent = 1;
            scope HdrGenState hgs;
            hgs.fullDump = 1;
            scope PrettyPrintVisitor ppv = new PrettyPrintVisitor(&buf, &hgs);
            mod.accept(ppv);

            // write the output to $(filename).cg
            auto modFilename = mod.srcfile.toChars();
            auto modFilenameLength = strlen(modFilename);
            auto cgFilename = cast(char*)allocmemory(modFilenameLength + 4);
            memcpy(cgFilename, modFilename, modFilenameLength);
            cgFilename[modFilenameLength .. modFilenameLength + 4] = ".cg\0";
            auto cgFile = File(cgFilename);
            cgFile.setbuffer(buf.data, buf.offset);
            cgFile._ref = 1;
            cgFile.write();
        }
    }
  version (IN_LLVM)
  {
    codegenModules(modules);
  }
  else
  {
    if (!global.params.obj)
    {
    }
    else if (global.params.oneobj)
    {
        if (modules.dim)
            obj_start(cast(char*)modules[0].srcfile.toChars());
        foreach (m; modules)
        {
            if (global.params.verbose)
                message("code      %s", m.toChars());
            genObjFile(m, false);
            if (entrypoint && m == rootHasMain)
                genObjFile(entrypoint, false);
        }
        if (!global.errors && modules.dim)
        {
            obj_end(library, modules[0].objfile);
        }
    }
    else
    {
        foreach (m; modules)
        {
            if (global.params.verbose)
                message("code      %s", m.toChars());
            obj_start(cast(char*)m.srcfile.toChars());
            genObjFile(m, global.params.multiobj);
            if (entrypoint && m == rootHasMain)
                genObjFile(entrypoint, global.params.multiobj);
            obj_end(library, m.objfile);
            obj_write_deferred(library);
            if (global.errors && !global.params.lib)
                m.deleteObjFile();
        }
    }
    if (global.params.lib && !global.errors)
        library.write();
    backend_term();
  }
    if (global.errors)
        fatal();
    int status = EXIT_SUCCESS;
    if (!global.params.objfiles.dim)
    {
      version (IN_LLVM)
      {
        if (global.params.link)
            error(Loc.initial, "no object files to link");
        else if (global.params.lib)
            error(Loc.initial, "no object files");
      }
      else
      {
        if (global.params.link)
            error(Loc.initial, "no object files to link");
      }
    }
    else
    {
      version (IN_LLVM)
      {
        if (global.params.link)
            status = linkObjToBinary();
        else if (global.params.lib)
            status = createStaticLibrary();

        if (status == EXIT_SUCCESS &&
            (global.params.cleanupObjectFiles || global.params.run))
        {
            for (size_t i = 0; i < modules.dim; i++)
            {
                modules[i].deleteObjFile();
                if (global.params.oneobj)
                    break;
            }
        }
      }
      else
      {
        if (global.params.link)
            status = runLINK();
      }
        if (global.params.run)
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
                    if (global.params.oneobj)
                        break;
                }
              }
                remove(global.params.exefile);
            }
        }
    }
    if (global.errors || global.warnings)
        fatal();
    return status;
}

// IN_LLVM replaced: `private` by `extern (C++)`
extern (C++) void generateJson(Modules* modules)
{
    OutBuffer buf;
    json_generate(&buf, modules);

    // Write buf to file
    const(char)* name = global.params.jsonfilename;
    if (name && name[0] == '-' && name[1] == 0)
    {
        // Write to stdout; assume it succeeds
        size_t n = fwrite(buf.data, 1, buf.offset, stdout);
        assert(n == buf.offset); // keep gcc happy about return values
    }
    else
    {
        /* The filename generation code here should be harmonized with Module::setOutfile()
         */
        const(char)* jsonfilename;
        if (name && *name)
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
            const(char)* n = global.params.objfiles[0];
            n = FileName.name(n);
            //if (!FileName::absolute(name))
            //    name = FileName::combine(dir, name);
            jsonfilename = FileName.forceExt(n, global.json_ext);
        }
        ensurePathToNameExists(Loc.initial, jsonfilename);
        auto jsonfile = new File(jsonfilename);
        jsonfile.setbuffer(buf.data, buf.offset);
        jsonfile._ref = 1;
        writeFile(Loc.initial, jsonfile);
    }
}


version (IN_LLVM) {} else
{

/**
 * Entry point which forwards to `tryMain`.
 *
 * Returns:
 *   Return code of the application
 */
version(NoMain) {} else
int main()
{
    import core.memory;
    import core.runtime;

    version (GC)
    {
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

    auto args = Runtime.cArgs();
    return tryMain(args.argc, cast(const(char)**)args.argv);
}


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
private void getenv_setargv(const(char)* envvalue, Strings* args)
{
    if (!envvalue)
        return;
    char* p;
    int instring;
    int slash;
    char c;
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
            args.push(env); // append
            p = env;
            slash = 0;
            instring = 0;
            while (1)
            {
                c = *env++;
                switch (c)
                {
                case '"':
                    p -= (slash >> 1);
                    if (slash & 1)
                    {
                        p--;
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
                Laddc:
                    slash = 0;
                    *p++ = c;
                    continue;
                }
                break;
            }
        }
    }
}

/**
 * Parse command line arguments for -m32 or -m64
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
private const(char)* parse_arch_arg(Strings* args, const(char)* arch)
{
    for (size_t i = 0; i < args.dim; ++i)
    {
        const(char)* p = (*args)[i];
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
 * Parse command line arguments for -conf=path.
 *
 * Params:
 *   args = Command line arguments
 *
 * Returns:
 *   Path to the config file to use
 */
private const(char)* parse_conf_arg(Strings* args)
{
    const(char)* conf = null;
    for (size_t i = 0; i < args.dim; ++i)
    {
        const(char)* p = (*args)[i];
        if (p[0] == '-')
        {
            if (strncmp(p + 1, "conf=", 5) == 0)
                conf = p + 6;
            else if (strcmp(p + 1, "run") == 0)
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
    else if (!global.params.defaultlibname[0])  // if `-defaultlib=` (i.e. an empty defaultlib)
        global.params.defaultlibname = null;

    if (global.params.debuglibname is null)
        global.params.debuglibname = global.params.defaultlibname;
}


/**
 * Add default `version` identifier for dmd, and set the
 * target platform in `global`.
 * https://dlang.org/spec/version.html#predefined-versions
 *
 * Needs to be run after all arguments parsing (command line, DFLAGS environment
 * variable and config file) in order to add final flags (such as `X86_64` or
 * the `CRuntime` used).
 */
void addDefaultVersionIdentifiers()
{
    VersionCondition.addPredefinedGlobalIdent("DigitalMars");
    static if (TARGET.Windows)
    {
        VersionCondition.addPredefinedGlobalIdent("Windows");
        global.params.isWindows = true;
    }
    else static if (TARGET.Linux)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("linux");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        global.params.isLinux = true;
    }
    else static if (TARGET.OSX)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("OSX");
        global.params.isOSX = true;
        // For legacy compatibility
        VersionCondition.addPredefinedGlobalIdent("darwin");
    }
    else static if (TARGET.FreeBSD)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("FreeBSD");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        global.params.isFreeBSD = true;
    }
    else static if (TARGET.OpenBSD)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("OpenBSD");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        global.params.isOpenBSD = true;
    }
    else static if (TARGET.DragonFlyBSD)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("DragonFlyBSD");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        global.params.isDragonFlyBSD = true;
    }
    else static if (TARGET.Solaris)
    {
        VersionCondition.addPredefinedGlobalIdent("Posix");
        VersionCondition.addPredefinedGlobalIdent("Solaris");
        VersionCondition.addPredefinedGlobalIdent("ELFv1");
        global.params.isSolaris = true;
    }
    else
    {
        static assert(0, "fix this");
    }
    VersionCondition.addPredefinedGlobalIdent("LittleEndian");
    VersionCondition.addPredefinedGlobalIdent("D_Version2");
    VersionCondition.addPredefinedGlobalIdent("all");

    if (global.params.cpu >= CPU.sse2)
    {
        VersionCondition.addPredefinedGlobalIdent("D_SIMD");
        if (global.params.cpu >= CPU.avx)
            VersionCondition.addPredefinedGlobalIdent("D_AVX");
        if (global.params.cpu >= CPU.avx2)
            VersionCondition.addPredefinedGlobalIdent("D_AVX2");
    }

    if (global.params.is64bit)
    {
        VersionCondition.addPredefinedGlobalIdent("D_InlineAsm_X86_64");
        VersionCondition.addPredefinedGlobalIdent("X86_64");
        static if (TARGET.Windows)
        {
            VersionCondition.addPredefinedGlobalIdent("Win64");
        }
    }
    else
    {
        VersionCondition.addPredefinedGlobalIdent("D_InlineAsm"); //legacy
        VersionCondition.addPredefinedGlobalIdent("D_InlineAsm_X86");
        VersionCondition.addPredefinedGlobalIdent("X86");
        static if (TARGET.Windows)
        {
            VersionCondition.addPredefinedGlobalIdent("Win32");
        }
    }
    static if (TARGET.Windows)
    {
        if (global.params.mscoff)
            VersionCondition.addPredefinedGlobalIdent("CRuntime_Microsoft");
        else
            VersionCondition.addPredefinedGlobalIdent("CRuntime_DigitalMars");
    }
    else static if (TARGET.Linux)
    {
        VersionCondition.addPredefinedGlobalIdent("CRuntime_Glibc");
    }

    if (global.params.isLP64)
        VersionCondition.addPredefinedGlobalIdent("D_LP64");
    if (global.params.doDocComments)
        VersionCondition.addPredefinedGlobalIdent("D_Ddoc");
    if (global.params.cov)
        VersionCondition.addPredefinedGlobalIdent("D_Coverage");
    if (global.params.pic)
        VersionCondition.addPredefinedGlobalIdent("D_PIC");
    if (global.params.useUnitTests)
        VersionCondition.addPredefinedGlobalIdent("unittest");
    if (global.params.useAssert == CHECKENABLE.on)
        VersionCondition.addPredefinedGlobalIdent("assert");
    if (global.params.useArrayBounds == CHECKENABLE.off)
        VersionCondition.addPredefinedGlobalIdent("D_NoBoundsChecks");
    if (global.params.betterC)
        VersionCondition.addPredefinedGlobalIdent("D_BetterC");

    VersionCondition.addPredefinedGlobalIdent("D_HardFloat");
}

} // !IN_LLVM

private void printPredefinedVersions()
{
    if (global.params.verbose && global.versionids)
    {
        OutBuffer buf;
        foreach (const str; *global.versionids)
        {
            buf.writeByte(' ');
            buf.writestring(str.toChars());
        }
        message("predefs  %s", buf.peekString());
    }
}


/****************************************
 * Determine the instruction set to be used.
 * Params:
 *      cpu = value set by command line switch
 * Returns:
 *      value to generate code for
 */

version (IN_LLVM) {} else
private CPU setTargetCPU(CPU cpu)
{
    // Determine base line for target
    CPU baseline = CPU.x87;
    if (global.params.is64bit)
        baseline = CPU.sse2;
    else
    {
        static if (TARGET.OSX)
        {
            baseline = CPU.sse2;
        }
    }

    if (baseline < CPU.sse2)
        return baseline;        // can't support other instruction sets

    switch (cpu)
    {
        case CPU.baseline:
            cpu = baseline;
            break;

        case CPU.native:
        {
            import core.cpuid;
            cpu = baseline;
            if (core.cpuid.avx2)
                cpu = CPU.avx2;
            else if (core.cpuid.avx)
                cpu = CPU.avx;
            break;
        }

        default:
            break;
    }
    return cpu;
}


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

version (IN_LLVM) {} else
private bool parseCommandLine(const ref Strings arguments, const size_t argc, ref Param params, ref Strings files)
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
        const(char)[] arg = p[0 .. strlen(p)];
        if (*p == '-')
        {
            if (arg == "-allinst")               // https://dlang.org/dmd.html#switch-allinst
                params.allInst = true;
            else if (arg == "-de")               // https://dlang.org/dmd.html#switch-de
                params.useDeprecated = 0;
            else if (arg == "-d")                // https://dlang.org/dmd.html#switch-d
                params.useDeprecated = 1;
            else if (arg == "-dw")               // https://dlang.org/dmd.html#switch-dw
                params.useDeprecated = 2;
            else if (arg == "-c")                // https://dlang.org/dmd.html#switch-c
                params.link = false;
            else if (startsWith(p + 1, "color")) // https://dlang.org/dmd.html#switch-color
            {
                params.color = true;
                // Parse:
                //      -color
                //      -color=on|off
                if (p[6] == '=')
                {
                    if (strcmp(p + 7, "off") == 0)
                        params.color = false;
                    else if (strcmp(p + 7, "on") != 0)
                        goto Lerror;
                }
                else if (p[6])
                    goto Lerror;
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
                        goto Lerror;
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
                    params.pic = 1;
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
            else if (arg == "-g") // https://dlang.org/dmd.html#switch-g
                params.symdebug = 1;
            else if (arg == "-gc")  // https://dlang.org/dmd.html#switch-gc
            {
                Loc loc;
                deprecation(loc, "use -g instead of -gc");
                params.symdebug = 2;
            }
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
            else if (arg == "-gt")
            {
                error("use -profile instead of -gt");
                params.trace = true;
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
                    params.mscrtlib = p + 10;
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
                        goto Lerror;
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
                else
                    goto Lerror;
            }
            else if (startsWith(p + 1, "mcpu")) // https://dlang.org/dmd.html#switch-mcpu
            {
                // Parse:
                //      -mcpu=identifier
                if (p[5] == '=')
                {
                    if (strcmp(p + 6, "?") == 0)
                    {
                        params.mcpuUsage = true;
                        return false;
                    }
                    else if (Identifier.isValidIdentifier(p + 6))
                    {
                        const ident = p + 6;
                        switch (ident[0 .. strlen(ident)])
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
                            goto Lerror;
                        }
                    }
                    else
                        goto Lerror;
                }
                else
                    goto Lerror;
            }
            else if (startsWith(p + 1, "transition") ) // https://dlang.org/dmd.html#switch-transition
            {
                // Parse:
                //      -transition=number
                if (p[11] == '=')
                {
                    if (strcmp(p + 12, "?") == 0)
                    {
                        params.transitionUsage = true;
                        return false;
                    }
                    if (isdigit(cast(char)p[12]))
                    {
                        const num = parseDigits(p + 12, int.max);
                        if (num == uint.max)
                            goto Lerror;

                        string generateTransitionsNumbers()
                        {
                            import dmd.cli : Usage;
                            string buf;
                            foreach (t; Usage.transitions)
                            {
                                if (t.bugzillaNumber !is null)
                                    buf ~= `case `~t.bugzillaNumber~`: params.`~t.paramName~` = true;break;`;
                            }
                            return buf;
                        }

                        // Bugzilla issue number
                        switch (num)
                        {
                        mixin(generateTransitionsNumbers());
                        default:
                            goto Lerror;
                        }
                    }
                    else if (Identifier.isValidIdentifier(p + 12))
                    {
                        string generateTransitionsText()
                        {
                            import dmd.cli : Usage;
                            string buf = `case "all":`;
                            foreach (t; Usage.transitions)
                                buf ~= `params.`~t.paramName~` = true;`;
                            buf ~= "break;";

                            foreach (t; Usage.transitions)
                            {
                                buf ~= `case "`~t.name~`": params.`~t.paramName~` = true;break;`;
                            }
                            return buf;
                        }
                        const ident = p + 12;
                        switch (ident[0 .. strlen(ident)])
                        {
                        mixin(generateTransitionsText());
                        default:
                            goto Lerror;
                        }
                    }
                    else
                        goto Lerror;
                }
                else
                    goto Lerror;
            }
            else if (arg == "-w")   // https://dlang.org/dmd.html#switch-w
                params.warnings = 1;
            else if (arg == "-wi")  // https://dlang.org/dmd.html#switch-wi
                params.warnings = 2;
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
                    params.objdir = path;
                    break;
                case 'f':                       // https://dlang.org/dmd.html#switch-of
                    if (!p[3])
                        goto Lnoarg;
                    path = p + 3 + (p[3] == '=');
                    version (Windows)
                    {
                        path = toWinPath(path);
                    }
                    params.objname = path;
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
                    params.hdrdir = p + 3 + (p[3] == '=');
                    break;
                case 'f':               // https://dlang.org/dmd.html#switch-Hf
                    if (!p[3])
                        goto Lnoarg;
                    params.hdrname = p + 3 + (p[3] == '=');
                    break;
                case 0:
                    break;
                default:
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
                    params.jsonfilename = p + 3 + (p[3] == '=');
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
            else if (arg == "-property")
                params.enforcePropertySyntax = true;
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
                params.useArrayBounds = CHECKENABLE.off;
            }
            else if (startsWith(p + 1, "boundscheck")) // https://dlang.org/dmd.html#switch-boundscheck
            {
                // Parse:
                //      -boundscheck=[on|safeonly|off]
                if (p[12] == '=')
                {
                    if (strcmp(p + 13, "on") == 0)
                    {
                        params.useArrayBounds = CHECKENABLE.on;
                    }
                    else if (strcmp(p + 13, "safeonly") == 0)
                    {
                        params.useArrayBounds = CHECKENABLE.safeonly;
                    }
                    else if (strcmp(p + 13, "off") == 0)
                    {
                        params.useArrayBounds = CHECKENABLE.off;
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
                    if (!params.modFileAliasStrings)
                        params.modFileAliasStrings = new Strings();
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
            }
            else if (startsWith(p + 1, "defaultlib="))   // https://dlang.org/dmd.html#switch-defaultlib
            {
                params.defaultlibname = p + 1 + 11;
            }
            else if (startsWith(p + 1, "debuglib="))     // https://dlang.org/dmd.html#switch-debuglib
            {
                params.debuglibname = p + 1 + 9;
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
                    params.moduleDepsFile = p + 1 + 5;
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
        else
        {
            static if (TARGET.Windows)
            {
                const(char)* ext = FileName.ext(p);
                if (ext && FileName.compare(ext, "exe") == 0)
                {
                    params.objname = p;
                    continue;
                }
                if (arg == "/?")
                {
                    params.usage = true;
                    return false;
                }
            }
            files.push(p);
        }
    }
    return errors;
}


// IN_LLVM: `private` replaced by `extern(C++)`
extern(C++) __gshared bool includeImports = false;
// array of module patterns used to include/exclude imported modules
extern(C++) __gshared Array!(const(char)*) includeModulePatterns;
private __gshared Modules compiledImports;
private extern(C++) bool marsOnImport(Module m)
{
    if (includeImports)
    {
        Identifiers empty;
        if (includeImportedModuleCheck(ModuleComponentRange(
            (m.md && m.md.packages) ? m.md.packages : &empty, m.ident, m.isPackageFile)))
        {
            if (global.params.verbose)
                message("compileimport (%s)", m.srcfile.toChars);
            compiledImports.push(m);
            return true; // this import will be compiled
        }
    }
    return false; // this import will not be compiled
}

// A range of component identifiers for a module
private struct ModuleComponentRange
{
    Identifiers* packages;
    Identifier name;
    bool isPackageFile;
    size_t index;
    @property auto totalLength() const { return packages.dim + 1 + (isPackageFile ? 1 : 0); }

    @property auto empty() { return index >= totalLength(); }
    @property auto front() const
    {
        if (index < packages.dim)
            return (*packages)[index];
        if (index == packages.dim)
            return name;
        else
            return Identifier.idPool("package");
    }
    void popFront() { index++; }
}

/*
 * Determines if the given module should be included in the compilation.
 * Returns:
 *  True if the given module should be included in the compilation.
 */
private bool includeImportedModuleCheck(ModuleComponentRange components)
    in { assert(includeImports); } body
{
    createMatchNodes();
    size_t nodeIndex = 0;
    while (nodeIndex < matchNodes.dim)
    {
        //printf("matcher ");printMatcher(nodeIndex);printf("\n");
        auto info = matchNodes[nodeIndex++];
        if (info.depth <= components.totalLength())
        {
            size_t nodeOffset = 0;
            for (auto range = components;;range.popFront())
            {
                if (range.empty || nodeOffset >= info.depth)
                {
                    // MATCH
                    //printf("matcher ");printMatcher(nodeIndex - 1);
                    //printf(" MATCHES module '");components.print();printf("'\n");
                    return !info.isExclude;
                }
                if (!range.front.equals(matchNodes[nodeIndex + nodeOffset].id))
                {
                    break;
                }
                nodeOffset++;
            }
        }
        //printf("matcher ");printMatcher(nodeIndex-1);
        //printf(" does not match module '");components.print();printf("'\n");
        nodeIndex += info.depth;
    }
    assert(nodeIndex == matchNodes.dim, "code bug");
    return includeByDefault;
}

// Matching module names is done with an array of matcher nodes.
// The nodes are sorted by "component depth" from largest to smallest
// so that the first match is always the longest (best) match.
private struct MatcherNode
{
    union
    {
        struct
        {
            ushort depth;
            bool isExclude;
        }
        Identifier id;
    }
    this(Identifier id) { this.id = id; }
    this(bool isExclude, ushort depth)
    {
        this.depth = depth;
        this.isExclude = isExclude;
    }
}

/*
 * $(D includeByDefault) determines whether to include/exclude modules when they don't
 * match any pattern. This setting changes depending on if the user provided any "inclusive" module
 * patterns. When a single "inclusive" module pattern is given, it likely means the user only
 * intends to include modules they've "included", however, if no module patterns are given or they
 * are all "exclusive", then it is likely they intend to include everything except modules
 * that have been excluded. i.e.
 * ---
 * -i=-foo // include everything except modules that match "foo*"
 * -i=foo  // only include modules that match "foo*" (exclude everything else)
 * ---
 * Note that this default behavior can be overriden using the '.' module pattern. i.e.
 * ---
 * -i=-foo,-.  // this excludes everything
 * -i=foo,.    // this includes everything except the default exclusions (-std,-core,-etc.-object)
 * ---
*/
private __gshared bool includeByDefault = true;
private __gshared Array!MatcherNode matchNodes;

/*
 * Creates the global list of match nodes used to match module names
 * given strings provided by the -i commmand line option.
 */
private void createMatchNodes()
{
    static size_t findSortedIndexToAddForDepth(size_t depth)
    {
        size_t index = 0;
        while (index < matchNodes.dim)
        {
            auto info = matchNodes[index];
            if (depth > info.depth)
                break;
            index += 1 + info.depth;
        }
        return index;
    }

    if (matchNodes.dim == 0)
    {
        foreach (modulePattern; includeModulePatterns)
        {
            auto depth = parseModulePatternDepth(modulePattern);
            auto entryIndex = findSortedIndexToAddForDepth(depth);
            matchNodes.split(entryIndex, depth + 1);
            parseModulePattern(modulePattern, &matchNodes[entryIndex], depth);
            // if at least 1 "include pattern" is given, then it is assumed the
            // user only wants to include modules that were explicitly given, which
            // changes the default behavior from inclusion to exclusion.
            if (includeByDefault && !matchNodes[entryIndex].isExclude)
            {
                //printf("Matcher: found 'include pattern', switching default behavior to exclusion\n");
                includeByDefault = false;
            }
        }

        // Add the default 1 depth matchers
        MatcherNode[8] defaultDepth1MatchNodes = [
            MatcherNode(true, 1), MatcherNode(Id.std),
            MatcherNode(true, 1), MatcherNode(Id.core),
            MatcherNode(true, 1), MatcherNode(Id.etc),
            MatcherNode(true, 1), MatcherNode(Id.object),
        ];
        {
            auto index = findSortedIndexToAddForDepth(1);
            matchNodes.split(index, defaultDepth1MatchNodes.length);
            matchNodes.data[index .. index + defaultDepth1MatchNodes.length] = defaultDepth1MatchNodes[];
        }
    }
}

/*
 * Determines the depth of the given module pattern.
 * Params:
 *  modulePattern = The module pattern to determine the depth of.
 * Returns:
 *  The component depth of the given module pattern.
 */
private ushort parseModulePatternDepth(const(char)* modulePattern)
{
    if (modulePattern[0] == '-')
        modulePattern++;

    // handle special case
    if (modulePattern[0] == '.' && modulePattern[1] == '\0')
        return 0;

    ushort depth = 1;
    for (;; modulePattern++)
    {
        auto c = *modulePattern;
        if (c == '.')
            depth++;
        if (c == '\0')
            return depth;
    }
}
unittest
{
    assert(".".parseModulePatternDepth == 0);
    assert("-.".parseModulePatternDepth == 0);
    assert("abc".parseModulePatternDepth == 1);
    assert("-abc".parseModulePatternDepth == 1);
    assert("abc.foo".parseModulePatternDepth == 2);
    assert("-abc.foo".parseModulePatternDepth == 2);
}

/*
 * Parses a 'module pattern', which is the "include import" components
 * given on the command line, i.e. "-i=<module_pattern>,<module_pattern>,...".
 * Params:
 *  modulePattern = The module pattern to parse.
 *  dst = the data structure to save the parsed module pattern to.
 *  depth = the depth of the module pattern previously retrieved from $(D parseModulePatternDepth).
 */
private void parseModulePattern(const(char)* modulePattern, MatcherNode* dst, ushort depth)
{
    bool isExclude = false;
    if (modulePattern[0] == '-')
    {
        isExclude = true;
        modulePattern++;
    }

    *dst = MatcherNode(isExclude, depth);
    dst++;

    // Create and add identifiers for each component in the modulePattern
    if (depth > 0)
    {
        auto idStart = modulePattern;
        auto lastNode = dst + depth - 1;
        for (; dst < lastNode; dst++)
        {
            for (;; modulePattern++)
            {
                if (*modulePattern == '.')
                {
                    assert(modulePattern > idStart, "empty module pattern");
                    *dst = MatcherNode(Identifier.idPool(idStart, cast(uint)(modulePattern - idStart)));
                    modulePattern++;
                    idStart = modulePattern;
                    break;
                }
            }
        }
        for (;; modulePattern++)
        {
            if (*modulePattern == '\0')
            {
                assert(modulePattern > idStart, "empty module pattern");
                *lastNode = MatcherNode(Identifier.idPool(idStart, cast(uint)(modulePattern - idStart)));
                break;
            }
        }
    }
}
