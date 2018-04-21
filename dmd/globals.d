/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/globals.d, _globals.d)
 * Documentation:  https://dlang.org/phobos/dmd_globals.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/globals.d
 */

module dmd.globals;

import core.stdc.stdint;
import dmd.root.array;
import dmd.root.filename;
import dmd.root.outbuffer;
import dmd.compiler;
import dmd.identifier;

template xversion(string s)
{
    enum xversion = mixin(`{ version (` ~ s ~ `) return true; else return false; }`)();
}

enum IN_LLVM    = xversion!`IN_LLVM`;
enum IN_LLVM_MSVC = xversion!`IN_LLVM_MSVC`;

enum TARGET : bool
{
    Linux        = xversion!`linux`,
    OSX          = xversion!`OSX`,
    FreeBSD      = xversion!`FreeBSD`,
    OpenBSD      = xversion!`OpenBSD`,
    Solaris      = xversion!`Solaris`,
    Windows      = xversion!`Windows`,
    DragonFlyBSD = xversion!`DragonFlyBSD`,
}

version(IN_LLVM)
{
    enum OUTPUTFLAG : int
    {
        OUTPUTFLAGno,
        OUTPUTFLAGdefault, // for the .o default
        OUTPUTFLAGset      // for -output
    }
    alias OUTPUTFLAGno      = OUTPUTFLAG.OUTPUTFLAGno;
    alias OUTPUTFLAGdefault = OUTPUTFLAG.OUTPUTFLAGdefault;
    alias OUTPUTFLAGset     = OUTPUTFLAG.OUTPUTFLAGset;
}

enum CHECKENABLE : ubyte
{
    _default,     // initial value
    off,          // never do checking
    on,           // always do checking
    safeonly,     // do checking only in @safe functions
}

enum CHECKACTION : ubyte
{
    D,            // call D assert on failure
    C,            // call C assert on failure
}

enum CPU
{
    x87,
    mmx,
    sse,
    sse2,
    sse3,
    ssse3,
    sse4_1,
    sse4_2,
    avx,                // AVX1 instruction set
    avx2,               // AVX2 instruction set
    avx512,             // AVX-512 instruction set

    // Special values that don't survive past the command line processing
    baseline,           // (default) the minimum capability CPU
    native              // the machine the compiler is being run on
}

/**
Each flag represents a field that can be included in the JSON output.

NOTE: set type to uint so its size matches C++ unsigned type
*/
enum JsonFieldFlags : uint
{
    none         = 0,
    compilerInfo = (1 << 0),
    buildInfo    = (1 << 1),
    modules      = (1 << 2),
    semantics    = (1 << 3),
}

// Put command line switches in here
struct Param
{
    bool obj = true;        // write object file
    bool link = true;       // perform link
    bool dll;               // generate shared dynamic library
    bool lib;               // write library file instead of object file(s)
    bool multiobj;          // break one object file into multiple ones
    bool oneobj;            // write one object file instead of multiple ones
    bool trace;             // insert profiling hooks
    bool tracegc;           // instrument calls to 'new'
    bool verbose;           // verbose compile
    bool vcg_ast;           // write-out codegen-ast
    bool showColumns;       // print character (column) numbers in diagnostics
    bool vtls;              // identify thread local variables
    bool vgc;               // identify gc usage
    bool vfield;            // identify non-mutable field variables
    bool vcomplex;          // identify complex/imaginary type usage
    ubyte symdebug;         // insert debug symbolic information
    bool symdebugref;       // insert debug information for all referenced types, too
    bool alwaysframe;       // always emit standard stack frame
    bool optimize;          // run optimizer
    bool map;               // generate linker .map file
    bool is64bit = (size_t.sizeof == 8);  // generate 64 bit code; true by default for 64 bit dmd
    bool isLP64;            // generate code for LP64
    bool isLinux;           // generate code for linux
    bool isOSX;             // generate code for Mac OSX
    bool isWindows;         // generate code for Windows
    bool isFreeBSD;         // generate code for FreeBSD
    bool isOpenBSD;         // generate code for OpenBSD
    bool isDragonFlyBSD;    // generate code for DragonFlyBSD
    bool isSolaris;         // generate code for Solaris
    bool hasObjectiveC;     // target supports Objective-C
    bool mscoff = false;    // for Win32: write MsCoff object files instead of OMF
    // 0: don't allow use of deprecated features
    // 1: silently allow use of deprecated features
    // 2: warn about the use of deprecated features
    byte useDeprecated = 2;
    bool useInvariants = true;  // generate class invariant checks
    bool useIn = true;          // generate precondition checks
    bool useOut = true;         // generate postcondition checks
    bool stackstomp;            // add stack stomping code
    bool useUnitTests;          // generate unittest code
    bool useInline = false;     // inline expand functions
    bool useDIP25;          // implement http://wiki.dlang.org/DIP25
    bool release;           // build release version
    bool preservePaths;     // true means don't strip path from source file
    // 0: disable warnings
    // 1: warnings as errors
    // 2: informational warnings (no errors)
    byte warnings;
    bool pic;               // generate position-independent-code for shared libs
    bool color = true;      // use ANSI colors in console output
    bool cov;               // generate code coverage data
    ubyte covPercent;       // 0..100 code coverage percentage required
    bool nofloat;           // code should not pull in floating point support
    bool ignoreUnsupportedPragmas;  // rather than error on them
    bool enforcePropertySyntax;
    bool useModuleInfo = true;   // generate runtime module information
    bool useTypeInfo = true;     // generate runtime type information
    bool useExceptions = true;   // support exception handling
    bool betterC;           // be a "better C" compiler; no dependency on D runtime
    bool addMain;           // add a default main() function
    bool allInst;           // generate code for all template instantiations
    bool check10378;        // check for issues transitioning to 10738
    bool bug10378;          // use pre- https://issues.dlang.org/show_bug.cgi?id=10378 search strategy
    bool fix16997;          // fix integral promotions for unary + - ~ operators
                            // https://issues.dlang.org/show_bug.cgi?id=16997
    bool vsafe;             // use enhanced @safe checking
    bool ehnogc;            // use @nogc exception handling
    /** The --transition=safe switch should only be used to show code with
     * silent semantics changes related to @safe improvements.  It should not be
     * used to hide a feature that will have to go through deprecate-then-error
     * before becoming default.
     */

    bool showGaggedErrors;  // print gagged errors anyway
    bool manual;            // open browser on compiler manual
    bool usage;             // print usage and exit
    bool mcpuUsage;         // print help on -mcpu switch
    bool transitionUsage;   // print help on -transition switch
    bool logo;              // print compiler logo

    CPU cpu = CPU.baseline; // CPU instruction set to target

    CHECKENABLE useArrayBounds = CHECKENABLE._default;  // when to generate code for array bounds checks
    CHECKENABLE useAssert      = CHECKENABLE._default;  // when to generate code for assert()'s
    CHECKENABLE useSwitchError = CHECKENABLE._default;  // check for switches without a default
    CHECKACTION checkAction;       // action to take when bounds, asserts or switch defaults are violated

    uint errorLimit = 20;

    const(char)* argv0;                 // program name
    Array!(const(char)*)* modFileAliasStrings; // array of char*'s of -I module filename alias strings
    Array!(const(char)*)* imppath;      // array of char*'s of where to look for import modules
    Array!(const(char)*)* fileImppath;  // array of char*'s of where to look for file import modules
    const(char)* objdir;                // .obj/.lib file output directory
    const(char)* objname;               // .obj file output name
    const(char)* libname;               // .lib file output name

    bool doDocComments;                 // process embedded documentation comments
    const(char)* docdir;                // write documentation file to docdir directory
    const(char)* docname;               // write documentation file to docname
    Array!(const(char)*) ddocfiles;     // macro include files for Ddoc

    bool doHdrGeneration;               // process embedded documentation comments
    const(char)* hdrdir;                // write 'header' file to docdir directory
    const(char)* hdrname;               // write 'header' file to docname
    bool hdrStripPlainFunctions = true; // strip the bodies of plain (non-template) functions

    bool doJsonGeneration;              // write JSON file
    const(char)* jsonfilename;          // write JSON file to jsonfilename
    JsonFieldFlags jsonFieldFlags;      // JSON field flags to include

    uint debuglevel;                    // debug level
    Array!(const(char)*)* debugids;     // debug identifiers

    uint versionlevel;                  // version level
    Array!(const(char)*)* versionids;   // version identifiers

    const(char)* defaultlibname;        // default library for non-debug builds
    const(char)* debuglibname;          // default library for debug builds
    const(char)* mscrtlib;              // MS C runtime library

    const(char)* moduleDepsFile;        // filename for deps output
    OutBuffer* moduleDeps;              // contents to be written to deps file

    // Hidden debug switches
    bool debugb;
    bool debugc;
    bool debugf;
    bool debugr;
    bool debugx;
    bool debugy;

    bool run; // run resulting executable
    Strings runargs; // arguments for executable

    // Linker stuff
    Array!(const(char)*) objfiles;
    Array!(const(char)*) linkswitches;
    Array!(const(char)*) libfiles;
    Array!(const(char)*) dllfiles;
    const(char)* deffile;
    const(char)* resfile;
    const(char)* exefile;
    const(char)* mapfile;

    version(IN_LLVM)
    {
        Array!(const(char)*) bitcodeFiles; // LLVM bitcode files passed on cmdline

        uint nestedTmpl; // maximum nested template instantiations

        // LDC stuff
        OUTPUTFLAG output_ll;
        OUTPUTFLAG output_bc;
        OUTPUTFLAG output_s;
        OUTPUTFLAG output_o;
        bool useInlineAsm;
        bool verbose_cg;
        bool fullyQualifiedObjectFiles;
        bool cleanupObjectFiles;

        // Profile-guided optimization:
        const(char)* datafileInstrProf; // Either the input or output file for PGO data

        // target stuff
        const(void)* targetTriple; // const llvm::Triple*

        // Codegen cl options
        bool disableRedZone;
        uint dwarfVersion;

        uint hashThreshold; // MD5 hash symbols larger than this threshold (0 = no hashing)

        bool outputSourceLocations; // if true, output line tables.
    }
}

alias structalign_t = uint;

// magic value means "match whatever the underlying C compiler does"
// other values are all powers of 2
enum STRUCTALIGN_DEFAULT = (cast(structalign_t)~0);

struct Global
{
    const(char)* inifilename;
    const(char)* mars_ext = "d";
    const(char)* obj_ext;
    version(IN_LLVM)
    {
        const(char)* ll_ext;
        const(char)* bc_ext;
        const(char)* s_ext;
        const(char)* ldc_version;
        const(char)* llvm_version;

        bool gaggedForInlining; // Set for functionSemantic3 for external inlining candidates
    }
    const(char)* lib_ext;
    const(char)* dll_ext;
    const(char)* doc_ext = "html";      // for Ddoc generated files
    const(char)* ddoc_ext = "ddoc";     // for Ddoc macro include files
    const(char)* hdr_ext = "di";        // for D 'header' import files
    const(char)* json_ext = "json";     // for JSON files
    const(char)* map_ext = "map";       // for .map files
    bool run_noext;                     // allow -run sources without extensions.

    const(char)* copyright = "Copyright (C) 1999-2018 by The D Language Foundation, All Rights Reserved";
    const(char)* written = "written by Walter Bright";
    const(char)* main_d = "__main.d";   // dummy filename for dummy main()
    Array!(const(char)*)* path;         // Array of char*'s which form the import lookup path
    Array!(const(char)*)* filePath;     // Array of char*'s which form the file import lookup path

    const(char)* _version;

    Compiler compiler;
    Param params;
    uint errors;            // number of errors reported so far
    uint warnings;          // number of warnings reported so far
    uint gag;               // !=0 means gag reporting of errors & warnings
    uint gaggedErrors;      // number of errors reported while gagged

    void* console;         // opaque pointer to console for controlling text attributes

    Array!Identifier* versionids;    // command line versions and predefined versions
    Array!Identifier* debugids;      // command line debug versions and predefined versions

    /* Start gagging. Return the current number of gagged errors
     */
    extern (C++) uint startGagging()
    {
        ++gag;
        return gaggedErrors;
    }

    /* End gagging, restoring the old gagged state.
     * Return true if errors occurred while gagged.
     */
    extern (C++) bool endGagging(uint oldGagged)
    {
        bool anyErrs = (gaggedErrors != oldGagged);
        --gag;
        // Restore the original state of gagged errors; set total errors
        // to be original errors + new ungagged errors.
        errors -= (gaggedErrors - oldGagged);
        gaggedErrors = oldGagged;
        return anyErrs;
    }

    /*  Increment the error count to record that an error
     *  has occurred in the current context. An error message
     *  may or may not have been printed.
     */
    extern (C++) void increaseErrorCount()
    {
        if (gag)
            ++gaggedErrors;
        ++errors;
    }

    extern (C++) void _init()
    {
version(IN_LLVM)
{
        obj_ext = "o";
        ll_ext  = "ll";
        bc_ext  = "bc";
        s_ext   = "s";
}
else
{
        static if (TARGET.Windows)
        {
            obj_ext = "obj";
        }
        else static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
        {
            obj_ext = "o";
        }
        else
        {
            static assert(0, "fix this");
        }
        static if (TARGET.Windows)
        {
            lib_ext = "lib";
        }
        else static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
        {
            lib_ext = "a";
        }
        else
        {
            static assert(0, "fix this");
        }
        static if (TARGET.Windows)
        {
            dll_ext = "dll";
        }
        else static if (TARGET.Linux || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
        {
            dll_ext = "so";
        }
        else static if (TARGET.OSX)
        {
            dll_ext = "dylib";
        }
        else
        {
            static assert(0, "fix this");
        }
        static if (TARGET.Windows)
        {
            run_noext = false;
        }
        else static if (TARGET.Linux || TARGET.OSX || TARGET.FreeBSD || TARGET.OpenBSD || TARGET.Solaris || TARGET.DragonFlyBSD)
        {
            // Allow 'script' D source files to have no extension.
            run_noext = true;
        }
        else
        {
            static assert(0, "fix this");
        }
}
version(IN_LLVM)
{
        compiler.vendor = "LDC";
}
else
{
        _version = (import("VERSION") ~ '\0').ptr;
        compiler.vendor = "Digital Mars D";
}
    }
}

// Because int64_t and friends may be any integral type of the
// correct size, we have to explicitly ask for the correct
// integer type to get the correct mangling with dmd

// Be careful not to care about sign when using dinteger_t
// use this instead of integer_t to
// avoid conflicts with system #include's
alias dinteger_t = ulong;
// Signed and unsigned variants
alias sinteger_t = long;
alias uinteger_t = ulong;

alias d_int8 = int8_t;
alias d_uns8 = uint8_t;
alias d_int16 = int16_t;
alias d_uns16 = uint16_t;
alias d_int32 = int32_t;
alias d_uns32 = uint32_t;
alias d_int64 = int64_t;
alias d_uns64 = uint64_t;

// file location
struct Loc
{
    const(char)* filename; // either absolute or relative to cwd
    uint linnum;
    uint charnum;

    static immutable Loc initial;       /// use for default initialization of const ref Loc's

nothrow:
    extern (D) this(const(char)* filename, uint linnum, uint charnum)
    {
        this.linnum = linnum;
        this.charnum = charnum;
        this.filename = filename;
    }

    extern (C++) const(char)* toChars() const
    {
        OutBuffer buf;
        if (filename)
        {
            buf.writestring(filename);
        }
        if (linnum)
        {
            buf.writeByte('(');
            buf.print(linnum);
            if (global.params.showColumns && charnum)
            {
                buf.writeByte(',');
                buf.print(charnum);
            }
            buf.writeByte(')');
        }
        return buf.extractString();
    }

    extern (C++) bool equals(ref const(Loc) loc) const
    {
        return (!global.params.showColumns || charnum == loc.charnum) &&
               linnum == loc.linnum &&
               FileName.equals(filename, loc.filename);
    }

    /******************
     * Returns:
     *   true if Loc has been set to other than the default initialization
     */
    bool isValid() const pure
    {
        return filename !is null;
    }
}

enum LINK : int
{
    default_,
    d,
    c,
    cpp,
    windows,
    pascal,
    objc,
    system,
}

enum CPPMANGLE : int
{
    def,
    asStruct,
    asClass,
}

enum MATCH : int
{
    nomatch,   // no match
    convert,   // match with conversions
    constant,  // match with conversion to const
    exact,     // exact match
}

enum PINLINE : int
{
    default_,     // as specified on the command line
    never,   // never inline
    always,  // always inline
}

alias StorageClass = uinteger_t;

extern (C++) __gshared Global global;
