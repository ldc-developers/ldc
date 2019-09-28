/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
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
import dmd.identifier;

template xversion(string s)
{
    enum xversion = mixin(`{ version (` ~ s ~ `) return true; else return false; }`)();
}

enum IN_LLVM = xversion!`IN_LLVM`;

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

version (IN_LLVM)
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

enum DiagnosticReporting : ubyte
{
    error,        // generate an error
    inform,       // generate a warning
    off,          // disable diagnostic
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
    halt,         // cause program halt on failure
    context,      // call D assert with the error context on failure
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

enum PIC : ubyte
{
    fixed,              /// located at a specific address
    pic,                /// Position Independent Code
    pie,                /// Position Independent Executable
}

/**
Each flag represents a field that can be included in the JSON output.

NOTE: set type to uint so its size matches C++ unsigned type
*/
enum JsonFieldFlags : int // IN_LLVM: changed from uint to int due to https://issues.dlang.org/show_bug.cgi?id=19658
{
    none         = 0,
    compilerInfo = (1 << 0),
    buildInfo    = (1 << 1),
    modules      = (1 << 2),
    semantics    = (1 << 3),
}

enum CppStdRevision : uint
{
    cpp98 = 199711,
    cpp11 = 201103,
    cpp14 = 201402,
    cpp17 = 201703
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
    DiagnosticReporting useDeprecated = DiagnosticReporting.inform;  // how use of deprecated features are handled
    bool stackstomp;            // add stack stomping code
    bool useUnitTests;          // generate unittest code
    bool useInline = false;     // inline expand functions
    bool useDIP25;          // implement http://wiki.dlang.org/DIP25
    bool noDIP25;           // revert to pre-DIP25 behavior
    bool release;           // build release version
    bool preservePaths;     // true means don't strip path from source file
    DiagnosticReporting warnings = DiagnosticReporting.off;  // how compiler warnings are handled
    PIC pic = PIC.fixed;    // generate fixed, pic or pie code
    bool color;             // use ANSI colors in console output
    bool cov;               // generate code coverage data
    ubyte covPercent;       // 0..100 code coverage percentage required
    bool nofloat;           // code should not pull in floating point support
    bool ignoreUnsupportedPragmas;  // rather than error on them
    bool useModuleInfo = true;   // generate runtime module information
    bool useTypeInfo = true;     // generate runtime type information
    bool useExceptions = true;   // support exception handling
    bool noSharedAccess;         // read/write access to shared memory objects
    bool betterC;           // be a "better C" compiler; no dependency on D runtime
    bool addMain;           // add a default main() function
    bool allInst;           // generate code for all template instantiations
    bool check10378;        // check for issues transitioning to 10738 @@@DEPRECATED@@@ Remove in 2020-05 or later
    bool bug10378;          // use pre- https://issues.dlang.org/show_bug.cgi?id=10378 search strategy  @@@DEPRECATED@@@ Remove in 2020-05 or later
    bool fix16997;          // fix integral promotions for unary + - ~ operators
                            // https://issues.dlang.org/show_bug.cgi?id=16997
    bool fixAliasThis;      // if the current scope has an alias this, check it before searching upper scopes
    /** The --transition=safe switch should only be used to show code with
     * silent semantics changes related to @safe improvements.  It should not be
     * used to hide a feature that will have to go through deprecate-then-error
     * before becoming default.
     */
    bool vsafe;             // use enhanced @safe checking
    bool ehnogc;            // use @nogc exception handling
    bool dtorFields;        // destruct fields of partially constructed objects
                            // https://issues.dlang.org/show_bug.cgi?id=14246
    bool fieldwise;         // do struct equality testing field-wise rather than by memcmp()
    bool rvalueRefParam;    // allow rvalues to be arguments to ref parameters

    CppStdRevision cplusplus = CppStdRevision.cpp98;    // version of C++ standard to support

    bool markdown;          // enable Markdown replacements in Ddoc
    bool vmarkdown;         // list instances of Markdown replacements in Ddoc

    bool showGaggedErrors;  // print gagged errors anyway
    bool printErrorContext;  // print errors with the error context (the error line in the source file)
    bool manual;            // open browser on compiler manual
    bool usage;             // print usage and exit
    bool mcpuUsage;         // print help on -mcpu switch
    bool transitionUsage;   // print help on -transition switch
    bool checkUsage;        // print help on -check switch
    bool checkActionUsage;  // print help on -checkaction switch
    bool revertUsage;       // print help on -revert switch
    bool previewUsage;      // print help on -preview switch
    bool externStdUsage;    // print help on -extern-std switch
    bool logo;              // print compiler logo

    CPU cpu = CPU.baseline; // CPU instruction set to target

    CHECKENABLE useInvariants  = CHECKENABLE._default;  // generate class invariant checks
    CHECKENABLE useIn          = CHECKENABLE._default;  // generate precondition checks
    CHECKENABLE useOut         = CHECKENABLE._default;  // generate postcondition checks
    CHECKENABLE useArrayBounds = CHECKENABLE._default;  // when to generate code for array bounds checks
    CHECKENABLE useAssert      = CHECKENABLE._default;  // when to generate code for assert()'s
    CHECKENABLE useSwitchError = CHECKENABLE._default;  // check for switches without a default
    CHECKENABLE boundscheck    = CHECKENABLE._default;  // state of -boundscheck switch

    CHECKACTION checkAction = CHECKACTION.D; // action to take when bounds, asserts or switch defaults are violated

    uint errorLimit = 20;

    const(char)[] argv0;                // program name
    Array!(const(char)*) modFileAliasStrings; // array of char*'s of -I module filename alias strings
    Array!(const(char)*)* imppath;      // array of char*'s of where to look for import modules
    Array!(const(char)*)* fileImppath;  // array of char*'s of where to look for file import modules
    const(char)[] objdir;                // .obj/.lib file output directory
    const(char)[] objname;               // .obj file output name
    const(char)[] libname;               // .lib file output name

    bool doDocComments;                 // process embedded documentation comments
    const(char)* docdir;                // write documentation file to docdir directory
    const(char)* docname;               // write documentation file to docname
    Array!(const(char)*) ddocfiles;     // macro include files for Ddoc

    bool doHdrGeneration;               // process embedded documentation comments
    const(char)[] hdrdir;                // write 'header' file to docdir directory
    const(char)[] hdrname;               // write 'header' file to docname
    bool hdrStripPlainFunctions = true; // strip the bodies of plain (non-template) functions

    bool doJsonGeneration;              // write JSON file
    const(char)[] jsonfilename;          // write JSON file to jsonfilename
    JsonFieldFlags jsonFieldFlags;      // JSON field flags to include

    OutBuffer* mixinOut;                // write expanded mixins for debugging
    const(char)* mixinFile;             // .mixin file output name
    int mixinLines;                     // Number of lines in writeMixins

    uint debuglevel;                    // debug level
    Array!(const(char)*)* debugids;     // debug identifiers

    uint versionlevel;                  // version level
    Array!(const(char)*)* versionids;   // version identifiers

    const(char)[] defaultlibname;        // default library for non-debug builds
    const(char)[] debuglibname;          // default library for debug builds
    const(char)[] mscrtlib;              // MS C runtime library

    const(char)[] moduleDepsFile;        // filename for deps output
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
    const(char)[] deffile;
    const(char)[] resfile;
    const(char)[] exefile;
    const(char)[] mapfile;

    /* LDC: unused function featuring syntax not supported by ltsmaster
    // generate code for POSIX
    @property bool isPOSIX() scope const pure nothrow @nogc @safe
    out(result) { assert(result || isWindows); }
    do
    {
        return isLinux
            || isOSX
            || isFreeBSD
            || isOpenBSD
            || isDragonFlyBSD
            || isSolaris;
    }
    */

version (IN_LLVM)
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
    bool isUClibcEnvironment;

    // Codegen cl options
    bool disableRedZone;
    uint dwarfVersion;

    uint hashThreshold; // MD5 hash symbols larger than this threshold (0 = no hashing)

    bool outputSourceLocations; // if true, output line tables.
} // IN_LLVM
}

alias structalign_t = uint;

// magic value means "match whatever the underlying C compiler does"
// other values are all powers of 2
enum STRUCTALIGN_DEFAULT = (cast(structalign_t)~0);

struct Global
{
    const(char)[] inifilename;
    string mars_ext = "d";
    const(char)[] obj_ext;
version (IN_LLVM)
{
    const(char)[] ll_ext;
    const(char)[] bc_ext;
    const(char)[] s_ext;
    const(char)[] ldc_version;
    const(char)[] llvm_version;

    bool gaggedForInlining; // Set for functionSemantic3 for external inlining candidates
}
    const(char)[] lib_ext;
    const(char)[] dll_ext;
    string doc_ext = "html";      // for Ddoc generated files
    string ddoc_ext = "ddoc";     // for Ddoc macro include files
    string hdr_ext = "di";        // for D 'header' import files
    string json_ext = "json";     // for JSON files
    string map_ext = "map";       // for .map files
    bool run_noext;                     // allow -run sources without extensions.

    string copyright = "Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved";
    string written = "written by Walter Bright";

    Array!(const(char)*)* path;         // Array of char*'s which form the import lookup path
    Array!(const(char)*)* filePath;     // Array of char*'s which form the file import lookup path

    string _version;
    const(char)[] vendor;    // Compiler backend name

    Param params;
    uint errors;            // number of errors reported so far
    uint warnings;          // number of warnings reported so far
    uint gag;               // !=0 means gag reporting of errors & warnings
    uint gaggedErrors;      // number of errors reported while gagged
    uint gaggedWarnings;    // number of warnings reported while gagged

    void* console;         // opaque pointer to console for controlling text attributes

    Array!Identifier* versionids;    // command line versions and predefined versions
    Array!Identifier* debugids;      // command line debug versions and predefined versions

  nothrow:

    /* Start gagging. Return the current number of gagged errors
     */
    extern (C++) uint startGagging()
    {
        ++gag;
        gaggedWarnings = 0;
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
        static if (!IN_LLVM) _version = import("VERSION") ~ '\0';

        version (MARS)
        {
            vendor = "Digital Mars D";
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
            static if (TARGET.Windows)
            {
                params.mscoff = params.is64bit;
            }

            // -color=auto is the default value
            import dmd.console : Console;
            params.color = Console.detectTerminal();
        }
        else version (IN_GCC)
        {
            vendor = "GNU D";
            obj_ext = "o";
            lib_ext = "a";
            dll_ext = "so";
            run_noext = true;
        }
        else version (IN_LLVM)
        {
            vendor = "LDC";
            obj_ext = "o";
            ll_ext  = "ll";
            bc_ext  = "bc";
            s_ext   = "s";

            import dmd.console : Console;
            params.color = Console.detectTerminal();
        }
    }

    /**
     * Deinitializes the global state of the compiler.
     *
     * This can be used to restore the state set by `_init` to its original
     * state.
     */
    void deinitialize()
    {
        this = this.init;
    }

    /**
    Returns: the version as the number that would be returned for __VERSION__
    */
    extern(C++) uint versionNumber()
    {
        import core.stdc.ctype;
        __gshared uint cached = 0;
        if (cached == 0)
        {
            //
            // parse _version
            //
            uint major = 0;
            uint minor = 0;
            bool point = false;
            for (const(char)* p = _version.ptr + 1;; p++)
            {
                const c = *p;
                if (isdigit(cast(char)c))
                {
                    minor = minor * 10 + c - '0';
                }
                else if (c == '.')
                {
                    if (point)
                        break; // ignore everything after second '.'
                    point = true;
                    major = minor;
                    minor = 0;
                }
                else
                    break;
            }
            cached = major * 1000 + minor;
        }
        return cached;
    }

    /**
    Returns: the final defaultlibname based on the command-line parameters
    */
    const(char)[] finalDefaultlibname() const
    {
        return params.betterC ? null :
            params.symdebug ? params.debuglibname : params.defaultlibname;
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
    extern (D) this(const(char)* filename, uint linnum, uint charnum) pure
    {
        this.linnum = linnum;
        this.charnum = charnum;
        this.filename = filename;
    }

    extern (C++) const(char)* toChars(bool showColumns = global.params.showColumns) const pure nothrow
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
            if (showColumns && charnum)
            {
                buf.writeByte(',');
                buf.print(charnum);
            }
            buf.writeByte(')');
        }
        return buf.extractChars();
    }

    /* Checks for equivalence,
     * a) comparing the filename contents (not the pointer), case-
     *    insensitively on Windows, and
     * b) ignoring charnum if `global.params.showColumns` is false.
     */
    extern (C++) bool equals(ref const(Loc) loc) const
    {
        return (!global.params.showColumns || charnum == loc.charnum) &&
               linnum == loc.linnum &&
               FileName.equals(filename, loc.filename);
    }

    /* opEquals() / toHash() for AA key usage:
     *
     * Compare filename contents (case-sensitively on Windows too), not
     * the pointer - a static foreach loop repeatedly mixing in a mixin
     * may lead to multiple equivalent filenames (`foo.d-mixin-<line>`),
     * e.g., for test/runnable/test18880.d.
     */
    extern (D) bool opEquals(ref const(Loc) loc) const @trusted pure nothrow @nogc
    {
        import core.stdc.string : strcmp;

        return charnum == loc.charnum &&
               linnum == loc.linnum &&
               (filename == loc.filename ||
                (filename && loc.filename && strcmp(filename, loc.filename) == 0));
    }

    extern (D) size_t toHash() const @trusted pure nothrow
    {
        import dmd.utils : toDString;

        auto hash = hashOf(linnum);
        hash = hashOf(charnum, hash);
        hash = hashOf(filename.toDString, hash);
        return hash;
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
