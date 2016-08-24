// Compiler implementation of the D programming language
// Copyright (c) 1999-2015 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt

module ddmd.globals;

import core.stdc.stdint;
import core.stdc.stdio;
import core.stdc.string;
import ddmd.root.array;
import ddmd.root.filename;
import ddmd.root.outbuffer;

template xversion(string s)
{
    enum xversion = mixin(`{ version (` ~ s ~ `) return true; else return false; }`)();
}

private string stripRight(string s)
{
    while (s.length && (s[$ - 1] == ' ' || s[$ - 1] == '\n' || s[$ - 1] == '\r'))
        s = s[0 .. $ - 1];
    return s;
}

enum __linux__      = xversion!`linux`;
enum __APPLE__      = xversion!`OSX`;
enum __FreeBSD__    = xversion!`FreeBSD`;
enum __OpenBSD__    = xversion!`OpenBSD`;
enum __sun          = xversion!`Solaris`;

enum IN_GCC     = xversion!`IN_GCC`;
enum IN_LLVM    = xversion!`IN_LLVM`;
enum IN_LLVM_MSVC = xversion!`IN_LLVM_MSVC`;

enum TARGET_LINUX   = xversion!`linux`;
enum TARGET_OSX     = xversion!`OSX`;
enum TARGET_FREEBSD = xversion!`FreeBSD`;
enum TARGET_OPENBSD = xversion!`OpenBSD`;
enum TARGET_SOLARIS = xversion!`Solaris`;
enum TARGET_WINDOS  = xversion!`Windows`;

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

enum BOUNDSCHECK : int
{
    BOUNDSCHECKdefault,     // initial value
    BOUNDSCHECKoff,         // never do bounds checking
    BOUNDSCHECKon,          // always do bounds checking
    BOUNDSCHECKsafeonly,    // do bounds checking only in @safe functions
}

alias BOUNDSCHECKdefault = BOUNDSCHECK.BOUNDSCHECKdefault;
alias BOUNDSCHECKoff = BOUNDSCHECK.BOUNDSCHECKoff;
alias BOUNDSCHECKon = BOUNDSCHECK.BOUNDSCHECKon;
alias BOUNDSCHECKsafeonly = BOUNDSCHECK.BOUNDSCHECKsafeonly;

// Put command line switches in here
struct Param
{
    bool obj;               // write object file
    bool link;              // perform link
    bool dll;               // generate shared dynamic library
    bool lib;               // write library file instead of object file(s)
    bool multiobj;          // break one object file into multiple ones
    bool oneobj;            // write one object file instead of multiple ones
    bool trace;             // insert profiling hooks
    bool tracegc;           // instrument calls to 'new'
    bool verbose;           // verbose compile
    bool showColumns;       // print character (column) numbers in diagnostics
    bool vtls;              // identify thread local variables
    bool vgc;               // identify gc usage
    bool vfield;            // identify non-mutable field variables
    bool vcomplex;          // identify complex/imaginary type usage
    ubyte symdebug;         // insert debug symbolic information
    bool alwaysframe;       // always emit standard stack frame
    bool optimize;          // run optimizer
    bool map;               // generate linker .map file
    bool is64bit;           // generate 64 bit code
    bool isLP64;            // generate code for LP64
    bool isLinux;           // generate code for linux
    bool isOSX;             // generate code for Mac OSX
    bool isWindows;         // generate code for Windows
    bool isFreeBSD;         // generate code for FreeBSD
    bool isOpenBSD;         // generate code for OpenBSD
    bool isSolaris;         // generate code for Solaris
    bool mscoff;            // for Win32: write COFF object files instead of OMF
    // 0: don't allow use of deprecated features
    // 1: silently allow use of deprecated features
    // 2: warn about the use of deprecated features
    byte useDeprecated;
    bool useAssert;         // generate runtime code for assert()'s
    bool useInvariants;     // generate class invariant checks
    bool useIn;             // generate precondition checks
    bool useOut;            // generate postcondition checks
    bool stackstomp;        // add stack stomping code
    bool useSwitchError;    // check for switches without a default
    bool useUnitTests;      // generate unittest code
    bool useInline;         // inline expand functions
    bool useDIP25;          // implement http://wiki.dlang.org/DIP25
    bool release;           // build release version
    bool preservePaths;     // true means don't strip path from source file
    // 0: disable warnings
    // 1: warnings as errors
    // 2: informational warnings (no errors)
    byte warnings;
    bool pic;               // generate position-independent-code for shared libs
    bool color;             // use ANSI colors in console output
    bool cov;               // generate code coverage data
    ubyte covPercent;       // 0..100 code coverage percentage required
    bool nofloat;           // code should not pull in floating point support
    bool ignoreUnsupportedPragmas;  // rather than error on them
    bool enforcePropertySyntax;
    bool betterC;           // be a "better C" compiler; no dependency on D runtime
    bool addMain;           // add a default main() function
    bool allInst;           // generate code for all template instantiations
    bool dwarfeh;           // generate dwarf eh exception handling
    bool check10378;        // check for issues transitioning to 10738
    bool bug10378;          // use pre-bugzilla 10378 search strategy

    BOUNDSCHECK useArrayBounds;

    const(char)* argv0;                 // program name
    Array!(const(char)*)* imppath;      // array of char*'s of where to look for import modules
    Array!(const(char)*)* fileImppath;  // array of char*'s of where to look for file import modules
    const(char)* objdir;                // .obj/.lib file output directory
    const(char)* objname;               // .obj file output name
    const(char)* libname;               // .lib file output name

    bool doDocComments;                 // process embedded documentation comments
    const(char)* docdir;                // write documentation file to docdir directory
    const(char)* docname;               // write documentation file to docname
    Array!(const(char)*)* ddocfiles;    // macro include files for Ddoc

    bool doHdrGeneration;               // process embedded documentation comments
    const(char)* hdrdir;                // write 'header' file to docdir directory
    const(char)* hdrname;               // write 'header' file to docname

    bool doJsonGeneration;              // write JSON file
    const(char)* jsonfilename;          // write JSON file to jsonfilename

    uint debuglevel;                    // debug level
    Array!(const(char)*)* debugids;     // debug identifiers

    uint versionlevel;                  // version level
    Array!(const(char)*)* versionids;   // version identifiers

    const(char)* defaultlibname;        // default library for non-debug builds
    const(char)* debuglibname;          // default library for debug builds

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
    Array!(const(char)*)* objfiles;
    Array!(const(char)*)* linkswitches;
    Array!(const(char)*)* libfiles;
    Array!(const(char)*)* dllfiles;
    const(char)* deffile;
    const(char)* resfile;
    const(char)* exefile;
    const(char)* mapfile;

    version(IN_LLVM)
    {
        Array!(const(char)*)* bitcodeFiles; // LLVM bitcode files passed on cmdline

        uint nestedTmpl; // maximum nested template instantiations

        // Whether to keep all function bodies in .di file generation or to strip
        // those of plain functions. For DMD, this is govenered by the -inline
        // flag, which does not directly translate to LDC.
        bool hdrKeepAllBodies;

        // LDC stuff
        OUTPUTFLAG output_ll;
        OUTPUTFLAG output_bc;
        OUTPUTFLAG output_s;
        OUTPUTFLAG output_o;
        bool useInlineAsm;
        bool verbose_cg;
        bool hasObjectiveC;

        // Profile-guided optimization:
        bool genInstrProf;             // Whether to generate PGO instrumented code
        const(char)* datafileInstrProf; // Either the input or output file for PGO data

        // target stuff
        const(void)* targetTriple; // const llvm::Triple*

        // Codegen cl options
        bool singleObj;
        bool disableRedZone;

        uint hashThreshold; // MD5 hash symbols larger than this threshold (0 = no hashing)
    }
}

struct Compiler
{
    const(char)* vendor; // Compiler backend name
}

alias structalign_t = uint;

// magic value means "match whatever the underlying C compiler does"
// other values are all powers of 2
enum STRUCTALIGN_DEFAULT = (cast(structalign_t)~0);

struct Global
{
    const(char)* inifilename;
    const(char)* mars_ext;
    const(char)* obj_ext;
    version(IN_LLVM)
    {
        const(char)* obj_ext_alt;
        const(char)* ll_ext;
        const(char)* bc_ext;
        const(char)* s_ext;
        const(char)* ldc_version;
        const(char)* llvm_version;

        bool gaggedForInlining; // Set for functionSemantic3 for external inlining candidates
    }
    const(char)* lib_ext;
    const(char)* dll_ext;
    const(char)* doc_ext;           // for Ddoc generated files
    const(char)* ddoc_ext;          // for Ddoc macro include files
    const(char)* hdr_ext;           // for D 'header' import files
    const(char)* json_ext;          // for JSON files
    const(char)* map_ext;           // for .map files
    bool run_noext;                 // allow -run sources without extensions.

    const(char)* copyright;
    const(char)* written;
    const(char)* main_d;            // dummy filename for dummy main()
    Array!(const(char)*)* path;     // Array of char*'s which form the import lookup path
    Array!(const(char)*)* filePath; // Array of char*'s which form the file import lookup path

    const(char)* _version;

    Compiler compiler;
    Param params;
    uint errors;            // number of errors reported so far
    uint warnings;          // number of warnings reported so far
    FILE* stdmsg;           // where to send verbose messages
    uint gag;               // !=0 means gag reporting of errors & warnings
    uint gaggedErrors;      // number of errors reported while gagged

    uint errorLimit;

    /* Start gagging. Return the current number of gagged errors
     */
    extern (C++) uint startGagging()
    {
        ++gag;
        return gaggedErrors;
    }

    /* End gagging, restoring the old gagged state.
     * Return true if errors occured while gagged.
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
     *  has occured in the current context. An error message
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
        inifilename = null;
        mars_ext = "d";
        hdr_ext = "di";
        doc_ext = "html";
        ddoc_ext = "ddoc";
        json_ext = "json";
        map_ext = "map";
version(IN_LLVM)
{
        ll_ext  = "ll";
        bc_ext  = "bc";
        s_ext   = "s";
        obj_ext = "o";
        obj_ext_alt = "obj";
}
else
{
        static if (TARGET_WINDOS)
        {
            obj_ext = "obj";
        }
        else static if (TARGET_LINUX || TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS)
        {
            obj_ext = "o";
        }
        else
        {
            static assert(0, "fix this");
        }
        static if (TARGET_WINDOS)
        {
            lib_ext = "lib";
        }
        else static if (TARGET_LINUX || TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS)
        {
            lib_ext = "a";
        }
        else
        {
            static assert(0, "fix this");
        }
        static if (TARGET_WINDOS)
        {
            dll_ext = "dll";
        }
        else static if (TARGET_LINUX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS)
        {
            dll_ext = "so";
        }
        else static if (TARGET_OSX)
        {
            dll_ext = "dylib";
        }
        else
        {
            static assert(0, "fix this");
        }
        static if (TARGET_WINDOS)
        {
            run_noext = false;
        }
        else static if (TARGET_LINUX || TARGET_OSX || TARGET_FREEBSD || TARGET_OPENBSD || TARGET_SOLARIS)
        {
            // Allow 'script' D source files to have no extension.
            run_noext = true;
        }
        else
        {
            static assert(0, "fix this");
        }
}
        copyright = "Copyright (c) 1999-2016 by Digital Mars";
        written = "written by Walter Bright";
version(IN_LLVM)
{
        compiler.vendor = "LDC";
}
else
{
        _version = ('v' ~ stripRight(import("verstr.h"))[1 .. $ - 1] ~ '\0').ptr;
        compiler.vendor = "Digital Mars D";
}
        stdmsg = stdout;
        main_d = "__main.d";
        errorLimit = 20;
    }
}

// Because int64_t and friends may be any integral type of the
// correct size, we have to explicitly ask for the correct
// integer type to get the correct mangling with ddmd

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
    const(char)* filename;
    uint linnum;
    uint charnum;

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
            buf.printf("%s", filename);
        }
        if (linnum)
        {
            buf.printf("(%d", linnum);
            if (global.params.showColumns && charnum)
                buf.printf(",%d", charnum);
            buf.writeByte(')');
        }
        return buf.extractString();
    }

    extern (C++) bool equals(ref const(Loc) loc)
    {
        return (!global.params.showColumns || charnum == loc.charnum) && linnum == loc.linnum && FileName.equals(filename, loc.filename);
    }
}

enum LINK : int
{
    def,        // default
    d,
    c,
    cpp,
    windows,
    pascal,
    objc,
}

alias LINKdefault = LINK.def;
alias LINKd = LINK.d;
alias LINKc = LINK.c;
alias LINKcpp = LINK.cpp;
alias LINKwindows = LINK.windows;
alias LINKpascal = LINK.pascal;
alias LINKobjc = LINK.objc;

enum CPPMANGLE : int
{
    def,
    asStruct,
    asClass,
}

enum DYNCAST : int
{
    object,
    expression,
    dsymbol,
    type,
    identifier,
    tuple,
    parameter,
}

alias DYNCAST_OBJECT = DYNCAST.object;
alias DYNCAST_EXPRESSION = DYNCAST.expression;
alias DYNCAST_DSYMBOL = DYNCAST.dsymbol;
alias DYNCAST_TYPE = DYNCAST.type;
alias DYNCAST_IDENTIFIER = DYNCAST.identifier;
alias DYNCAST_TUPLE = DYNCAST.tuple;
alias DYNCAST_PARAMETER = DYNCAST.parameter;

enum MATCH : int
{
    nomatch,   // no match
    convert,   // match with conversions
    constant,  // match with conversion to const
    exact,     // exact match
}

alias MATCHnomatch = MATCH.nomatch;
alias MATCHconvert = MATCH.convert;
alias MATCHconst = MATCH.constant;
alias MATCHexact = MATCH.exact;

enum PINLINE : int
{
    def,     // as specified on the command line
    never,   // never inline
    always,  // always inline
}

alias PINLINEdefault = PINLINE.def;
alias PINLINEnever = PINLINE.never;
alias PINLINEalways = PINLINE.always;

alias StorageClass = uinteger_t;

extern (C++) __gshared Global global;
