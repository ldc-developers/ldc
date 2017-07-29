
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2016 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/dlang/dmd/blob/master/src/mars.h
 */

#ifndef DMD_GLOBALS_H
#define DMD_GLOBALS_H

#ifdef __DMC__
#pragma once
#endif

#include "ctfloat.h"
#include "outbuffer.h"
#include "filename.h"

// Can't include arraytypes.h here, need to declare these directly.
template <typename TYPE> struct Array;

#if IN_LLVM
#include "llvm/ADT/Triple.h"
#include <cstdint>

enum OUTPUTFLAG
{
    OUTPUTFLAGno,
    OUTPUTFLAGdefault, // for the .o default
    OUTPUTFLAGset      // for -output
};

using ubyte = uint8_t;
#endif


// The state of array bounds checking
enum BOUNDSCHECK
{
    BOUNDSCHECKdefault, // initial value
    BOUNDSCHECKoff,     // never do bounds checking
    BOUNDSCHECKon,      // always do bounds checking
    BOUNDSCHECKsafeonly // do bounds checking only in @safe functions
};

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
};

// Put command line switches in here
struct Param
{
    bool obj;           // write object file
    bool link;          // perform link
    bool dll;           // generate shared dynamic library
    bool lib;           // write library file instead of object file(s)
    bool multiobj;      // break one object file into multiple ones
    bool oneobj;        // write one object file instead of multiple ones
    bool trace;         // insert profiling hooks
    bool tracegc;       // instrument calls to 'new'
    bool verbose;       // verbose compile
    bool vcg_ast;       // write-out codegen-ast
    bool showColumns;   // print character (column) numbers in diagnostics
    bool vtls;          // identify thread local variables
#if !IN_LLVM
    char vgc;           // identify gc usage
#else
    bool vgc;           // identify gc usage
#endif
    bool vfield;        // identify non-mutable field variables
    bool vcomplex;      // identify complex/imaginary type usage
#if !IN_LLVM
    char symdebug;      // insert debug symbolic information
#else
    ubyte symdebug;     // insert debug symbolic information
#endif
    bool alwaysframe;   // always emit standard stack frame
    bool optimize;      // run optimizer
    bool map;           // generate linker .map file
    bool is64bit;       // generate 64 bit code
    bool isLP64;        // generate code for LP64
    bool isLinux;       // generate code for linux
    bool isOSX;         // generate code for Mac OSX
    bool isWindows;     // generate code for Windows
    bool isFreeBSD;     // generate code for FreeBSD
    bool isOpenBSD;     // generate code for OpenBSD
    bool isSolaris;     // generate code for Solaris
    bool hasObjectiveC; // target supports Objective-C
    bool mscoff;        // for Win32: write COFF object files instead of OMF
    // 0: don't allow use of deprecated features
    // 1: silently allow use of deprecated features
    // 2: warn about the use of deprecated features
#if !IN_LLVM
    char useDeprecated;
#else
    ubyte useDeprecated;
#endif
    bool useAssert;     // generate runtime code for assert()'s
    bool useInvariants; // generate class invariant checks
    bool useIn;         // generate precondition checks
    bool useOut;        // generate postcondition checks
    bool stackstomp;    // add stack stomping code
    bool useSwitchError; // check for switches without a default
    bool useUnitTests;  // generate unittest code
    bool useInline;     // inline expand functions
    bool useDIP25;      // implement http://wiki.dlang.org/DIP25
    bool release;       // build release version
    bool preservePaths; // true means don't strip path from source file
    // 0: disable warnings
    // 1: warnings as errors
    // 2: informational warnings (no errors)
#if !IN_LLVM
    char warnings;
#else
    ubyte warnings;
#endif
    bool pic;           // generate position-independent-code for shared libs
    bool color;         // use ANSI colors in console output
    bool cov;           // generate code coverage data
    unsigned char covPercent;   // 0..100 code coverage percentage required
    bool nofloat;       // code should not pull in floating point support
    bool ignoreUnsupportedPragmas;      // rather than error on them
    bool enforcePropertySyntax;
    bool betterC;       // be a "better C" compiler; no dependency on D runtime
    bool addMain;       // add a default main() function
    bool allInst;       // generate code for all template instantiations
    bool check10378;    // check for issues transitioning to 10738
    bool bug10378;      // use pre-bugzilla 10378 search strategy
    bool vsafe;         // use enhanced @safe checking
    bool showGaggedErrors;  // print gagged errors anyway

    CPU cpu;                // CPU instruction set to target
    BOUNDSCHECK useArrayBounds;

    const char *argv0;    // program name
    Array<const char *> *modFileAliasStrings; // array of char*'s of -I module filename alias strings
    Array<const char *> *imppath;     // array of char*'s of where to look for import modules
    Array<const char *> *fileImppath; // array of char*'s of where to look for file import modules
    const char *objdir;   // .obj/.lib file output directory
    const char *objname;  // .obj file output name
    const char *libname;  // .lib file output name

    bool doDocComments;  // process embedded documentation comments
    const char *docdir;  // write documentation file to docdir directory
    const char *docname; // write documentation file to docname
    Array<const char *> *ddocfiles;  // macro include files for Ddoc

    bool doHdrGeneration;  // process embedded documentation comments
    const char *hdrdir;    // write 'header' file to docdir directory
    const char *hdrname;   // write 'header' file to docname
    bool hdrStripPlainFunctions; // strip the bodies of plain (non-template) functions

    bool doJsonGeneration;    // write JSON file
    const char *jsonfilename; // write JSON file to jsonfilename

    unsigned debuglevel;   // debug level
    Array<const char *> *debugids;     // debug identifiers

    unsigned versionlevel; // version level
    Array<const char *> *versionids;   // version identifiers

    const char *defaultlibname; // default library for non-debug builds
    const char *debuglibname;   // default library for debug builds
    const char *mscrtlib;       // MS C runtime library

    const char *moduleDepsFile; // filename for deps output
    OutBuffer *moduleDeps;      // contents to be written to deps file

    // Hidden debug switches
    bool debugb;
    bool debugc;
    bool debugf;
    bool debugr;
    bool debugx;
    bool debugy;

    bool run;           // run resulting executable
    Strings runargs;    // arguments for executable

    // Linker stuff
    Array<const char *> *objfiles;
    Array<const char *> *linkswitches;
    Array<const char *> *libfiles;
    Array<const char *> *dllfiles;
    const char *deffile;
    const char *resfile;
    const char *exefile;
    const char *mapfile;

#if IN_LLVM
    Array<const char *> *bitcodeFiles; // LLVM bitcode files passed on cmdline

    uint32_t nestedTmpl; // maximum nested template instantiations

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
    bool genInstrProf;             // Whether to generate PGO instrumented code
    const char *datafileInstrProf; // Either the input or output file for PGO data

    const llvm::Triple *targetTriple;

    // Codegen cl options
    bool disableRedZone;
    uint32_t dwarfVersion;

    uint32_t hashThreshold; // MD5 hash symbols larger than this threshold (0 = no hashing)

    bool outputSourceLocations; // if true, output line tables.
#endif
};

struct Compiler
{
    const char *vendor;     // Compiler backend name
};

typedef unsigned structalign_t;
// magic value means "match whatever the underlying C compiler does"
// other values are all powers of 2
#define STRUCTALIGN_DEFAULT ((structalign_t) ~0)

struct Global
{
    const char *inifilename;
    const char *mars_ext;
    const char *obj_ext;
#if IN_LLVM
    const char *ll_ext;
    const char *bc_ext;
    const char *s_ext;
    const char *ldc_version;
    const char *llvm_version;

    bool gaggedForInlining; // Set for functionSemantic3 for external inlining candidates
#endif
    const char *lib_ext;
    const char *dll_ext;
    const char *doc_ext;        // for Ddoc generated files
    const char *ddoc_ext;       // for Ddoc macro include files
    const char *hdr_ext;        // for D 'header' import files
    const char *json_ext;       // for JSON files
    const char *map_ext;        // for .map files
    bool run_noext;             // allow -run sources without extensions.

    const char *copyright;
    const char *written;
    const char *main_d;         // dummy filename for dummy main()
    Array<const char *> *path;        // Array of char*'s which form the import lookup path
    Array<const char *> *filePath;    // Array of char*'s which form the file import lookup path

    const char *version;

    Compiler compiler;
    Param params;
    unsigned errors;       // number of errors reported so far
    unsigned warnings;     // number of warnings reported so far
    FILE *stdmsg;          // where to send verbose messages
    unsigned gag;          // !=0 means gag reporting of errors & warnings
    unsigned gaggedErrors; // number of errors reported while gagged

    unsigned errorLimit;

    /* Start gagging. Return the current number of gagged errors
     */
    unsigned startGagging();

    /* End gagging, restoring the old gagged state.
     * Return true if errors occured while gagged.
     */
    bool endGagging(unsigned oldGagged);

    /*  Increment the error count to record that an error
     *  has occured in the current context. An error message
     *  may or may not have been printed.
     */
    void increaseErrorCount();

    void _init();
};

extern Global global;

// Because int64_t and friends may be any integral type of the
// correct size, we have to explicitly ask for the correct
// integer type to get the correct mangling with ddmd
#if __LP64__
// Be careful not to care about sign when using dinteger_t
// use this instead of integer_t to
// avoid conflicts with system #include's
typedef unsigned long dinteger_t;
// Signed and unsigned variants
typedef long sinteger_t;
typedef unsigned long uinteger_t;
#else
typedef unsigned long long dinteger_t;
typedef long long sinteger_t;
typedef unsigned long long uinteger_t;
#endif

typedef int8_t                  d_int8;
typedef uint8_t                 d_uns8;
typedef int16_t                 d_int16;
typedef uint16_t                d_uns16;
typedef int32_t                 d_int32;
typedef uint32_t                d_uns32;
typedef int64_t                 d_int64;
typedef uint64_t                d_uns64;

// Represents a D [ ] array
template<typename T>
struct DArray
{
    size_t length;
    T *ptr;
};

// file location
struct Loc
{
    const char *filename;
    unsigned linnum;
    unsigned charnum;

    Loc()
    {
        linnum = 0;
        charnum = 0;
        filename = NULL;
    }

#if IN_LLVM
    Loc(const char *filename, unsigned linnum, unsigned charnum)
        : filename(filename), linnum(linnum), charnum(charnum) {}
#else
    Loc(const char *filename, unsigned linnum, unsigned charnum);
#endif

    const char *toChars() const;
    bool equals(const Loc& loc);
};

enum LINK
{
    LINKdefault,
    LINKd,
    LINKc,
    LINKcpp,
    LINKwindows,
    LINKpascal,
    LINKobjc,
};

enum CPPMANGLE
{
    CPPMANGLEdefault,
    CPPMANGLEstruct,
    CPPMANGLEclass,
};

enum MATCH
{
    MATCHnomatch,       // no match
    MATCHconvert,       // match with conversions
    MATCHconst,         // match with conversion to const
    MATCHexact          // exact match
};

enum PINLINE
{
    PINLINEdefault,      // as specified on the command line
    PINLINEnever,        // never inline
    PINLINEalways        // always inline
};

typedef uinteger_t StorageClass;

#endif /* DMD_GLOBALS_H */
