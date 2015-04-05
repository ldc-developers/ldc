
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/mars.h
 */

#ifndef DMD_GLOBALS_H
#define DMD_GLOBALS_H

#ifdef __DMC__
#pragma once
#endif

#include "longdouble.h"
#include "int128.h"
#include "outbuffer.h"

// Can't include arraytypes.h here, need to declare these directly.
template <typename TYPE> struct Array;

#if IN_LLVM
#include "llvm/ADT/Triple.h"

enum OUTPUTFLAG
{
    OUTPUTFLAGno,
    OUTPUTFLAGdefault, // for the .o default
    OUTPUTFLAGset // for -output
};

typedef unsigned char ubyte;
#endif

// Put command line switches in here
struct Param
{
    bool obj;           // write object file
    bool link;          // perform link
#if !IN_LLVM
    bool dll;           // generate shared dynamic library
    bool lib;           // write library file instead of object file(s)
    bool multiobj;      // break one object file into multiple ones
    bool oneobj;        // write one object file instead of multiple ones
#endif
    bool trace;         // insert profiling hooks
    bool verbose;       // verbose compile
    bool showColumns;   // print character (column) numbers in diagnostics
    bool vtls;          // identify thread local variables
#if !IN_LLVM
    char vgc;           // identify gc usage
#else
    bool vgc;           // identify gc usage
#endif
    bool vfield;        // identify non-mutable field variables
#if !IN_LLVM
    char symdebug;      // insert debug symbolic information
#else
    ubyte symdebug;     // insert debug symbolic information
#endif
    bool alwaysframe;   // always emit standard stack frame
    bool optimize;      // run optimizer
#if !IN_LLVM
    bool map;           // generate linker .map file
#endif
    bool is64bit;       // generate 64 bit code
    bool isLP64;        // generate code for LP64
    bool isLinux;       // generate code for linux
    bool isOSX;         // generate code for Mac OSX
    bool isWindows;     // generate code for Windows
    bool isFreeBSD;     // generate code for FreeBSD
    bool isOpenBSD;     // generate code for OpenBSD
    bool isSolaris;     // generate code for Solaris
#if !IN_LLVM
    bool mscoff;        // for Win32: write COFF object files instead of OMF
    char useDeprecated; // 0: don't allow use of deprecated features
                        // 1: silently allow use of deprecated features
                        // 2: warn about the use of deprecated features
#else
    ubyte useDeprecated; // 0: don't allow use of deprecated features
                         // 1: silently allow use of deprecated features
                         // 2: warn about the use of deprecated features
#endif
    bool useAssert;     // generate runtime code for assert()'s
    bool useInvariants; // generate class invariant checks
    bool useIn;         // generate precondition checks
    bool useOut;        // generate postcondition checks
    char useArrayBounds; // 0: no array bounds checks
                         // 1: array bounds checks for safe functions only
                         // 2: array bounds checks for all functions
    bool stackstomp;    // add stack stomping code
    bool useSwitchError; // check for switches without a default
    bool useUnitTests;  // generate unittest code
    bool useInline;     // inline expand functions
    bool useDIP25;      // implement http://wiki.dlang.org/DIP25
#if !IN_LLVM
    bool release;       // build release version
    bool preservePaths; // true means don't strip path from source file
    char warnings;      // 0: disable warnings
                        // 1: warnings as errors
                        // 2: informational warnings (no errors)
#else
    ubyte warnings;      // 0: disable warnings
                         // 1: warnings as errors
                         // 2: informational warnings (no errors)
#endif

#if IN_LLVM
    bool color;         // use ANSI colors in console output
    bool cov;           // generate code coverage data
    unsigned char covPercent;   // 0..100 code coverage percentage required
    bool ignoreUnsupportedPragmas;      // rather than error on them
    bool enforcePropertySyntax;
    bool addMain; // LDC_FIXME: Implement.
    bool allInst; // LDC_FIXME: Implement.
    unsigned nestedTmpl; // maximum nested template instantiations
#else
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
#endif

    const char *argv0;    // program name
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

    bool doJsonGeneration;    // write JSON file
    const char *jsonfilename; // write JSON file to jsonfilename

    unsigned debuglevel;   // debug level
    Array<const char *> *debugids;     // debug identifiers

    unsigned versionlevel; // version level
    Array<const char *> *versionids;   // version identifiers

#if !IN_LLVM
    const char *defaultlibname; // default library for non-debug builds
    const char *debuglibname;   // default library for debug builds
#endif

    const char *moduleDepsFile; // filename for deps output
    OutBuffer *moduleDeps;      // contents to be written to deps file

#if !IN_LLVM
    // Hidden debug switches
    bool debugb;
    bool debugc;
    bool debugf;
    bool debugr;
    bool debugx;
    bool debugy;
#endif

    bool run;           // run resulting executable
#if !IN_LLVM
    size_t runargs_length;
    const char** runargs; // arguments for executable
#endif

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

    // target stuff
    llvm::Triple targetTriple;

    // Codegen cl options
    bool singleObj;
    bool disableRedZone;
#endif
};

struct Compiler
{
    const char *vendor;     // Compiler backend name
};

typedef unsigned structalign_t;
#define STRUCTALIGN_DEFAULT ((structalign_t) ~0)  // magic value means "match whatever the underlying C compiler does"
// other values are all powers of 2

struct Global
{
    const char *inifilename;
    const char *mars_ext;
    const char *obj_ext;
#if IN_LLVM
    const char *obj_ext_alt;
    const char *ll_ext;
    const char *bc_ext;
    const char *s_ext;
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
#if IN_LLVM
    const char *ldc_version;
    const char *llvm_version;
#endif

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

    void init();
};

extern Global global;

#if WANT_CENT
typedef uint128_t dinteger_t;
typedef int128_t sinteger_t;
typedef uint128_t uinteger_t;
#else
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
#endif

typedef int8_t                  d_int8;
typedef uint8_t                 d_uns8;
typedef int16_t                 d_int16;
typedef uint16_t                d_uns16;
typedef int32_t                 d_int32;
typedef uint32_t                d_uns32;
typedef int64_t                 d_int64;
typedef uint64_t                d_uns64;
#if WANT_CENT
typedef int128_t                d_int128;
typedef uint128_t               d_uns128;
#endif

typedef float                   d_float32;
typedef double                  d_float64;
typedef longdouble              d_float80;

typedef d_uns8                  d_char;
typedef d_uns16                 d_wchar;
typedef d_uns32                 d_dchar;

typedef longdouble real_t;

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

    Loc(const char *filename, unsigned linnum, unsigned charnum);

    char *toChars();
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
};

enum DYNCAST
{
    DYNCAST_OBJECT,
    DYNCAST_EXPRESSION,
    DYNCAST_DSYMBOL,
    DYNCAST_TYPE,
    DYNCAST_IDENTIFIER,
    DYNCAST_TUPLE,
    DYNCAST_PARAMETER,
};

enum MATCH
{
    MATCHnomatch,       // no match
    MATCHconvert,       // match with conversions
    MATCHconst,         // match with conversion to const
    MATCHexact          // exact match
};

typedef uinteger_t StorageClass;

#endif /* DMD_GLOBALS_H */
