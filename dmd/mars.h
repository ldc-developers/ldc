
// Compiler implementation of the D programming language
// Copyright (c) 1999-2009 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef DMD_MARS_H
#define DMD_MARS_H

#ifdef __DMC__
#pragma once
#endif

/*
It is very important to use version control macros correctly - the
idea is that host and target are independent. If these are done
correctly, cross compilers can be built.
The host compiler and host operating system are also different,
and are predefined by the host compiler. The ones used in
dmd are:

Macros defined by the compiler, not the code:

    Compiler:
	__DMC__		Digital Mars compiler
	_MSC_VER	Microsoft compiler
	__GNUC__	Gnu compiler

    Host operating system:
	_WIN32		Microsoft NT, Windows 95, Windows 98, Win32s,
			Windows 2000, Win XP, Vista
	_WIN64		Windows for AMD64
	linux		Linux
	__APPLE__	Mac OSX
	__FreeBSD__	FreeBSD
	__sun&&__SVR4	Solaris, OpenSolaris (yes, both macros are necessary)

For the target systems, there are the target operating system and
the target object file format:

    Target operating system:
	TARGET_WINDOS	Covers 32 bit windows and 64 bit windows
	TARGET_LINUX	Covers 32 and 64 bit linux
	TARGET_OSX	Covers 32 and 64 bit Mac OSX
	TARGET_FREEBSD	Covers 32 and 64 bit FreeBSD
	TARGET_SOLARIS	Covers 32 and 64 bit Solaris
	TARGET_NET	Covers .Net

    It is expected that the compiler for each platform will be able
    to generate 32 and 64 bit code from the same compiler binary.

    Target object module format:
	OMFOBJ		Intel Object Module Format, used on Windows
	ELFOBJ		Elf Object Module Format, used on linux, FreeBSD and Solaris
	MACHOBJ		Mach-O Object Module Format, used on Mac OSX

    There are currently no macros for byte endianness order.
 */


#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>
#include <stdarg.h>

#ifdef __DMC__
#ifdef DEBUG
#undef assert
#define assert(e) (static_cast<void>((e) || (printf("assert %s(%d) %s\n", __FILE__, __LINE__, #e), halt())))
#endif
#endif

#ifndef IS_PRINTF
# ifdef __GNUC__
#  define IS_PRINTF(FMTARG) __attribute((__format__ (__printf__, (FMTARG), (FMTARG)+1) ))
# else
#  define IS_PRINTF(FMTARG)
# endif
#endif

#ifdef IN_GCC
/* Changes for the GDC compiler by David Friedman */
#endif

#define DMDV2	0	// Version 2.0 features
#define BREAKABI 1	// 0 if not ready to break the ABI just yet
#define STRUCTTHISREF DMDV2	// if 'this' for struct is a reference, not a pointer
#define SNAN_DEFAULT_INIT DMDV2	// if floats are default initialized to signalling NaN

/* Other targets are TARGET_LINUX, TARGET_OSX, TARGET_FREEBSD and
 * TARGET_SOLARIS, which are
 * set on the command line via the compiler makefile.
 */

#if _WIN32
#define TARGET_WINDOS 1		// Windows dmd generates Windows targets
#define OMFOBJ 1
#endif

#if TARGET_LINUX || TARGET_FREEBSD || TARGET_SOLARIS
#ifndef ELFOBJ
#define ELFOBJ 1
#endif
#endif

#if TARGET_OSX
#ifndef MACHOBJ
#define MACHOBJ 1
#endif
#endif


struct Array;
struct OutBuffer;

// LDC
enum ARCH
{
    ARCHinvalid,
    ARCHx86,
    ARCHx86_64,
    ARCHppc,
    ARCHppc_64,
    ARCHarm,
    ARCHthumb
};
enum OUTPUTFLAG
{
    OUTPUTFLAGno,
    OUTPUTFLAGdefault, // for the .o default
    OUTPUTFLAGset // for -output
};

enum OS
{
    OSinvalid,
    OSLinux,
    OSWindows,
    OSMacOSX,
    OSFreeBSD,
    OSSolaris,
};

typedef unsigned char ubyte;

// Put command line switches in here
struct Param
{
    bool obj;           // write object file
    bool link;          // perform link
    bool verbose;       // verbose compile
    ubyte symdebug;	// insert debug symbolic information
#if !IN_LLVM
    // LDC uses a different mechanism
    bool optimize;      // run optimizer
    char optimizeLevel; // optimization level
#endif
    char vtls;		// identify thread local variables
    ARCH cpu;		// target CPU
    OS   os;		// target OS
    bool is64bit;       // generate 64 bit code
    bool isLE;          // generate little endian code
    bool useDeprecated; // allow use of deprecated features
    bool useAssert;     // generate runtime code for assert()'s
    bool useInvariants; // generate class invariant checks
    bool useIn;         // generate precondition checks
    bool useOut;        // generate postcondition checks
    bool useArrayBounds; // generate array bounds checks
    bool useSwitchError; // check for switches without a default
    bool useUnitTests;  // generate unittest code
    bool useInline;     // inline expand functions
    bool warnings;      // enable warnings
    ubyte Dversion;	// D version number
    char safe;		// enforce safe memory model

    char *argv0;	// program name
    Array *imppath;	// array of char*'s of where to look for import modules
    Array *fileImppath;	// array of char*'s of where to look for file import modules
    char *objdir;	// .obj file output directory
    char *objname;	// .obj file output name

    bool doDocComments; // process embedded documentation comments
    char *docdir;	// write documentation file to docdir directory
    char *docname;	// write documentation file to docname
    Array *ddocfiles;	// macro include files for Ddoc

    bool doHdrGeneration;       // process embedded documentation comments
    char *hdrdir;		// write 'header' file to docdir directory
    char *hdrname;		// write 'header' file to docname

    char doXGeneration;		// write JSON file
    char *xfilename;		// write JSON file to xfilename

    unsigned debuglevel;	// debug level
    Array *debugids;		// debug identifiers

    unsigned versionlevel;	// version level
    Array *versionids;		// version identifiers

    bool dump_source;

    Array *defaultlibnames;	// default libraries for non-debug builds
    Array *debuglibnames;	// default libraries for debug builds

    char *moduleDepsFile;	// filename for deps output
    OutBuffer *moduleDeps;	// contents to be written to deps file

    // Hidden debug switches
    bool debuga;
    bool debugb;
    bool debugc;
    bool debugf;
    bool debugr;
    bool debugw;
    bool debugx;
    bool debugy;

    bool run;           // run resulting executable

    // Linker stuff
    Array *objfiles;
    Array *linkswitches;
    Array *libfiles;
    char *deffile;
    char *resfile;
    char *exefile;

    // LDC stuff
    OUTPUTFLAG output_ll;
    OUTPUTFLAG output_bc;
    OUTPUTFLAG output_s;
    OUTPUTFLAG output_o;
    bool llvmAnnotate;
    bool useInlineAsm;
    bool verbose_cg;
    bool useAvailableExternally;

    // target stuff
    const char* llvmArch;
    const char *targetTriple;
    const char *dataLayout;
};

struct Global
{
    const char *mars_ext;
    const char *sym_ext;
    const char *obj_ext;
#if _WIN32
    char *obj_ext_alt;
#endif
    char *ll_ext;
    char *bc_ext;
    char *s_ext;
    const char *lib_ext;
    const char *doc_ext;	// for Ddoc generated files
    const char *ddoc_ext;	// for Ddoc macro include files
    const char *hdr_ext;	// for D 'header' import files
    const char *json_ext;	// for JSON files
    const char *copyright;
    const char *written;
    Array *path;	// Array of char*'s which form the import lookup path
    Array *filePath;	// Array of char*'s which form the file import lookup path
    int structalign;
    const char *version;
    char *ldc_version;
    char *llvm_version;

    Param params;
    unsigned errors;	// number of errors reported so far
    unsigned gag;	// !=0 means gag reporting of errors

    Global();
};

extern Global global;

/* Set if Windows Structured Exception Handling C extensions are supported.
 * Apparently, VC has dropped support for these?
 */
#define WINDOWS_SEH	(_WIN32 && __DMC__)


#ifdef __DMC__
 typedef _Complex long double complex_t;
#else
 #ifndef IN_GCC
  #include "complex_t.h"
 #endif
 #ifdef __APPLE__
  //#include "complex.h"//This causes problems with include the c++ <complex> and not the C "complex.h"
 #endif
#endif

// Be careful not to care about sign when using dinteger_t
//typedef uint64_t integer_t;
typedef uint64_t dinteger_t;	// use this instead of integer_t to
				// avoid conflicts with system #include's

// Signed and unsigned variants
typedef int64_t sinteger_t;
typedef uint64_t uinteger_t;

typedef int8_t			d_int8;
typedef uint8_t			d_uns8;
typedef int16_t			d_int16;
typedef uint16_t		d_uns16;
typedef int32_t			d_int32;
typedef uint32_t		d_uns32;
typedef int64_t			d_int64;
typedef uint64_t		d_uns64;

typedef float			d_float32;
typedef double			d_float64;
typedef long double		d_float80;

typedef d_uns8			d_char;
typedef d_uns16			d_wchar;
typedef d_uns32			d_dchar;

#ifdef IN_GCC
#include "d-gcc-real.h"
#else
typedef long double real_t;
#endif

// Modify OutBuffer::writewchar to write the correct size of wchar
#if _WIN32
#define writewchar writeword
#else
// This needs a configuration test...
#define writewchar write4
#endif

#ifdef IN_GCC
#include "d-gcc-complex_t.h"
#endif

struct Module;

//typedef unsigned Loc;		// file location
struct Loc
{
    const char *filename;
    unsigned linnum;

    Loc()
    {
	linnum = 0;
	filename = NULL;
    }

    Loc(int x)
    {
	linnum = x;
	filename = NULL;
    }

    Loc(Module *mod, unsigned linnum);

    char *toChars() const;
    bool equals(const Loc& loc);
};

#ifndef GCC_SAFE_DMD
#define TRUE	1
#define FALSE	0
#endif

#define INTERFACE_OFFSET	0	// if 1, put classinfo as first entry
					// in interface vtbl[]'s
#define INTERFACE_VIRTUAL	0	// 1 means if an interface appears
					// in the inheritance graph multiple
					// times, only one is used

enum LINK
{
    LINKdefault,
    LINKd,
    LINKc,
    LINKcpp,
    LINKwindows,
    LINKpascal,

    // LDC
    LINKintrinsic,
};

enum DYNCAST
{
    DYNCAST_OBJECT,
    DYNCAST_EXPRESSION,
    DYNCAST_DSYMBOL,
    DYNCAST_TYPE,
    DYNCAST_IDENTIFIER,
    DYNCAST_TUPLE,
};

enum MATCH
{
    MATCHnomatch,	// no match
    MATCHconvert,	// match with conversions
#if DMDV2
    MATCHconst,		// match with conversion to const
#endif
    MATCHexact		// exact match
};

void warning(Loc loc, const char *format, ...) IS_PRINTF(2);
void vwarning(Loc loc, const char *format, va_list);
void error(Loc loc, const char *format, ...) IS_PRINTF(2);
void verror(Loc loc, const char *format, va_list);

#ifdef __GNUC__
__attribute__((noreturn))
#endif
void fatal();

void err_nomem();
void inifile(char *argv0, const char *inifile);
void halt();

/*** Where to send error messages ***/
#if IN_GCC || IN_LLVM
#define stdmsg stderr
#else
#define stdmsg stderr
#endif

#endif /* DMD_MARS_H */
