
// Compiler implementation of the D programming language
// Copyright (c) 1999-2011 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef DMD_MODULE_H
#define DMD_MODULE_H

#ifdef __DMC__
#pragma once
#endif /* __DMC__ */

#include "root.h"
#include "dsymbol.h"

struct ModuleInfoDeclaration;
struct ClassDeclaration;
struct ModuleDeclaration;
struct Macro;
struct Escape;
struct VarDeclaration;
struct Library;

// Back end
#if IN_LLVM
class Ir;
struct DValue;
typedef DValue elem;
namespace llvm {
    class LLVMContext;
    class Module;
    class GlobalVariable;
    class StructType;
}
#else

#ifdef IN_GCC
union tree_node; typedef union tree_node elem;
#else
struct elem;
#endif
#endif

struct Package : ScopeDsymbol
{
    Package(Identifier *ident);
    const char *kind();

    static DsymbolTable *resolve(Identifiers *packages, Dsymbol **pparent, Package **ppkg);

    Package *isPackage() { return this; }

    virtual void semantic(Scope *sc) { }
};

struct Module : Package
{
    static Module *rootModule;
    static DsymbolTable *modules;       // symbol table of all modules
    static Modules amodules;            // array of all modules
    static Dsymbols deferred;   // deferred Dsymbol's needing semantic() run on them
    static unsigned dprogress;  // progress resolving the deferred list
    static void init();

    static ClassDeclaration *moduleinfo;


    const char *arg;    // original argument name
    ModuleDeclaration *md; // if !NULL, the contents of the ModuleDeclaration declaration
    File *srcfile;      // input source file

    File *objfile; // output object file
    File *docfile; // output doc file
    File *hdrfile; // output hdr file

    unsigned errors;    // if any errors in file
    unsigned numlines;  // number of lines in source file
    int isHtml;         // if it is an HTML file
    int isDocFile;      // if it is a documentation input file, not D source
    int needmoduleinfo;
#ifdef IN_GCC
    int strictlyneedmoduleinfo;
#endif

    int selfimports;            // 0: don't know, 1: does not, 2: does
    int selfImports();          // returns !=0 if module imports itself

    int insearch;
    Identifier *searchCacheIdent;
    Dsymbol *searchCacheSymbol; // cached value of search
    int searchCacheFlags;       // cached flags

    int semanticstarted;        // has semantic() been started?
    int semanticRun;            // has semantic() been done?
    int root;                   // != 0 if this is a 'root' module,
                                // i.e. a module that will be taken all the
                                // way to an object file
    Module *importedFrom;       // module from command line we're imported from,
                                // i.e. a module that will be taken all the
                                // way to an object file

    Dsymbols *decldefs;         // top level declarations for this Module

    Modules aimports;             // all imported modules

    ModuleInfoDeclaration *vmoduleinfo;

    unsigned debuglevel;        // debug level
    Strings *debugids;      // debug identifiers
    Strings *debugidsNot;       // forward referenced debug identifiers

    unsigned versionlevel;      // version level
    Strings *versionids;    // version identifiers
    Strings *versionidsNot;     // forward referenced version identifiers

    Macro *macrotable;          // document comment macros
    struct Escape *escapetable; // document comment escapes

    int doDocComment;          // enable generating doc comments for this module
    int doHdrGen;              // enable generating header file for this module

    bool safe;                  // TRUE if module is marked as 'safe'

    Module(char *arg, Identifier *ident, int doDocComment, int doHdrGen);
    ~Module();

    static Module *load(Loc loc, Identifiers *packages, Identifier *ident);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toJsonBuffer(OutBuffer *buf);
    const char *kind();
    void read(Loc loc); // read file
#if IN_LLVM
    void parse(bool gen_docs = false);       // syntactic parse
#elif IN_GCC
    void parse(bool dump_source = false);       // syntactic parse
#else
    void parse();       // syntactic parse
#endif
    void importAll(Scope *sc);
    void semantic();     // semantic analysis
    void semantic2();    // pass 2 semantic analysis
    void semantic3();    // pass 3 semantic analysis
    void inlineScan();  // scan for functions to inline
    void setHdrfile();  // set hdrfile member
    void genhdrfile();  // generate D import file
    void genobjfile(int multiobj);
//    void gensymfile();
    void gendocfile();
    int needModuleInfo();
    Dsymbol *search(Loc loc, Identifier *ident, int flags);
    Dsymbol *symtabInsert(Dsymbol *s);
    void deleteObjFile();
    void addDeferredSemantic(Dsymbol *s);
    static void runDeferredSemantic();
    static void clearCache();
    int imports(Module *m);

    // Back end
#if IN_DMD
    int doppelganger;           // sub-module
    Symbol *cov;                // private uint[] __coverage;
    unsigned *covb;             // bit array of valid code line numbers

    Symbol *sictor;             // module order independent constructor
    Symbol *sctor;              // module constructor
    Symbol *sdtor;              // module destructor
    Symbol *stest;              // module unit test

    Symbol *sfilename;          // symbol for filename

    Symbol *massert;            // module assert function
    Symbol *toModuleAssert();   // get module assert function

    Symbol *munittest;          // module unittest failure function
    Symbol *toModuleUnittest(); // get module unittest failure function

    Symbol *marray;             // module array bounds function
    Symbol *toModuleArray();    // get module array bounds function


    static Symbol *gencritsec();
    elem *toEfilename();
    elem *toEmodulename();

    Symbol *toSymbol();
#endif
    void genmoduleinfo();

#if IN_LLVM
    // LDC
    llvm::Module* genLLVMModule(llvm::LLVMContext& context, Ir* sir);
    void buildTargetFiles(bool singleObj);
    File* buildFilePath(const char* forcename, const char* path, const char* ext);
    Module *isModule() { return this; }
    llvm::GlobalVariable* moduleInfoSymbol();

    bool llvmForceLogging;
    llvm::GlobalVariable* moduleInfoVar;
    llvm::StructType* moduleInfoType;

    // array ops emitted in this module already
    StringTable arrayfuncs;

    bool isRoot;
#endif
};


struct ModuleDeclaration
{
    Identifier *id;
    Identifiers *packages;            // array of Identifier's representing packages
    bool safe;

    ModuleDeclaration(Identifiers *packages, Identifier *id);

    char *toChars();
};

#endif /* DMD_MODULE_H */
