
// Compiler implementation of the D programming language
// Copyright (c) 1999-2012 by Digital Mars
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
class Library;

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

    static AggregateDeclaration *moduleinfo;

    const char *arg;    // original argument name
    ModuleDeclaration *md; // if !NULL, the contents of the ModuleDeclaration declaration
    File *srcfile;      // input source file
    File *objfile;      // output .obj file
    File *hdrfile;      // 'header' file
    File *symfile;      // output symbol file
    File *docfile;      // output documentation file

    unsigned errors;    // if any errors in file
    unsigned numlines;  // number of lines in source file
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
    Escape *escapetable;        // document comment escapes
    bool safe;                  // TRUE if module is marked as 'safe'

    size_t nameoffset;          // offset of module name from start of ModuleInfo
    size_t namelen;             // length of module name in characters

    int doDocComment;          // enable generating doc comments for this module
    int doHdrGen;              // enable generating header file for this module

    Module(char *arg, Identifier *ident, int doDocComment, int doHdrGen);
    ~Module();

    static Module *load(Loc loc, Identifiers *packages, Identifier *ident);

    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);
    void toJsonBuffer(OutBuffer *buf);
    const char *kind();
#if !IN_LLVM
    void setDocfile();  // set docfile member
#endif
    bool read(Loc loc); // read file, returns 'true' if succeed, 'false' otherwise.
#if IN_LLVM
    void parse(bool gen_docs = false);       // syntactic parse
#else
    void parse();       // syntactic parse
#endif
    void importAll(Scope *sc);
    void semantic(Scope* unused_sc = NULL);    // semantic analysis
    void semantic2(Scope* unused_sc = NULL);   // pass 2 semantic analysis
    void semantic3(Scope* unused_sc = NULL);   // pass 3 semantic analysis
    void inlineScan();  // scan for functions to inline
#if !IN_LLVM
    void setHdrfile();  // set hdrfile member
#endif
    void genhdrfile();  // generate D import file
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
    Symbol *ssharedctor;        // module shared constructor
    Symbol *sshareddtor;        // module shared destructor
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
    AA *arrayfuncs;

    bool isRoot;
#endif
};


struct ModuleDeclaration
{
    Identifier *id;
    Identifiers *packages;            // array of Identifier's representing packages
    bool safe;

    ModuleDeclaration(Identifiers *packages, Identifier *id, bool safe);

    char *toChars();
};

#endif /* DMD_MODULE_H */
