
// Copyright (c) 1999-2004 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <cstddef>
#include <iostream>
#include <fstream>

#include "gen/llvm.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "declaration.h"
#include "statement.h"
#include "enum.h"
#include "aggregate.h"
#include "init.h"
#include "attrib.h"
#include "id.h"
#include "import.h"
#include "template.h"
#include "scope.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/arrays.h"
#include "gen/structs.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/todebug.h"
#include "gen/runtime.h"

//////////////////////////////////////////////////////////////////////////////////////////

void
Module::genobjfile()
{
    Logger::cout() << "Generating module: " << (md ? md->toChars() : toChars()) << '\n';
    LOG_SCOPE;

    // start by deleting the old object file
    deleteObjFile();

    // create a new ir state
    IRState ir;
    gIR = &ir;
    ir.dmodule = this;

    // name the module
    std::string mname(toChars());
    if (md != 0)
        mname = md->toChars();
    ir.module = new llvm::Module(mname);

    // set target stuff
    std::string target_triple(global.params.tt_arch);
    target_triple.append(global.params.tt_os);
    ir.module->setTargetTriple(target_triple);
    ir.module->setDataLayout(global.params.data_layout);

    // heavily inspired by tools/llc/llc.cpp:200-230
    const llvm::TargetMachineRegistry::Entry* targetEntry;
    std::string targetError;
    targetEntry = llvm::TargetMachineRegistry::getClosestStaticTargetForModule(*ir.module, targetError);
    assert(targetEntry && "Failed to find a static target for module");
    std::auto_ptr<llvm::TargetMachine> targetPtr(targetEntry->CtorFn(*ir.module, "")); // TODO: replace "" with features
    assert(targetPtr.get() && "Could not allocate target machine!");
    llvm::TargetMachine &targetMachine = *targetPtr.get();
    gTargetData = targetMachine.getTargetData();

    // debug info
    if (global.params.symdebug) {
        RegisterDwarfSymbols(ir.module);
        ir.dmodule->llvmCompileUnit = DtoDwarfCompileUnit(this,true);
    }

    // start out by providing opaque for the built-in class types
    if (!ClassDeclaration::object->type->llvmType)
        ClassDeclaration::object->type->llvmType = new llvm::PATypeHolder(llvm::OpaqueType::get());

    if (!Type::typeinfo->type->llvmType)
        Type::typeinfo->type->llvmType = new llvm::PATypeHolder(llvm::OpaqueType::get());

    if (!ClassDeclaration::classinfo->type->llvmType)
        ClassDeclaration::classinfo->type->llvmType = new llvm::PATypeHolder(llvm::OpaqueType::get());

    /*if (!Type::typeinfoclass->type->llvmType)
        Type::typeinfoclass->type->llvmType = new llvm::PATypeHolder(llvm::OpaqueType::get());*/

    // process module members
    for (int k=0; k < members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(members->data[k]);
        assert(dsym);
        dsym->toObjFile();
    }

    // main driver loop
    for(;;)
    {
        Dsymbol* dsym;
        if (!ir.resolveList.empty()) {
            dsym = ir.resolveList.front();
            ir.resolveList.pop_front();
            DtoResolveDsymbol(dsym);
        }
        else if (!ir.declareList.empty()) {
            dsym = ir.declareList.front();
            ir.declareList.pop_front();
            DtoDeclareDsymbol(dsym);
        }
        else if (!ir.constInitList.empty()) {
            dsym = ir.constInitList.front();
            ir.constInitList.pop_front();
            DtoConstInitDsymbol(dsym);
        }
        else if (!ir.defineList.empty()) {
            dsym = ir.defineList.front();
            ir.defineList.pop_front();
            DtoDefineDsymbol(dsym);
        }
        else {
            break;
        }
    }

    // generate ModuleInfo
    genmoduleinfo();

    gTargetData = 0;

    // emit the llvm main function if necessary
    if (ir.emitMain) {
        DtoMain();
    }

    // verify the llvm
    if (!global.params.novalidate) {
        std::string verifyErr;
        Logger::println("Verifying module...");
        if (llvm::verifyModule(*ir.module,llvm::ReturnStatusAction,&verifyErr))
        {
            error("%s", verifyErr.c_str());
            fatal();
        }
        else {
            Logger::println("Verification passed!");
        }
    }

    // run passes
    // TODO

    // write bytecode
    {
        Logger::println("Writing LLVM bitcode\n");
        std::ofstream bos(bcfile->name->toChars(), std::ios::binary);
        llvm::WriteBitcodeToFile(ir.module, bos);
    }

    // disassemble ?
    if (global.params.disassemble) {
        Logger::println("Writing LLVM asm to: %s\n", llfile->name->toChars());
        std::ofstream aos(llfile->name->toChars());
        ir.module->print(aos);
    }

    delete ir.module;
    gIR = NULL;
}

/* ================================================================== */

// Put out instance of ModuleInfo for this Module

void Module::genmoduleinfo()
{
//      The layout is:
//        {
//         void **vptr;
//         monitor_t monitor;
//         char[] name;        // class name
//         ModuleInfo importedModules[];
//         ClassInfo localClasses[];
//         uint flags;         // initialization state
//         void *ctor;
//         void *dtor;
//         void *unitTest;
//        }

    if (moduleinfo) {
        Logger::println("moduleinfo");
    }
    if (vmoduleinfo) {
        Logger::println("vmoduleinfo");
    }
    if (needModuleInfo()) {
        Logger::attention("module info is needed but skipped");
    }


    /*
    Symbol *msym = toSymbol();
    unsigned offset;
    unsigned sizeof_ModuleInfo = 12 * PTRSIZE;

    //////////////////////////////////////////////

    csym->Sclass = SCglobal;
    csym->Sfl = FLdata;

//      The layout is:
//        {
//         void **vptr;
//         monitor_t monitor;
//         char[] name;        // class name
//         ModuleInfo importedModules[];
//         ClassInfo localClasses[];
//         uint flags;         // initialization state
//         void *ctor;
//         void *dtor;
//         void *unitTest;
//        }
    dt_t *dt = NULL;

    if (moduleinfo)
    dtxoff(&dt, moduleinfo->toVtblSymbol(), 0, TYnptr); // vtbl for ModuleInfo
    else
    dtdword(&dt, 0);        // BUG: should be an assert()
    dtdword(&dt, 0);            // monitor

    // name[]
    char *name = toPrettyChars();
    size_t namelen = strlen(name);
    dtdword(&dt, namelen);
    dtabytes(&dt, TYnptr, 0, namelen + 1, name);

    ClassDeclarations aclasses;
    int i;

    //printf("members->dim = %d\n", members->dim);
    for (i = 0; i < members->dim; i++)
    {
    Dsymbol *member;

    member = (Dsymbol *)members->data[i];
    //printf("\tmember '%s'\n", member->toChars());
    member->addLocalClass(&aclasses);
    }

    // importedModules[]
    int aimports_dim = aimports.dim;
    for (i = 0; i < aimports.dim; i++)
    {   Module *m = (Module *)aimports.data[i];
    if (!m->needModuleInfo())
        aimports_dim--;
    }
    dtdword(&dt, aimports_dim);
    if (aimports.dim)
    dtxoff(&dt, csym, sizeof_ModuleInfo, TYnptr);
    else
    dtdword(&dt, 0);

    // localClasses[]
    dtdword(&dt, aclasses.dim);
    if (aclasses.dim)
    dtxoff(&dt, csym, sizeof_ModuleInfo + aimports_dim * PTRSIZE, TYnptr);
    else
    dtdword(&dt, 0);

    if (needmoduleinfo)
    dtdword(&dt, 0);        // flags (4 means MIstandalone)
    else
    dtdword(&dt, 4);        // flags (4 means MIstandalone)

    if (sctor)
    dtxoff(&dt, sctor, 0, TYnptr);
    else
    dtdword(&dt, 0);

    if (sdtor)
    dtxoff(&dt, sdtor, 0, TYnptr);
    else
    dtdword(&dt, 0);

    if (stest)
    dtxoff(&dt, stest, 0, TYnptr);
    else
    dtdword(&dt, 0);

    //////////////////////////////////////////////

    for (i = 0; i < aimports.dim; i++)
    {
    Module *m;

    m = (Module *)aimports.data[i];
    if (m->needModuleInfo())
    {   Symbol *s = m->toSymbol();
        s->Sflags |= SFLweak;
        dtxoff(&dt, s, 0, TYnptr);
    }
    }

    for (i = 0; i < aclasses.dim; i++)
    {
    ClassDeclaration *cd;

    cd = (ClassDeclaration *)aclasses.data[i];
    dtxoff(&dt, cd->toSymbol(), 0, TYnptr);
    }

    csym->Sdt = dt;
#if ELFOBJ
    // Cannot be CONST because the startup code sets flag bits in it
    csym->Sseg = DATA;
#endif
    outdata(csym);

    //////////////////////////////////////////////

    obj_moduleinfo(msym);
    */
}

/* ================================================================== */

void Dsymbol::toObjFile()
{
    Logger::println("Ignoring Dsymbol::toObjFile for %s", toChars());
}

/* ================================================================== */

void Declaration::toObjFile()
{
    Logger::println("Ignoring Declaration::toObjFile for %s", toChars());
}

/* ================================================================== */

void InterfaceDeclaration::toObjFile()
{
    Logger::println("Ignoring InterfaceDeclaration::toObjFile for %s", toChars());
}

/* ================================================================== */

void StructDeclaration::toObjFile()
{
    gIR->resolveList.push_back(this);
}

/* ================================================================== */

static unsigned LLVM_ClassOffsetToIndex(ClassDeclaration* cd, unsigned os, unsigned& idx)
{
    // start at the bottom of the inheritance chain
    if (cd->baseClass != 0) {
        unsigned o = LLVM_ClassOffsetToIndex(cd->baseClass, os, idx);
        if (o != (unsigned)-1)
            return o;
    }

    // check this class
    unsigned i;
    for (i=0; i<cd->fields.dim; ++i) {
        VarDeclaration* vd = (VarDeclaration*)cd->fields.data[i];
        if (os == vd->offset)
            return i+idx;
    }
    idx += i;

    return (unsigned)-1;
}

void ClassDeclaration::offsetToIndex(Type* t, unsigned os, std::vector<unsigned>& result)
{
    unsigned idx = 0;
    unsigned r = LLVM_ClassOffsetToIndex(this, os, idx);
    assert(r != (unsigned)-1 && "Offset not found in any aggregate field");
    result.push_back(r+1); // vtable is 0
}

/* ================================================================== */

void ClassDeclaration::toObjFile()
{
    gIR->resolveList.push_back(this);
}

/******************************************
 * Get offset of base class's vtbl[] initializer from start of csym.
 * Returns ~0 if not this csym.
 */

unsigned ClassDeclaration::baseVtblOffset(BaseClass *bc)
{
  return ~0;
}

/* ================================================================== */

void VarDeclaration::toObjFile()
{
    Logger::print("VarDeclaration::toObjFile(): %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (aliassym)
    {
        Logger::println("alias sym");
        toAlias()->toObjFile();
        return;
    }

    // global variable or magic
    if (isDataseg())
    {
        if (llvmResolved) return;
        llvmResolved = true;
        llvmDeclared = true;

        llvmIRGlobal = new IRGlobal(this);

        Logger::println("parent: %s (%s)", parent->toChars(), parent->kind());

        bool _isconst = isConst();
        if (parent && parent->isFuncDeclaration() && init && init->isExpInitializer())
            _isconst = false;

        llvm::GlobalValue::LinkageTypes _linkage;
        bool istempl = false;
        bool static_local = false;
        if ((storage_class & STCcomdat) || (parent && DtoIsTemplateInstance(parent))) {
            _linkage = llvm::GlobalValue::WeakLinkage;
            istempl = true;
        }
        else if (parent && parent->isFuncDeclaration()) {
            _linkage = llvm::GlobalValue::InternalLinkage;
            static_local = true;
        }
        else
            _linkage = DtoLinkage(protection, storage_class);

        const llvm::Type* _type = llvmIRGlobal->type.get();

        Logger::println("Creating global variable");
        std::string _name(mangle());

        llvm::GlobalVariable* gvar = new llvm::GlobalVariable(_type,_isconst,_linkage,NULL,_name,gIR->module);
        llvmValue = gvar;

        if (static_local)
            DtoConstInitGlobal(this);
        else
            gIR->constInitList.push_back(this);

        //if (storage_class & STCprivate)
        //    gvar->setVisibility(llvm::GlobalValue::ProtectedVisibility);
    }

    // inside aggregate declaration. declare a field.
    else
    {
        Logger::println("Aggregate var declaration: '%s' offset=%d", toChars(), offset);

        const llvm::Type* _type = DtoType(type);

        // add the field in the IRStruct
        gIR->topstruct()->offsets.insert(std::make_pair(offset, IRStruct::Offset(this, _type)));
    }

    Logger::println("VarDeclaration::toObjFile is done");
}

/* ================================================================== */

void TypedefDeclaration::toObjFile()
{
    static int tdi = 0;
    Logger::print("TypedefDeclaration::toObjFile(%d): %s\n", tdi++, toChars());
    LOG_SCOPE;

    // generate typeinfo
    type->getTypeInfo(NULL);
}

/* ================================================================== */

void EnumDeclaration::toObjFile()
{
    Logger::println("Ignoring EnumDeclaration::toObjFile for %s", toChars());
}

/* ================================================================== */

void FuncDeclaration::toObjFile()
{
    gIR->resolveList.push_back(this);
}
