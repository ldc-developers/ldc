
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
#include "llvm/System/Path.h"

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
#include "gen/llvmhelpers.h"
#include "gen/arrays.h"
#include "gen/structs.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/todebug.h"
#include "gen/runtime.h"

#include "ir/irvar.h"
#include "ir/irmodule.h"

//////////////////////////////////////////////////////////////////////////////////////////

// in gen/optimize.cpp
void llvmdc_optimize_module(llvm::Module* m, char lvl, bool doinline);

//////////////////////////////////////////////////////////////////////////////////////////

void Module::genobjfile()
{
    Logger::cout() << "Generating module: " << (md ? md->toChars() : toChars()) << '\n';
    LOG_SCOPE;

    // start by deleting the old object file
    deleteObjFile();

    // create a new ir state
    // TODO look at making the instance static and moving most functionality into IrModule where it belongs
    IRState ir;
    gIR = &ir;
    ir.dmodule = this;

    // reset all IR data stored in Dsymbols and Types
    IrDsymbol::resetAll();
    IrType::resetAll();

    // module ir state
    // might already exist via import, just overwrite...
    this->ir.irModule = new IrModule(this);

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
    const llvm::TargetMachineRegistry::entry* targetEntry;
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
        DtoDwarfCompileUnit(this);
    }

    // start out by providing opaque for the built-in class types
    if (!ClassDeclaration::object->type->ir.type)
        ClassDeclaration::object->type->ir.type = new llvm::PATypeHolder(llvm::OpaqueType::get());

    if (!Type::typeinfo->type->ir.type)
        Type::typeinfo->type->ir.type = new llvm::PATypeHolder(llvm::OpaqueType::get());

    if (!ClassDeclaration::classinfo->type->ir.type)
        ClassDeclaration::classinfo->type->ir.type = new llvm::PATypeHolder(llvm::OpaqueType::get());

    /*if (!Type::typeinfoclass->type->ir.type)
        Type::typeinfoclass->type->ir.type = new llvm::PATypeHolder(llvm::OpaqueType::get());*/

    // process module members
    for (int k=0; k < members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(members->data[k]);
        assert(dsym);
        dsym->toObjFile();
    }

    // main driver loop
    DtoEmptyAllLists();
    // generate ModuleInfo
    genmoduleinfo();
    // do this again as moduleinfo might have pulled something in!
    DtoEmptyAllLists();

    // emit usedArray
    const LLArrayType* usedTy = LLArrayType::get(getVoidPtrType(), ir.usedArray.size());
    LLConstant* usedInit = LLConstantArray::get(usedTy, ir.usedArray);
    LLGlobalVariable* usedArray = new LLGlobalVariable(usedTy, true, LLGlobalValue::ExternalLinkage, usedInit, "llvm.used", ir.module);
    usedArray->setSection("llvm.metadata");

    // verify the llvm
    if (!global.params.novalidate) {
        std::string verifyErr;
        Logger::println("Verifying module...");
        LOG_SCOPE;
        if (llvm::verifyModule(*ir.module,llvm::ReturnStatusAction,&verifyErr))
        {
            error("%s", verifyErr.c_str());
            fatal();
        }
        else {
            Logger::println("Verification passed!");
        }
    }

    // run optimizer
    llvmdc_optimize_module(ir.module, global.params.optimizeLevel, global.params.llvmInline);

    // eventually do our own path stuff, dmd's is a bit strange.
    typedef llvm::sys::Path LLPath;
    LLPath bcpath;
    LLPath llpath;

    if (global.params.fqnPaths)
    {
        bcpath = LLPath(md->toChars());
        bcpath.appendSuffix("bc");

        llpath = LLPath(md->toChars());
        llpath.appendSuffix("ll");
    }
    else
    {
        bcpath = LLPath(bcfile->name->toChars());
        llpath = LLPath(llfile->name->toChars());
    }

    // write bytecode
    {
        Logger::println("Writing LLVM bitcode\n");
        std::ofstream bos(bcpath.c_str(), std::ios::binary);
        llvm::WriteBitcodeToFile(ir.module, bos);
    }

    // disassemble ?
    if (global.params.disassemble) {
        Logger::println("Writing LLVM asm to: %s\n", llfile->name->toChars());
        std::ofstream aos(llpath.c_str());
        ir.module->print(aos);
    }

    delete ir.module;
    gTargetData = 0;
    gIR = NULL;
}

/* ================================================================== */

// build module ctor

static llvm::Function* build_module_ctor()
{
    if (gIR->ctors.empty())
        return NULL;

    size_t n = gIR->ctors.size();
    if (n == 1)
        return gIR->ctors[0]->ir.irFunc->func;

    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("6__ctorZ");

    std::vector<const LLType*> argsTy;
    const llvm::FunctionType* fnTy = llvm::FunctionType::get(LLType::VoidTy,argsTy,false);
    assert(gIR->module->getFunction(name) == NULL);
    llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, name, gIR->module);
    fn->setCallingConv(llvm::CallingConv::Fast);

    llvm::BasicBlock* bb = llvm::BasicBlock::Create("entry", fn);
    IRBuilder builder(bb);

    for (size_t i=0; i<n; i++) {
        llvm::Function* f = gIR->ctors[i]->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(llvm::CallingConv::Fast);
    }

    builder.CreateRetVoid();
    return fn;
}

// build module dtor

static llvm::Function* build_module_dtor()
{
    if (gIR->dtors.empty())
        return NULL;

    size_t n = gIR->dtors.size();
    if (n == 1)
        return gIR->dtors[0]->ir.irFunc->func;

    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("6__dtorZ");

    std::vector<const LLType*> argsTy;
    const llvm::FunctionType* fnTy = llvm::FunctionType::get(LLType::VoidTy,argsTy,false);
    assert(gIR->module->getFunction(name) == NULL);
    llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, name, gIR->module);
    fn->setCallingConv(llvm::CallingConv::Fast);

    llvm::BasicBlock* bb = llvm::BasicBlock::Create("entry", fn);
    IRBuilder builder(bb);

    for (size_t i=0; i<n; i++) {
        llvm::Function* f = gIR->dtors[i]->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(llvm::CallingConv::Fast);
    }

    builder.CreateRetVoid();
    return fn;
}

// build module unittest

static llvm::Function* build_module_unittest()
{
    if (gIR->unitTests.empty())
        return NULL;

    size_t n = gIR->unitTests.size();
    if (n == 1)
        return gIR->unitTests[0]->ir.irFunc->func;

    std::string name("_D");
    name.append(gIR->dmodule->mangle());
    name.append("10__unittestZ");

    std::vector<const LLType*> argsTy;
    const llvm::FunctionType* fnTy = llvm::FunctionType::get(LLType::VoidTy,argsTy,false);
    assert(gIR->module->getFunction(name) == NULL);
    llvm::Function* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, name, gIR->module);
    fn->setCallingConv(llvm::CallingConv::Fast);

    llvm::BasicBlock* bb = llvm::BasicBlock::Create("entry", fn);
    IRBuilder builder(bb);

    for (size_t i=0; i<n; i++) {
        llvm::Function* f = gIR->unitTests[i]->ir.irFunc->func;
        llvm::CallInst* call = builder.CreateCall(f,"");
        call->setCallingConv(llvm::CallingConv::Fast);
    }

    builder.CreateRetVoid();
    return fn;
}

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

    // resolve ModuleInfo
    assert(moduleinfo);
    DtoForceConstInitDsymbol(moduleinfo);

    // moduleinfo llvm struct type
    const llvm::StructType* moduleinfoTy = isaStruct(moduleinfo->type->ir.type->get());

    // classinfo llvm struct type
    const llvm::StructType* classinfoTy = isaStruct(ClassDeclaration::classinfo->type->ir.type->get());

    // initializer vector
    std::vector<LLConstant*> initVec;
    LLConstant* c = 0;

    // vtable
    c = moduleinfo->ir.irStruct->vtbl;
    initVec.push_back(c);

    // monitor
    c = getNullPtr(getPtrToType(LLType::Int8Ty));
    initVec.push_back(c);

    // name
    char *name = toPrettyChars();
    c = DtoConstString(name);
    initVec.push_back(c);

    // importedModules[]
    int aimports_dim = aimports.dim;
    std::vector<LLConstant*> importInits;
    for (size_t i = 0; i < aimports.dim; i++)
    {
        Module *m = (Module *)aimports.data[i];
        if (!m->needModuleInfo())
            aimports_dim--;
        else { // declare
            // create name
            std::string m_name("_D");
            m_name.append(m->mangle());
            m_name.append("8__ModuleZ");
            llvm::GlobalVariable* m_gvar = gIR->module->getGlobalVariable(m_name);
            if (!m_gvar) m_gvar = new llvm::GlobalVariable(moduleinfoTy, false, llvm::GlobalValue::ExternalLinkage, NULL, m_name, gIR->module);
            importInits.push_back(m_gvar);
        }
    }
    // has import array?
    if (!importInits.empty())
    {
        const llvm::ArrayType* importArrTy = llvm::ArrayType::get(getPtrToType(moduleinfoTy), importInits.size());
        c = llvm::ConstantArray::get(importArrTy, importInits);
        std::string m_name("_D");
        m_name.append(mangle());
        m_name.append("9__importsZ");
        llvm::GlobalVariable* m_gvar = gIR->module->getGlobalVariable(m_name);
        if (!m_gvar) m_gvar = new llvm::GlobalVariable(importArrTy, true, llvm::GlobalValue::InternalLinkage, c, m_name, gIR->module);
        c = llvm::ConstantExpr::getBitCast(m_gvar, getPtrToType(importArrTy->getElementType()));
        c = DtoConstSlice(DtoConstSize_t(importInits.size()), c);
    }
    else
        c = moduleinfo->ir.irStruct->constInit->getOperand(3);
    initVec.push_back(c);

    // localClasses[]
    ClassDeclarations aclasses;
    //printf("members->dim = %d\n", members->dim);
    for (size_t i = 0; i < members->dim; i++)
    {
        Dsymbol *member;

        member = (Dsymbol *)members->data[i];
        //printf("\tmember '%s'\n", member->toChars());
        member->addLocalClass(&aclasses);
    }
    // fill inits
    std::vector<LLConstant*> classInits;
    for (size_t i = 0; i < aclasses.dim; i++)
    {
        ClassDeclaration* cd = (ClassDeclaration*)aclasses.data[i];
        if (cd->isInterfaceDeclaration())
        {
            Logger::println("skipping interface '%s' in moduleinfo", cd->toPrettyChars());
            continue;
        }
        Logger::println("class: %s", cd->toPrettyChars());
        assert(cd->ir.irStruct->classInfo);
        classInits.push_back(cd->ir.irStruct->classInfo);
    }
    // has class array?
    if (!classInits.empty())
    {
        const llvm::ArrayType* classArrTy = llvm::ArrayType::get(getPtrToType(classinfoTy), classInits.size());
        c = llvm::ConstantArray::get(classArrTy, classInits);
        std::string m_name("_D");
        m_name.append(mangle());
        m_name.append("9__classesZ");
        assert(gIR->module->getGlobalVariable(m_name) == NULL);
        llvm::GlobalVariable* m_gvar = new llvm::GlobalVariable(classArrTy, true, llvm::GlobalValue::InternalLinkage, c, m_name, gIR->module);
        c = llvm::ConstantExpr::getBitCast(m_gvar, getPtrToType(classArrTy->getElementType()));
        c = DtoConstSlice(DtoConstSize_t(classInits.size()), c);
    }
    else
        c = moduleinfo->ir.irStruct->constInit->getOperand(4);
    initVec.push_back(c);

    // flags
    c = DtoConstUint(0);
    if (!needmoduleinfo)
        c = DtoConstUint(4);        // flags (4 means MIstandalone)
    initVec.push_back(c);

    // ctor
    llvm::Function* fctor = build_module_ctor();
    c = fctor ? fctor : moduleinfo->ir.irStruct->constInit->getOperand(6);
    initVec.push_back(c);

    // dtor
    llvm::Function* fdtor = build_module_dtor();
    c = fdtor ? fdtor : moduleinfo->ir.irStruct->constInit->getOperand(7);
    initVec.push_back(c);

    // unitTest
    llvm::Function* unittest = build_module_unittest();
    c = unittest ? unittest : moduleinfo->ir.irStruct->constInit->getOperand(8);
    initVec.push_back(c);

    // xgetMembers
    c = moduleinfo->ir.irStruct->constInit->getOperand(9);
    initVec.push_back(c);

    // ictor
    c = moduleinfo->ir.irStruct->constInit->getOperand(10);
    initVec.push_back(c);

    /*Logger::println("MODULE INFO INITIALIZERS");
    for (size_t i=0; i<initVec.size(); ++i)
    {
        Logger::cout() << *initVec[i] << '\n';
        if (initVec[i]->getType() != moduleinfoTy->getElementType(i))
            assert(0);
    }*/

    // create initializer
    LLConstant* constMI = llvm::ConstantStruct::get(moduleinfoTy, initVec);

    // create name
    std::string MIname("_D");
    MIname.append(mangle());
    MIname.append("8__ModuleZ");

    // declare
    // flags will be modified at runtime so can't make it constant

    llvm::GlobalVariable* gvar = gIR->module->getGlobalVariable(MIname);
    if (!gvar) gvar = new llvm::GlobalVariable(moduleinfoTy, false, llvm::GlobalValue::ExternalLinkage, NULL, MIname, gIR->module);
    gvar->setInitializer(constMI);

    // declare the appending array
    const llvm::ArrayType* appendArrTy = llvm::ArrayType::get(getPtrToType(LLType::Int8Ty), 1);
    std::vector<LLConstant*> appendInits;
    appendInits.push_back(llvm::ConstantExpr::getBitCast(gvar, getPtrToType(LLType::Int8Ty)));
    LLConstant* appendInit = llvm::ConstantArray::get(appendArrTy, appendInits);
    std::string appendName("_d_moduleinfo_array");
    llvm::GlobalVariable* appendVar = new llvm::GlobalVariable(appendArrTy, true, llvm::GlobalValue::AppendingLinkage, appendInit, appendName, gIR->module);
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
    //Logger::println("Ignoring InterfaceDeclaration::toObjFile for %s", toChars());
    gIR->resolveList.push_back(this);
}

/* ================================================================== */

void StructDeclaration::toObjFile()
{
    gIR->resolveList.push_back(this);
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
        // we don't want to touch private static members at all !!!
        if ((prot() == PROTprivate) && getModule() != gIR->dmodule)
            return;

        // don't duplicate work
        if (this->ir.resolved) return;
        this->ir.resolved = true;
        this->ir.declared = true;

        this->ir.irGlobal = new IrGlobal(this);

        Logger::println("parent: %s (%s)", parent->toChars(), parent->kind());

        // handle static local variables
        bool static_local = false;
        bool _isconst = isConst();
        if (parent && parent->isFuncDeclaration())
        {
            static_local = true;
            if (init && init->isExpInitializer()) {
                _isconst = false;
            }
        }

        Logger::println("Creating global variable");

        const LLType* _type = this->ir.irGlobal->type.get();
        llvm::GlobalValue::LinkageTypes _linkage = DtoLinkage(this);
        std::string _name(mangle());

        llvm::GlobalVariable* gvar = new llvm::GlobalVariable(_type,_isconst,_linkage,NULL,_name,gIR->module);
        this->ir.irGlobal->value = gvar;

        Logger::cout() << *gvar << '\n';

        if (static_local)
            DtoConstInitGlobal(this);
        else
            gIR->constInitList.push_back(this);
    }

    // inside aggregate declaration. declare a field.
    else
    {
        Logger::println("Aggregate var declaration: '%s' offset=%d", toChars(), offset);

        const LLType* _type = DtoType(type);
        this->ir.irField = new IrField(this);

        // add the field in the IRStruct
        gIR->topstruct()->offsets.insert(std::make_pair(offset, IrStruct::Offset(this, _type)));
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
