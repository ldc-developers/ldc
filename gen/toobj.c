
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

#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"

#include "llvm/Target/TargetData.h"
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
#include "gen/elem.h"
#include "gen/logger.h"
#include "gen/tollvm.h"

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

    gTargetData = new llvm::TargetData(ir.module);

    // process module members
    for (int k=0; k < members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(members->data[k]);
        assert(dsym);
        dsym->toObjFile();
    }

    delete gTargetData;
    gTargetData = 0;

    // emit the llvm main function if necessary
    if (ir.emitMain) {
        LLVM_DtoMain();
    }

    // verify the llvm
    std::string verifyErr;
    Logger::println("Verifying module...");
    if (llvm::verifyModule(*ir.module,llvm::ReturnStatusAction,&verifyErr))
    {
        error("%s", verifyErr.c_str());
        fatal();
    }
    else
        Logger::println("Verification passed!");

    // run passes
    // TODO

    /*if (global.params.llvmLL) {
        //assert(0);
        std::ofstream os(llfile->name->toChars());
        //llvm::WriteAssemblyToFile(ir.module, os);
        ir.module->print(os);
    }*/

    // write bytecode
    //if (global.params.llvmBC) {
        Logger::println("Writing LLVM bitcode\n");
        std::ofstream os(bcfile->name->toChars(), std::ios::binary);
        llvm::WriteBitcodeToFile(ir.module, os);
    //}

    delete ir.module;
    gIR = NULL;
}

/* ================================================================== */

// Put out instance of ModuleInfo for this Module

void Module::genmoduleinfo()
{
}

/* ================================================================== */

void Dsymbol::toObjFile()
{
    warning("Ignoring Dsymbol::toObjFile for %s", toChars());
}

/* ================================================================== */

void Declaration::toObjFile()
{
    warning("Ignoring Declaration::toObjFile for %s", toChars());
}

/* ================================================================== */

/// Returns the LLVM style index from a DMD style offset
void AggregateDeclaration::offsetToIndex(unsigned os, std::vector<unsigned>& result)
{
    //Logger::println("checking for offset %u :", os);
    LOG_SCOPE;
    unsigned vos = 0;
    for (unsigned i=0; i<fields.dim; ++i) {
        VarDeclaration* vd = (VarDeclaration*)fields.data[i];
        //Logger::println("found %u", vd->offset);
        if (os == vd->offset) {
            result.push_back(i);
            return;
        }
        else if (vd->type->ty == Tstruct) {
            if (vos + vd->type->size() > os) {
                TypeStruct* ts = (TypeStruct*)vd->type;
                StructDeclaration* sd = ts->sym;
                result.push_back(i);
                sd->offsetToIndex(os - vos, result);
                return;
            }
        }
        vos += vd->offset;
    }
    assert(0 && "Offset not found in any aggregate field");
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

/// Returns the LLVM style index from a DMD style offset
/// Handles class inheritance
void ClassDeclaration::offsetToIndex(unsigned os, std::vector<unsigned>& result)
{
    unsigned idx = 0;
    unsigned r = LLVM_ClassOffsetToIndex(this, os, idx);
    assert(r != (unsigned)-1 && "Offset not found in any aggregate field");
    result.push_back(r+1); // vtable is 0
}

/* ================================================================== */

void InterfaceDeclaration::toObjFile()
{
    warning("Ignoring InterfaceDeclaration::toObjFile for %s", toChars());
}

/* ================================================================== */

void StructDeclaration::toObjFile()
{
    TypeStruct* ts = (TypeStruct*)type;
    if (llvmType != 0)
        return;

    static int sdi = 0;
    Logger::print("StructDeclaration::toObjFile(%d): %s\n", sdi++, toChars());
    LOG_SCOPE;

    gIR->structs.push_back(IRStruct(ts));

    std::vector<FuncDeclaration*> mfs;

    for (int k=0; k < members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(members->data[k]);

        // need late generation of member functions
        // they need the llvm::StructType to exist to take the 'this' parameter
        if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
            mfs.push_back(fd);
        }
        else {
            dsym->toObjFile();
        }
    }

    if (gIR->topstruct().fields.empty())
    {
        gIR->topstruct().fields.push_back(llvm::Type::Int8Ty);
        gIR->topstruct().inits.push_back(llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false));
    }

    llvm::StructType* structtype = llvm::StructType::get(gIR->topstruct().fields);

    // refine abstract types for stuff like: struct S{S* next;}
    if (gIR->topstruct().recty != 0)
    {
        llvm::PATypeHolder& pa = gIR->topstruct().recty;
        llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(structtype);
        structtype = llvm::cast<llvm::StructType>(pa.get());
    }

    ts->llvmType = structtype;
    llvmType = structtype;

    if (parent->isModule()) {
        gIR->module->addTypeName(mangle(),ts->llvmType);
    }

    // generate static data
    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;
    llvm::Constant* _init = 0;

    // always generate the constant initalizer
    if (!zeroInit) {
        Logger::println("Not zero initialized");
        //assert(tk == gIR->topstruct().size());
        #ifndef LLVMD_NO_LOGGER
        Logger::cout() << *structtype << '\n';
        for (size_t k=0; k<gIR->topstruct().inits.size(); ++k) {
            Logger::cout() << "Type:" << '\n';
            Logger::cout() << *gIR->topstruct().inits[k]->getType() << '\n';
            Logger::cout() << "Value:" << '\n';
            Logger::cout() << *gIR->topstruct().inits[k] << '\n';
        }
        Logger::cout() << "Initializer printed" << '\n';
        #endif
        llvmInitZ = llvm::ConstantStruct::get(structtype,gIR->topstruct().inits);
    }
    else {
        Logger::println("Zero initialized");
        llvmInitZ = llvm::ConstantAggregateZero::get(structtype);
    }

    // only provide the constant initializer for the defining module
    if (getModule() == gIR->dmodule)
    {
        _init = llvmInitZ;
    }

    std::string initname(mangle());
    initname.append("__initZ");
    llvm::GlobalVariable* initvar = new llvm::GlobalVariable(ts->llvmType, true, _linkage, _init, initname, gIR->module);
    ts->llvmInit = initvar;

    // generate member functions
    size_t n = mfs.size();
    for (size_t i=0; i<n; ++i) {
        mfs[i]->toObjFile();
    }

    llvmDModule = gIR->dmodule;

    gIR->structs.pop_back();

    // generate typeinfo
    type->getTypeInfo(NULL);    // generate TypeInfo
}

/* ================================================================== */

static void LLVM_AddBaseClassData(BaseClasses* bcs)
{
    // add base class data members first
    for (int j=0; j<bcs->dim; j++)
    {
        BaseClass* bc = (BaseClass*)(bcs->data[j]);
        assert(bc);
        LLVM_AddBaseClassData(&bc->base->baseclasses);
        for (int k=0; k < bc->base->members->dim; k++) {
            Dsymbol* dsym = (Dsymbol*)(bc->base->members->data[k]);
            if (dsym->isVarDeclaration())
            {
                dsym->toObjFile();
            }
        }
    }
}

void ClassDeclaration::toObjFile()
{
    TypeClass* ts = (TypeClass*)type;
    if (ts->llvmType != 0 || llvmInProgress)
        return;

    llvmInProgress = true;

    static int fdi = 0;
    Logger::print("ClassDeclaration::toObjFile(%d): %s\n", fdi++, toChars());
    LOG_SCOPE;

    gIR->structs.push_back(IRStruct(ts));
    gIR->classes.push_back(this);
    gIR->classmethods.push_back(IRState::FuncDeclVec());
    gIR->queueClassMethods.push_back(true);

    // add vtable
    llvm::PATypeHolder pa = llvm::OpaqueType::get();
    const llvm::Type* vtabty = llvm::PointerType::get(pa);
    gIR->topstruct().fields.push_back(vtabty);
    gIR->topstruct().inits.push_back(0);

    // base classes first
    LLVM_AddBaseClassData(&baseclasses);

    // then add own members
    for (int k=0; k < members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(members->data[k]);
        dsym->toObjFile();
    }

    llvm::StructType* structtype = llvm::StructType::get(gIR->topstruct().fields);
    // refine abstract types for stuff like: class C {C next;}
    if (gIR->topstruct().recty != 0)
    {
        llvm::PATypeHolder& pa = gIR->topstruct().recty;
        llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(structtype);
        structtype = llvm::cast<llvm::StructType>(pa.get());
    }

    ts->llvmType = structtype;
    llvmType = structtype;

    bool define_vtable = false;
    if (parent->isModule()) {
        gIR->module->addTypeName(mangle(),ts->llvmType);
        define_vtable = (getModule() == gIR->dmodule);
    }
    else {
        assert(0 && "class parent is not a module");
    }

    // generate vtable
    llvm::GlobalVariable* svtblVar = 0;
    std::vector<llvm::Constant*> sinits;
    std::vector<const llvm::Type*> sinits_ty;
    sinits.reserve(vtbl.dim);
    sinits_ty.reserve(vtbl.dim);

    for (int k=0; k < vtbl.dim; k++)
    {
        Dsymbol* dsym = (Dsymbol*)vtbl.data[k];
        assert(dsym);
        //Logger::cout() << "vtblsym: " << dsym->toChars() << '\n';

        if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
            fd->toObjFile();
            assert(fd->llvmValue);
            llvm::Constant* c = llvm::cast<llvm::Constant>(fd->llvmValue);
            sinits.push_back(c);
            sinits_ty.push_back(c->getType());
        }
        else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
            const llvm::Type* cty = llvm::PointerType::get(llvm::Type::Int8Ty);
            llvm::Constant* c = llvm::Constant::getNullValue(cty);
            sinits.push_back(c);
            sinits_ty.push_back(cty);
        }
        else
        assert(0);
    }

    const llvm::StructType* svtbl_ty = 0;
    if (!sinits.empty())
    {
        llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;

        std::string varname(mangle());
        varname.append("__vtblZ");
        std::string styname(mangle());
        styname.append("__vtblTy");

        svtbl_ty = llvm::StructType::get(sinits_ty);
        gIR->module->addTypeName(styname, svtbl_ty);
        svtblVar = new llvm::GlobalVariable(svtbl_ty, true, _linkage, 0, varname, gIR->module);

        if (define_vtable) {
            svtblVar->setInitializer(llvm::ConstantStruct::get(svtbl_ty, sinits));
        }
        llvmVtbl = svtblVar;
    }

    ////////////////////////////////////////////////////////////////////////////////

    // refine for final vtable type
    llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(svtbl_ty);
    svtbl_ty = llvm::cast<llvm::StructType>(pa.get());
    structtype = llvm::cast<llvm::StructType>(gIR->topstruct().recty.get());
    ts->llvmType = structtype;
    llvmType = structtype;

    // generate initializer
    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;
    llvm::Constant* _init = 0;

    // first field is always the vtable
    assert(svtblVar != 0);
    gIR->topstruct().inits[0] = svtblVar;

    _init = llvm::ConstantStruct::get(structtype,gIR->topstruct().inits);
    assert(_init);
    std::string initname(mangle());
    initname.append("__initZ");
    //Logger::cout() << *_init << '\n';
    llvm::GlobalVariable* initvar = new llvm::GlobalVariable(ts->llvmType, true, _linkage, 0, initname, gIR->module);
    ts->llvmInit = initvar;
    if (define_vtable) {
        initvar->setInitializer(_init);
    }

    // generate member function definitions
    gIR->queueClassMethods.back() = false;
    IRState::FuncDeclVec& mfs = gIR->classmethods.back();
    size_t n = mfs.size();
    for (size_t i=0; i<n; ++i) {
        mfs[i]->toObjFile();
    }

    gIR->queueClassMethods.pop_back();
    gIR->classmethods.pop_back();
    gIR->classes.pop_back();
    gIR->structs.pop_back();

    llvmInProgress = false;
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
    static int vdi = 0;
    Logger::print("VarDeclaration::toObjFile(%d): %s | %s\n", vdi++, toChars(), type->toChars());
    LOG_SCOPE;
    llvm::Module* M = gIR->module;

    // handle bind pragma
    if (llvmInternal == LLVMbind) {
        Logger::println("var is bound: %s", llvmInternal1);
        llvmValue = M->getGlobalVariable(llvmInternal1);  
        assert(llvmValue);
        return;
    }

    // global variable or magic
    if (isDataseg())
    {
        bool _isconst = isConst();
        if (!_isconst)
            _isconst = (storage_class & STCconst) ? true : false; // doesn't seem to work ):
        llvm::GlobalValue::LinkageTypes _linkage = LLVM_DtoLinkage(protection, storage_class);
        const llvm::Type* _type = LLVM_DtoType(type);
        assert(_type);

        llvm::Constant* _init = 0;
        bool _signed = !type->isunsigned();

        Logger::println("Creating global variable");
        std::string _name(mangle());
        llvm::GlobalVariable* gvar = new llvm::GlobalVariable(_type,_isconst,_linkage,0,_name,M);
        llvmValue = gvar;
        gIR->lvals.push_back(gvar);

        _init = LLVM_DtoInitializer(type, init);
        assert(_init);

        //Logger::cout() << "initializer: " << *_init << '\n';
        if (_type != _init->getType()) {
            Logger::cout() << "got type '" << *_init->getType() << "' expected '" << *_type << "'\n";
            // zero initalizer
            if (_init->isNullValue())
                _init = llvm::Constant::getNullValue(_type);
            // pointer to global constant (struct.init)
            else if (llvm::isa<llvm::GlobalVariable>(_init))
            {
                assert(_init->getType()->getContainedType(0) == _type);
                llvm::GlobalVariable* gv = llvm::cast<llvm::GlobalVariable>(_init);
                assert(type->ty == Tstruct);
                TypeStruct* ts = (TypeStruct*)type;
                assert(ts->sym->llvmInitZ);
                _init = ts->sym->llvmInitZ;
            }
            // array single value init
            else if (llvm::isa<llvm::ArrayType>(_type))
            {
                const llvm::ArrayType* at = llvm::cast<llvm::ArrayType>(_type);
                assert(_type->getContainedType(0) == _init->getType());
                std::vector<llvm::Constant*> initvals;
                initvals.resize(at->getNumElements(), _init);
                _init = llvm::ConstantArray::get(at, initvals);
            }
            else {
                Logger::cout() << "Unexpected initializer type: " << *_type << '\n';
                //assert(0);
            }
        }

        gIR->lvals.pop_back();

        gvar->setInitializer(_init);

        //if (storage_class & STCprivate)
        //    gvar->setVisibility(llvm::GlobalValue::ProtectedVisibility);
    }

    // inside aggregate declaration. declare a field.
    else
    {
        Logger::println("Aggregate var declaration: '%s' offset=%d", toChars(), offset);

        const llvm::Type* _type = LLVM_DtoType(type);
        gIR->topstruct().fields.push_back(_type);

        llvm::Constant* _init = LLVM_DtoInitializer(type, init);
        if (_type != _init->getType())
        {
            if (llvm::isa<llvm::ArrayType>(_type))
            {
                const llvm::ArrayType* arrty = llvm::cast<llvm::ArrayType>(_type);
                uint64_t n = arrty->getNumElements();
                std::vector<llvm::Constant*> vals(n,_init);
                _init = llvm::ConstantArray::get(arrty, vals);
            }
            else if (llvm::isa<llvm::StructType>(_type)) {
                const llvm::StructType* structty = llvm::cast<llvm::StructType>(_type);
                TypeStruct* ts = (TypeStruct*)type;
                assert(ts);
                assert(ts->sym);
                assert(ts->sym->llvmInitZ);
                _init = ts->sym->llvmInitZ;
            }
            else
            assert(0);
        }
        gIR->topstruct().inits.push_back(_init);
    }

    Logger::println("VarDeclaration::toObjFile is done");
}

/* ================================================================== */

void TypedefDeclaration::toObjFile()
{
    static int tdi = 0;
    Logger::print("TypedefDeclaration::toObjFile(%d): %s\n", tdi++, toChars());
    LOG_SCOPE;

    // TODO
}

/* ================================================================== */

void EnumDeclaration::toObjFile()
{
    warning("Ignoring EnumDeclaration::toObjFile for %s", toChars());
}

/* ================================================================== */

void FuncDeclaration::toObjFile()
{
    if (llvmDModule) {
        assert(llvmValue != 0);
        return;
    }

    llvm::Function* func = LLVM_DtoDeclareFunction(this);

    if (!gIR->queueClassMethods.empty() && gIR->queueClassMethods.back()) {
        if (!llvmQueued) {
            Logger::println("queueing %s", toChars());
            assert(!gIR->classmethods.empty());
            gIR->classmethods.back().push_back(this);
            llvmQueued = true;
        }
        return; // we wait with the definition as they might invoke a virtual method and the vtable is not yet complete
    }

    TypeFunction* f = (TypeFunction*)type;
    assert(f->llvmType);
    const llvm::FunctionType* functype = llvm::cast<llvm::FunctionType>(llvmValue->getType()->getContainedType(0));

    // only members of the current module maybe be defined
    if (getModule() == gIR->dmodule || parent->isTemplateInstance())
    {
        llvmDModule = gIR->dmodule;

        bool allow_fbody = true;
        // handle static constructor / destructor
        if (isStaticCtorDeclaration() || isStaticDtorDeclaration()) {
            const llvm::ArrayType* sctor_type = llvm::ArrayType::get(llvm::PointerType::get(functype),1);
            //Logger::cout() << "static ctor type: " << *sctor_type << '\n';

            llvm::Constant* sctor_func = llvm::cast<llvm::Constant>(llvmValue);
            //Logger::cout() << "static ctor func: " << *sctor_func << '\n';

            llvm::Constant* sctor_init = 0;
            if (llvmInternal == LLVMnull)
            {
                llvm::Constant* sctor_init_null = llvm::Constant::getNullValue(sctor_func->getType());
                sctor_init = llvm::ConstantArray::get(sctor_type,&sctor_init_null,1);
                allow_fbody = false;
            }
            else
            {
                sctor_init = llvm::ConstantArray::get(sctor_type,&sctor_func,1);
            }

            //Logger::cout() << "static ctor init: " << *sctor_init << '\n';

            // output the llvm.global_ctors array
            const char* varname = isStaticCtorDeclaration() ? "_d_module_ctor_array" : "_d_module_dtor_array";
            llvm::GlobalVariable* sctor_arr = new llvm::GlobalVariable(sctor_type, false, llvm::GlobalValue::AppendingLinkage, sctor_init, varname, gIR->module);
        }

        // function definition
        if (allow_fbody && fbody != 0)
        {
            gIR->funcdecls.push_back(this);

            // first make absolutely sure the type is up to date
            f->llvmType = llvmValue->getType()->getContainedType(0);

            // this handling
            if (f->llvmUsesThis) {
                if (f->llvmRetInPtr)
                    llvmThisVar = ++func->arg_begin();
                else
                    llvmThisVar = func->arg_begin();
                assert(llvmThisVar != 0);
            }

            if (isMain())
                gIR->emitMain = true;

            gIR->funcs.push(func);
            gIR->functypes.push(f);

            IRScope irs;
            irs.begin = new llvm::BasicBlock("entry",func);
            irs.end = new llvm::BasicBlock("endentry",func);

            //assert(gIR->scopes.empty());
            gIR->scopes.push_back(irs);

                // create alloca point
                f->llvmAllocaPoint = new llvm::BitCastInst(llvm::ConstantInt::get(llvm::Type::Int32Ty,0,false),llvm::Type::Int32Ty,"alloca point",gIR->scopebb());

                // output function body
                fbody->toIR(gIR);

                // llvm requires all basic blocks to end with a TerminatorInst but DMD does not put a return statement
                // in automatically, so we do it here.
                if (!isMain() && (gIR->scopebb()->empty() || !llvm::isa<llvm::TerminatorInst>(gIR->scopebb()->back()))) {
                    // pass the previous block into this block
                    //new llvm::BranchInst(irs.end, irs.begin);
                    new llvm::ReturnInst(gIR->scopebb());
                }

                // erase alloca point
                f->llvmAllocaPoint->eraseFromParent();
                f->llvmAllocaPoint = 0;

            gIR->scopes.pop_back();
            //assert(gIR->scopes.empty());

            gIR->functypes.pop();
            gIR->funcs.pop();

            // get rid of the endentry block, it's never used
            assert(!func->getBasicBlockList().empty());
            func->getBasicBlockList().pop_back();

            // if the last block is empty now, it must be unreachable or it's a bug somewhere else
            // would be nice to figure out how to assert that this is correct
            llvm::BasicBlock* lastbb = &func->getBasicBlockList().back();
            if (lastbb->empty()) {
                // possibly assert(lastbb->getNumPredecessors() == 0); ??? try it out sometime ...
                new llvm::UnreachableInst(lastbb);
            }

            gIR->funcdecls.pop_back();
        }

        // template instances should have weak linkage
        if (parent->isTemplateInstance()) {
            func->setLinkage(llvm::GlobalValue::WeakLinkage);
        }
    }
}
