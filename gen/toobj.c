
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
#include "irstate.h"
#include "elem.h"
#include "logger.h"

#include "tollvm.h"

//////////////////////////////////////////////////////////////////////////////////////////

void
Module::genobjfile()
{
    Logger::cout() << "Generating module: " << (md ? md->toChars() : toChars()) << '\n';
    LOG_SCOPE;

    deleteObjFile();

    IRState ir;
    gIR = &ir;

    ir.dmodule = this;

    std::string mname(toChars());
    if (md != 0)
        mname = md->toChars();
    ir.module = new llvm::Module(mname);

    std::string target_triple(global.params.tt_arch);
    target_triple.append(global.params.tt_os);
    ir.module->setTargetTriple(target_triple);
    ir.module->setDataLayout(global.params.data_layout);

    gTargetData = new llvm::TargetData(ir.module);

    for (int k=0; k < members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(members->data[k]);
        assert(dsym);
        dsym->toObjFile();
    }

    delete gTargetData;
    gTargetData = 0;

    std::string verifyErr;
    if (llvm::verifyModule(*ir.module,llvm::ReturnStatusAction,&verifyErr))
    {
        error("%s", verifyErr.c_str());
        fatal();
    }

    if (ir.emitMain) {
        LLVM_DtoMain();
    }

    // run passes
    // TODO

    /*if (global.params.llvmLL) {
        //assert(0);
        std::ofstream os(llfile->name->toChars());
        //llvm::WriteAssemblyToFile(ir.module, os);
        ir.module->print(os);
    }*/

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

unsigned AggregateDeclaration::offsetToIndex(unsigned os)
{
    for (unsigned i=0; i<fields.dim; ++i) {
        VarDeclaration* vd = (VarDeclaration*)fields.data[i];
        if (os == vd->offset)
            return i;
    }
    assert(0 && "Offset not found in any aggregate field");
    return 0;
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

unsigned ClassDeclaration::offsetToIndex(unsigned os)
{
    unsigned idx = 0;
    unsigned r = LLVM_ClassOffsetToIndex(this, os, idx);
    assert(r != (unsigned)-1 && "Offset not found in any aggregate field");
    return r+1; // vtable is 0
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

    gIR->structs.push_back(IRStruct());
    gIR->classes.push_back(this);
    gIR->classmethods.push_back(IRState::FuncDeclVec());
    gIR->queueClassMethods.push_back(true);

    // add vtable
    const llvm::Type* vtabty = llvm::PointerType::get(llvm::Type::Int8Ty);
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
    ts->llvmType = structtype;
    llvmType = structtype;

    bool emit_vtable = false;
    bool define_vtable = false;
    if (parent->isModule()) {
        gIR->module->addTypeName(mangle(),ts->llvmType);
        emit_vtable = true;
        define_vtable = (getModule() == gIR->dmodule);
    }
    else {
        assert(0 && "class parent is not a module");
    }

    // generate member functions
    gIR->queueClassMethods.back() = false;
    IRState::FuncDeclVec& mfs = gIR->classmethods.back();
    size_t n = mfs.size();
    for (size_t i=0; i<n; ++i) {
        mfs[i]->toObjFile();
    }

    // create vtable initializer
    if (emit_vtable)
    {
        llvm::GlobalVariable* vtblVar = 0;
        std::vector<llvm::Constant*> inits;
        inits.reserve(vtbl.dim);
        for (int k=0; k < vtbl.dim; k++)
        {
            Dsymbol* dsym = (Dsymbol*)vtbl.data[k];
            assert(dsym);
            //Logger::cout() << "vtblsym: " << dsym->toChars() << '\n';

            if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
                fd->toObjFile();
                Logger::cout() << "casting to constant" << *fd->llvmValue << '\n';
                llvm::Constant* c = llvm::cast<llvm::Constant>(fd->llvmValue);
                c = llvm::ConstantExpr::getBitCast(c, llvm::PointerType::get(llvm::Type::Int8Ty));
                inits.push_back(c);
            }
            else if (ClassDeclaration* cd = dsym->isClassDeclaration()) {
                llvm::Constant* c = llvm::Constant::getNullValue(llvm::PointerType::get(llvm::Type::Int8Ty));
                inits.push_back(c);
            }
            else
            assert(0);
        }
        if (!inits.empty())
        {
            llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;
            std::string varname(mangle());
            varname.append("__vtblZ");
            const llvm::ArrayType* vtbl_ty = llvm::ArrayType::get(llvm::PointerType::get(llvm::Type::Int8Ty), inits.size());
            vtblVar = new llvm::GlobalVariable(vtbl_ty, true, _linkage, 0, varname, gIR->module);
            if (define_vtable) {
                //Logger::cout() << "vtbl:::" << '\n' << *vtbl_st << '\n';// << " == | == " << _init << '\n';
                llvm::Constant* _init = llvm::ConstantArray::get(vtbl_ty, inits);
                vtblVar->setInitializer(_init);
            }
            llvmVtbl = vtblVar;
        }

        ////////////////////////////////////////////////////////////////////////////////

        // generate initializer
        llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::ExternalLinkage;
        llvm::Constant* _init = 0;

        // first field is always the vtable
        assert(vtblVar != 0);
        llvm::Constant* vtbl_init_var = llvm::ConstantExpr::getBitCast(vtblVar, llvm::PointerType::get(llvm::Type::Int8Ty));
        gIR->topstruct().inits[0] = vtbl_init_var;

        //assert(tk == gIR->topstruct().size());
        #ifndef LLVMD_NO_LOGGER
        Logger::cout() << *structtype << '\n';
        for (size_t k=0; k<gIR->topstruct().inits.size(); ++k)
            Logger::cout() << *gIR->topstruct().inits[k] << '\n';
        #endif
        _init = llvm::ConstantStruct::get(structtype,gIR->topstruct().inits);
        assert(_init);
        std::string initname(mangle());
        initname.append("__initZ");
        Logger::cout() << *_init << '\n';
        llvm::GlobalVariable* initvar = new llvm::GlobalVariable(ts->llvmType, true, _linkage, 0, initname, gIR->module);
        ts->llvmInit = initvar;
        if (define_vtable) {
            initvar->setInitializer(_init);
        }
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
    if (!parent || parent->isModule())
    {
        bool _isconst = isConst();
        if (!_isconst)
            _isconst = (storage_class & STCconst) ? true : false; // doesn't seem to work ):
        llvm::GlobalValue::LinkageTypes _linkage = LLVM_DtoLinkage(protection, storage_class);
        const llvm::Type* _type = LLVM_DtoType(type);

        llvm::Constant* _init = 0;
        bool _signed = !type->isunsigned();

        _init = LLVM_DtoInitializer(type, init);

        assert(_type);
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
            else
            assert(0);
        }

        Logger::println("Creating global variable");
        std::string _name(mangle());
        llvm::GlobalVariable* gvar = new llvm::GlobalVariable(_type,_isconst,_linkage,_init,_name,M);
        llvmValue = gvar;

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
    if (llvmValue != 0 && llvmDModule == gIR->dmodule) {
        return;
    }

    // has already been pulled in by a reference to (
    if (!gIR->queueClassMethods.empty() && gIR->queueClassMethods.back()) {
        Logger::println("queueing %s", toChars());
        assert(!gIR->classmethods.empty());
        gIR->classmethods.back().push_back(this);
        return; // will be generated later when the this parameter has a type
    }

    static int fdi = 0;
    Logger::print("FuncDeclaration::toObjFile(%d,%s): %s\n", fdi++, needThis()?"this":"static",toChars());
    LOG_SCOPE;

    if (llvmInternal == LLVMintrinsic && fbody) {
        error("intrinsics cannot have function bodies");
        fatal();
    }

    TypeFunction* f = (TypeFunction*)type;
    assert(f != 0);

    // return value type
    const llvm::Type* rettype;
    const llvm::Type* actualRettype;
    Type* rt = f->next;
    bool retinptr = false;
    bool usesthis = false;

    if (isMain()) {
        rettype = llvm::Type::Int32Ty;
        actualRettype = rettype;
        gIR->emitMain = true;
    }
    else if (rt) {
        if (rt->ty == Tstruct || rt->ty == Tdelegate || rt->ty == Tarray) {
            rettype = llvm::PointerType::get(LLVM_DtoType(rt));
            actualRettype = llvm::Type::VoidTy;
            f->llvmRetInPtr = retinptr = true;
        }
        else {
            rettype = LLVM_DtoType(rt);
            actualRettype = rettype;
        }
    }
    else {
        assert(0);
    }

    // parameter types
    std::vector<const llvm::Type*> paramvec;

    if (retinptr) {
        Logger::print("returning through pointer parameter\n");
        paramvec.push_back(rettype);
    }

    if (needThis()) {
        if (AggregateDeclaration* ad = isMember()) {
            Logger::print("isMember = this is: %s\n", ad->type->toChars());
            const llvm::Type* thisty = LLVM_DtoType(ad->type);
            if (llvm::isa<llvm::StructType>(thisty))
                thisty = llvm::PointerType::get(thisty);
            paramvec.push_back(thisty);
            usesthis = true;
        }
        else
        assert(0);
    }

    size_t n = Argument::dim(f->parameters);
    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = arg->type;
        assert(argT);

        if ((arg->storageClass & STCref) || (arg->storageClass & STCout)) {
            //assert(arg->vardecl);
            //arg->vardecl->refparam = true;
        }
        else
            arg->llvmCopy = true;

        const llvm::Type* at = LLVM_DtoType(argT);
        if (llvm::isa<llvm::StructType>(at)) {
            Logger::println("struct param");
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else if (llvm::isa<llvm::ArrayType>(at)) {
            Logger::println("sarray param");
            assert(argT->ty == Tsarray);
            //paramvec.push_back(llvm::PointerType::get(at->getContainedType(0)));
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else {
            if (!arg->llvmCopy) {
                Logger::println("ref param");
                at = llvm::PointerType::get(at);
            }
            else {
                Logger::println("in param");
            }
            paramvec.push_back(at);
        }
    }
    
    // construct function
    bool isvararg = f->varargs;
    llvm::FunctionType* functype = llvm::FunctionType::get(actualRettype, paramvec, isvararg);
    
    // mangled name
    char* mangled_name = (llvmInternal == LLVMintrinsic) ? llvmInternal1 : mangle();
    llvm::Function* func = gIR->module->getFunction(mangled_name);
    
    // make the function
    /*if (func != 0) {
        llvmValue = func;
        f->llvmType = functype;
        return; // already pulled in from a forward declaration
    }
    else */
    if (func == 0) {
        func = new llvm::Function(functype,LLVM_DtoLinkage(protection, storage_class),mangled_name,gIR->module);
    }

    if (llvmInternal != LLVMintrinsic)
        func->setCallingConv(LLVM_DtoCallingConv(f->linkage));

    llvmValue = func;
    f->llvmType = functype;

    if (isMain()) {
        gIR->mainFunc = func;
    }

    // name parameters
    llvm::Function::arg_iterator iarg = func->arg_begin();
    int k = 0;
    int nunnamed = 0;
    if (retinptr) {
        iarg->setName("retval");
        f->llvmRetArg = iarg;
        ++iarg;
    }
    if (usesthis) {
        iarg->setName("this");
        ++iarg;
    }
    for (; iarg != func->arg_end(); ++iarg)
    {
        Argument* arg = Argument::getNth(f->parameters, k++);
        //arg->llvmValue = iarg;
        //printf("identifier: '%s' %p\n", arg->ident->toChars(), arg->ident);
        if (arg->ident != 0) {
            if (arg->vardecl) {
                arg->vardecl->llvmValue = iarg;
            }
            iarg->setName(arg->ident->toChars());
        }
        else {
            ++nunnamed;
        }
    }

    // only members of the current module maybe be defined
    if (getModule() == gIR->dmodule)
    {
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
            assert(nunnamed == 0);
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
            func->getBasicBlockList().pop_back();

            // if the last block is empty now, it must be unreachable or it's a bug somewhere else
            llvm::BasicBlock* lastbb = &func->getBasicBlockList().back();
            if (lastbb->empty()) {
                new llvm::UnreachableInst(lastbb);
            }
        }
    }
    else
    {
        Logger::println("only declaration");
    }

    llvmDModule = gIR->dmodule;

    Logger::println("FuncDeclaration done\n");
}
