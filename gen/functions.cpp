#include "gen/llvm.h"

#include "mtype.h"
#include "aggregate.h"
#include "init.h"
#include "declaration.h"
#include "template.h"
#include "module.h"
#include "statement.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/runtime.h"
#include "gen/arrays.h"
#include "gen/logger.h"
#include "gen/functions.h"
#include "gen/todebug.h"
#include "gen/classes.h"

const llvm::FunctionType* DtoFunctionType(Type* type, const llvm::Type* thistype, bool ismain)
{
    TypeFunction* f = (TypeFunction*)type;
    assert(f != 0);

    if (type->llvmType != NULL) {
        return llvm::cast<llvm::FunctionType>(type->llvmType->get());
    }

    bool typesafeVararg = false;
    if (f->linkage == LINKd && f->varargs == 1) {
        typesafeVararg = true;
    }

    // return value type
    const llvm::Type* rettype;
    const llvm::Type* actualRettype;
    Type* rt = f->next;
    bool retinptr = false;
    bool usesthis = false;

    if (ismain) {
        rettype = llvm::Type::Int32Ty;
        actualRettype = rettype;
    }
    else {
        assert(rt);
        Type* rtfin = DtoDType(rt);
        if (DtoIsPassedByRef(rt)) {
            rettype = llvm::PointerType::get(DtoType(rt));
            actualRettype = llvm::Type::VoidTy;
            f->llvmRetInPtr = retinptr = true;
        }
        else {
            rettype = DtoType(rt);
            actualRettype = rettype;
        }
    }

    // parameter types
    std::vector<const llvm::Type*> paramvec;

    if (retinptr) {
        Logger::cout() << "returning through pointer parameter: " << *rettype << '\n';
        paramvec.push_back(rettype);
    }

    if (thistype) {
        paramvec.push_back(thistype);
        usesthis = true;
    }

    if (typesafeVararg) {
        ClassDeclaration* ti = Type::typeinfo;
        ti->toObjFile();
        DtoForceConstInitDsymbol(ti);
        assert(ti->llvmInitZ);
        std::vector<const llvm::Type*> types;
        types.push_back(DtoSize_t());
        types.push_back(llvm::PointerType::get(llvm::PointerType::get(ti->llvmInitZ->getType())));
        const llvm::Type* t1 = llvm::StructType::get(types);
        paramvec.push_back(llvm::PointerType::get(t1));
        paramvec.push_back(llvm::PointerType::get(llvm::Type::Int8Ty));
    }

    size_t n = Argument::dim(f->parameters);

    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = DtoDType(arg->type);
        assert(argT);

        if ((arg->storageClass & STCref) || (arg->storageClass & STCout)) {
            //assert(arg->vardecl);
            //arg->vardecl->refparam = true;
        }
        else
            arg->llvmCopy = true;

        const llvm::Type* at = DtoType(argT);
        if (isaStruct(at)) {
            Logger::println("struct param");
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else if (isaArray(at)) {
            Logger::println("sarray param");
            assert(argT->ty == Tsarray);
            //paramvec.push_back(llvm::PointerType::get(at->getContainedType(0)));
            paramvec.push_back(llvm::PointerType::get(at));
        }
        else if (llvm::isa<llvm::OpaqueType>(at)) {
            Logger::println("opaque param");
            assert(argT->ty == Tstruct || argT->ty == Tclass);
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

    // construct function type
    bool isvararg = !typesafeVararg && f->varargs;
    llvm::FunctionType* functype = llvm::FunctionType::get(actualRettype, paramvec, isvararg);

    f->llvmRetInPtr = retinptr;
    f->llvmUsesThis = usesthis;

    //if (!f->llvmType)
        f->llvmType = new llvm::PATypeHolder(functype);
    //else
        //assert(functype == f->llvmType->get());

    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static const llvm::FunctionType* DtoVaFunctionType(FuncDeclaration* fdecl)
{
    // type has already been resolved
    if (fdecl->type->llvmType != 0) {
        return llvm::cast<llvm::FunctionType>(fdecl->type->llvmType->get());
    }

    TypeFunction* f = (TypeFunction*)fdecl->type;
    assert(f != 0);

    const llvm::PointerType* i8pty = llvm::PointerType::get(llvm::Type::Int8Ty);
    std::vector<const llvm::Type*> args;

    if (fdecl->llvmInternal == LLVMva_start) {
        args.push_back(i8pty);
    }
    else if (fdecl->llvmInternal == LLVMva_intrinsic) {
        size_t n = Argument::dim(f->parameters);
        for (size_t i=0; i<n; ++i) {
            args.push_back(i8pty);
        }
    }
    else
    assert(0);

    const llvm::FunctionType* fty = llvm::FunctionType::get(llvm::Type::VoidTy, args, false);

    f->llvmType = new llvm::PATypeHolder(fty);

    return fty;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl)
{
    if ((fdecl->llvmInternal == LLVMva_start) || (fdecl->llvmInternal == LLVMva_intrinsic)) {
        return DtoVaFunctionType(fdecl);
    }

    // type has already been resolved
    if (fdecl->type->llvmType != 0) {
        return llvm::cast<llvm::FunctionType>(fdecl->type->llvmType->get());
    }

    const llvm::Type* thisty = NULL;
    if (fdecl->needThis()) {
        if (AggregateDeclaration* ad = fdecl->isMember()) {
            Logger::print("isMember = this is: %s\n", ad->type->toChars());
            thisty = DtoType(ad->type);
            //Logger::cout() << "this llvm type: " << *thisty << '\n';
            if (isaStruct(thisty) || (!gIR->structs.empty() && thisty == gIR->topstruct()->recty.get()))
                thisty = llvm::PointerType::get(thisty);
        }
        else
        assert(0);
    }
    else if (fdecl->isNested()) {
        thisty = llvm::PointerType::get(llvm::Type::Int8Ty);
    }

    const llvm::FunctionType* functype = DtoFunctionType(fdecl->type, thisty, fdecl->isMain());

    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* DtoDeclareVaFunction(FuncDeclaration* fdecl)
{
    TypeFunction* f = (TypeFunction*)DtoDType(fdecl->type);
    const llvm::FunctionType* fty = DtoVaFunctionType(fdecl);
    llvm::Constant* fn = 0;

    if (fdecl->llvmInternal == LLVMva_start) {
        fn = gIR->module->getOrInsertFunction("llvm.va_start", fty);
        assert(fn);
    }
    else if (fdecl->llvmInternal == LLVMva_intrinsic) {
        fn = gIR->module->getOrInsertFunction(fdecl->llvmInternal1, fty);
        assert(fn);
    }
    else
    assert(0);

    llvm::Function* func = llvm::dyn_cast<llvm::Function>(fn);
    assert(func);
    assert(func->isIntrinsic());
    fdecl->llvmValue = func;
    return func;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoResolveFunction(FuncDeclaration* fdecl)
{
    if (fdecl->llvmResolved) return;
    fdecl->llvmResolved = true;

    Logger::println("DtoResolveFunction(%s)", fdecl->toPrettyChars());
    LOG_SCOPE;

    if (fdecl->llvmRunTimeHack) {
        gIR->declareList.push_back(fdecl);
        return;
    }

    if (fdecl->isUnitTestDeclaration()) {
        Logger::attention("ignoring unittest declaration: %s", fdecl->toChars());
        return;
    }

    if (fdecl->parent)
    if (TemplateInstance* tinst = fdecl->parent->isTemplateInstance())
    {
        TemplateDeclaration* tempdecl = tinst->tempdecl;
        if (tempdecl->llvmInternal == LLVMva_arg)
        {
            Logger::println("magic va_arg found");
            fdecl->llvmInternal = LLVMva_arg;
            fdecl->llvmDeclared = true;
            fdecl->llvmInitialized = true;
            fdecl->llvmDefined = true;
            return; // this gets mapped to an instruction so a declaration makes no sence
        }
        else if (tempdecl->llvmInternal == LLVMva_start)
        {
            Logger::println("magic va_start found");
            fdecl->llvmInternal = LLVMva_start;
        }
    }

    DtoFunctionType(fdecl);

    // queue declaration
    if (!fdecl->isAbstract())
        gIR->declareList.push_back(fdecl);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareFunction(FuncDeclaration* fdecl)
{
    if (fdecl->llvmDeclared) return;
    fdecl->llvmDeclared = true;

    Logger::println("DtoDeclareFunction(%s)", fdecl->toPrettyChars());
    LOG_SCOPE;

    assert(!fdecl->isAbstract());

    if (fdecl->llvmRunTimeHack) {
        Logger::println("runtime hack func chars: %s", fdecl->toChars());
        if (!fdecl->llvmValue)
            fdecl->llvmValue = LLVM_D_GetRuntimeFunction(gIR->module, fdecl->toChars());
        return;
    }

    bool declareOnly = false;
    bool templInst = fdecl->parent && DtoIsTemplateInstance(fdecl->parent);
    if (!templInst && fdecl->getModule() != gIR->dmodule)
        declareOnly = true;
    else if (fdecl->llvmInternal == LLVMva_start)
        declareOnly = true;

    if (!fdecl->llvmIRFunc) {
        fdecl->llvmIRFunc = new IRFunction(fdecl);
    }

    // mangled name
    char* mangled_name;
    if (fdecl->llvmInternal == LLVMintrinsic)
        mangled_name = fdecl->llvmInternal1;
    else
        mangled_name = fdecl->mangle();

    // unit test special handling
    if (fdecl->isUnitTestDeclaration())
    {
        assert(0 && "no unittests yet");
        /*const llvm::FunctionType* fnty = llvm::FunctionType::get(llvm::Type::VoidTy, std::vector<const llvm::Type*>(), false);
        // make the function
        llvm::Function* func = gIR->module->getFunction(mangled_name);
        if (func == 0)
            func = new llvm::Function(fnty,llvm::GlobalValue::InternalLinkage,mangled_name,gIR->module);
        func->setCallingConv(llvm::CallingConv::Fast);
        fdecl->llvmValue = func;
        return func;
        */
    }

    if (fdecl->llvmInternal == LLVMintrinsic && fdecl->fbody) {
        error("intrinsics cannot have function bodies");
        fatal();
    }

    llvm::Function* vafunc = 0;
    if ((fdecl->llvmInternal == LLVMva_start) || (fdecl->llvmInternal == LLVMva_intrinsic)) {
        vafunc = DtoDeclareVaFunction(fdecl);
    }

    Type* t = DtoDType(fdecl->type);
    TypeFunction* f = (TypeFunction*)t;

    // construct function
    const llvm::FunctionType* functype = DtoFunctionType(fdecl);
    llvm::Function* func = vafunc ? vafunc : gIR->module->getFunction(mangled_name);
    if (!func)
        func = new llvm::Function(functype, DtoLinkage(fdecl->protection, fdecl->storage_class), mangled_name, gIR->module);
    else
        assert(func->getFunctionType() == functype);

    // add func to IRFunc
    fdecl->llvmIRFunc->func = func;

    // calling convention
    if (!vafunc && fdecl->llvmInternal != LLVMintrinsic)
        func->setCallingConv(DtoCallingConv(f->linkage));

    // template instances should have weak linkage
    if (!vafunc && fdecl->llvmInternal != LLVMintrinsic && fdecl->parent && DtoIsTemplateInstance(fdecl->parent))
        func->setLinkage(llvm::GlobalValue::WeakLinkage);

    fdecl->llvmValue = func;
    assert(llvm::isa<llvm::FunctionType>(f->llvmType->get()));

    // main
    if (fdecl->isMain()) {
        gIR->mainFunc = func;
    }

    // static ctor
    if (fdecl->isStaticCtorDeclaration()) {
        gIR->ctors.push_back(fdecl);
    }
    // static dtor
    else if (fdecl->isStaticDtorDeclaration()) {
        gIR->dtors.push_back(fdecl);
    }

    // name parameters
    llvm::Function::arg_iterator iarg = func->arg_begin();
    int k = 0;
    if (f->llvmRetInPtr) {
        iarg->setName("retval");
        f->llvmRetArg = iarg;
        ++iarg;
    }
    if (f->llvmUsesThis) {
        iarg->setName("this");
        fdecl->llvmThisVar = iarg;
        assert(fdecl->llvmThisVar);
        ++iarg;
    }
    int varargs = -1;
    if (f->linkage == LINKd && f->varargs == 1)
        varargs = 0;
    for (; iarg != func->arg_end(); ++iarg)
    {
        Argument* arg = Argument::getNth(f->parameters, k++);
        //arg->llvmValue = iarg;
        //Logger::println("identifier: '%s' %p\n", arg->ident->toChars(), arg->ident);
        if (arg && arg->ident != 0) {
            if (arg->vardecl) {
                arg->vardecl->llvmValue = iarg;
            }
            iarg->setName(arg->ident->toChars());
        }
        else if (!arg && varargs >= 0) {
            if (varargs == 0) {
                iarg->setName("_arguments");
                fdecl->llvmArguments = iarg;
            }
            else if (varargs == 1) {
                iarg->setName("_argptr");
                fdecl->llvmArgPtr = iarg;
            }
            else
            assert(0);
            varargs++;
        }
        else {
            iarg->setName("unnamed");
        }
    }

    if (!declareOnly)
        gIR->defineList.push_back(fdecl);

    Logger::cout() << "func decl: " << *func << '\n';
}

//////////////////////////////////////////////////////////////////////////////////////////

// TODO split this monster up
void DtoDefineFunc(FuncDeclaration* fd)
{
    if (fd->llvmDefined) return;
    fd->llvmDefined = true;

    assert(fd->llvmDeclared);

    Logger::println("DtoDefineFunc(%s)", fd->toPrettyChars());
    LOG_SCOPE;

    // debug info
    if (global.params.symdebug) {
        Module* mo = fd->getModule();
        if (!mo->llvmCompileUnit) {
            mo->llvmCompileUnit = DtoDwarfCompileUnit(mo,false);
        }
        fd->llvmDwarfSubProgram = DtoDwarfSubProgram(fd, mo->llvmCompileUnit);
    }

    Type* t = DtoDType(fd->type);
    TypeFunction* f = (TypeFunction*)t;

    assert(f->llvmType);
    llvm::Function* func = fd->llvmIRFunc->func;
    const llvm::FunctionType* functype = func->getFunctionType();

    // only members of the current module or template instances maybe be defined
    if (fd->getModule() == gIR->dmodule || DtoIsTemplateInstance(fd->parent))
    {
        fd->llvmDModule = gIR->dmodule;

        // function definition
        if (fd->fbody != 0)
        {
            Logger::println("Doing function body for: %s", fd->toChars());
            assert(fd->llvmIRFunc);
            gIR->functions.push_back(fd->llvmIRFunc);

            if (fd->isMain())
                gIR->emitMain = true;

            llvm::BasicBlock* beginbb = new llvm::BasicBlock("entry",func);
            llvm::BasicBlock* endbb = new llvm::BasicBlock("endentry",func);

            //assert(gIR->scopes.empty());
            gIR->scopes.push_back(IRScope(beginbb, endbb));

                // create alloca point
                f->llvmAllocaPoint = new llvm::BitCastInst(llvm::ConstantInt::get(llvm::Type::Int32Ty,0,false),llvm::Type::Int32Ty,"alloca point",gIR->scopebb());
                gIR->func()->allocapoint = f->llvmAllocaPoint;

                // need result variable? (not nested)
                if (fd->vresult && !fd->vresult->nestedref) {
                    Logger::println("non-nested vresult value");
                    fd->vresult->llvmValue = new llvm::AllocaInst(DtoType(fd->vresult->type),"function_vresult",f->llvmAllocaPoint);
                }

                // give arguments storage
                size_t n = Argument::dim(f->parameters);
                for (int i=0; i < n; ++i) {
                    Argument* arg = Argument::getNth(f->parameters, i);
                    if (arg && arg->vardecl) {
                        VarDeclaration* vd = arg->vardecl;
                        if (!vd->llvmNeedsStorage || vd->nestedref || vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type))
                            continue;
                        llvm::Value* a = vd->llvmValue;
                        assert(a);
                        std::string s(a->getName());
                        Logger::println("giving argument '%s' storage", s.c_str());
                        s.append("_storage");
                        llvm::Value* v = new llvm::AllocaInst(a->getType(),s,f->llvmAllocaPoint);
                        gIR->ir->CreateStore(a,v);
                        vd->llvmValue = v;
                    }
                    else {
                        Logger::attention("some unknown argument: %s", arg ? arg->toChars() : 0);
                    }
                }

                // debug info
                if (global.params.symdebug) DtoDwarfFuncStart(fd);

                llvm::Value* parentNested = NULL;
                if (FuncDeclaration* fd2 = fd->toParent()->isFuncDeclaration()) {
                    if (!fd->isStatic())
                        parentNested = fd2->llvmNested;
                }

                // need result variable? (nested)
                if (fd->vresult && fd->vresult->nestedref) {
                    Logger::println("nested vresult value: %s", fd->vresult->toChars());
                    fd->llvmNestedVars.insert(fd->vresult);
                }

                // construct nested variables struct
                if (!fd->llvmNestedVars.empty() || parentNested) {
                    std::vector<const llvm::Type*> nestTypes;
                    int j = 0;
                    if (parentNested) {
                        nestTypes.push_back(parentNested->getType());
                        j++;
                    }
                    for (std::set<VarDeclaration*>::iterator i=fd->llvmNestedVars.begin(); i!=fd->llvmNestedVars.end(); ++i) {
                        VarDeclaration* vd = *i;
                        vd->llvmNestedIndex = j++;
                        if (vd->isParameter()) {
                            assert(vd->llvmValue);
                            nestTypes.push_back(vd->llvmValue->getType());
                        }
                        else {
                            nestTypes.push_back(DtoType(vd->type));
                        }
                    }
                    const llvm::StructType* nestSType = llvm::StructType::get(nestTypes);
                    Logger::cout() << "nested var struct has type:" << '\n' << *nestSType;
                    fd->llvmNested = new llvm::AllocaInst(nestSType,"nestedvars",f->llvmAllocaPoint);
                    if (parentNested) {
                        assert(fd->llvmThisVar);
                        llvm::Value* ptr = gIR->ir->CreateBitCast(fd->llvmThisVar, parentNested->getType(), "tmp");
                        gIR->ir->CreateStore(ptr, DtoGEPi(fd->llvmNested, 0,0, "tmp"));
                    }
                    for (std::set<VarDeclaration*>::iterator i=fd->llvmNestedVars.begin(); i!=fd->llvmNestedVars.end(); ++i) {
                        VarDeclaration* vd = *i;
                        if (vd->isParameter()) {
                            gIR->ir->CreateStore(vd->llvmValue, DtoGEPi(fd->llvmNested, 0, vd->llvmNestedIndex, "tmp"));
                            vd->llvmValue = fd->llvmNested;
                        }
                    }
                }

                // copy _argptr to a memory location
                if (f->linkage == LINKd && f->varargs == 1)
                {
                    llvm::Value* argptrmem = new llvm::AllocaInst(fd->llvmArgPtr->getType(), "_argptrmem", gIR->topallocapoint());
                    new llvm::StoreInst(fd->llvmArgPtr, argptrmem, gIR->scopebb());
                    fd->llvmArgPtr = argptrmem;
                }

                // output function body
                fd->fbody->toIR(gIR);

                // llvm requires all basic blocks to end with a TerminatorInst but DMD does not put a return statement
                // in automatically, so we do it here.
                if (!fd->isMain()) {
                    if (!gIR->scopereturned()) {
                        // pass the previous block into this block
                        if (global.params.symdebug) DtoDwarfFuncEnd(fd);
                        if (func->getReturnType() == llvm::Type::VoidTy) {
                            new llvm::ReturnInst(gIR->scopebb());
                        }
                        else {
                            new llvm::ReturnInst(llvm::UndefValue::get(func->getReturnType()), gIR->scopebb());
                        }
                    }
                }

                // erase alloca point
                f->llvmAllocaPoint->eraseFromParent();
                f->llvmAllocaPoint = 0;
                gIR->func()->allocapoint = 0;

            gIR->scopes.pop_back();

            // get rid of the endentry block, it's never used
            assert(!func->getBasicBlockList().empty());
            func->getBasicBlockList().pop_back();

            // if the last block is empty now, it must be unreachable or it's a bug somewhere else
            // would be nice to figure out how to assert that this is correct
            llvm::BasicBlock* lastbb = &func->getBasicBlockList().back();
            if (lastbb->empty()) {
                if (lastbb->getNumUses() == 0)
                    lastbb->eraseFromParent();
                else {
                    new llvm::UnreachableInst(lastbb);
                    /*if (func->getReturnType() == llvm::Type::VoidTy) {
                        new llvm::ReturnInst(lastbb);
                    }
                    else {
                        new llvm::ReturnInst(llvm::UndefValue::get(func->getReturnType()), lastbb);
                    }*/
                }
            }

            gIR->functions.pop_back();
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoMain()
{
    // emit main function llvm style
    // int main(int argc, char**argv, char**env);

    assert(gIR != 0);
    IRState& ir = *gIR;

    assert(ir.emitMain && ir.mainFunc);

    // parameter types
    std::vector<const llvm::Type*> pvec;
    pvec.push_back((const llvm::Type*)llvm::Type::Int32Ty);
    const llvm::Type* chPtrType = (const llvm::Type*)llvm::PointerType::get(llvm::Type::Int8Ty);
    pvec.push_back((const llvm::Type*)llvm::PointerType::get(chPtrType));
    pvec.push_back((const llvm::Type*)llvm::PointerType::get(chPtrType));
    const llvm::Type* rettype = (const llvm::Type*)llvm::Type::Int32Ty;

    llvm::FunctionType* functype = llvm::FunctionType::get(rettype, pvec, false);
    llvm::Function* func = new llvm::Function(functype,llvm::GlobalValue::ExternalLinkage,"main",ir.module);

    llvm::BasicBlock* bb = new llvm::BasicBlock("entry",func);

    // call static ctors
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(ir.module,"_moduleCtor");
    llvm::Instruction* apt = new llvm::CallInst(fn,"",bb);

    // call user main function
    const llvm::FunctionType* mainty = ir.mainFunc->getFunctionType();
    llvm::CallInst* call;
    if (mainty->getNumParams() > 0)
    {
        // main with arguments
        assert(mainty->getNumParams() == 1);
        std::vector<llvm::Value*> args;
        llvm::Function* mfn = LLVM_D_GetRuntimeFunction(ir.module,"_d_main_args");

        llvm::Function::arg_iterator argi = func->arg_begin();
        args.push_back(argi++);
        args.push_back(argi++);

        const llvm::Type* at = mainty->getParamType(0)->getContainedType(0);
        llvm::Value* arr = new llvm::AllocaInst(at->getContainedType(1)->getContainedType(0), func->arg_begin(), "argstorage", apt);
        llvm::Value* a = new llvm::AllocaInst(at, "argarray", apt);
        llvm::Value* ptr = DtoGEPi(a,0,0,"tmp",bb);
        llvm::Value* v = args[0];
        if (v->getType() != DtoSize_t())
            v = new llvm::ZExtInst(v, DtoSize_t(), "tmp", bb);
        new llvm::StoreInst(v,ptr,bb);
        ptr = DtoGEPi(a,0,1,"tmp",bb);
        new llvm::StoreInst(arr,ptr,bb);
        args.push_back(a);
        new llvm::CallInst(mfn, args.begin(), args.end(), "", bb);
        call = new llvm::CallInst(ir.mainFunc,a,"ret",bb);
    }
    else
    {
        // main with no arguments
        call = new llvm::CallInst(ir.mainFunc,"ret",bb);
    }
    call->setCallingConv(ir.mainFunc->getCallingConv());

    // call static dtors
    fn = LLVM_D_GetRuntimeFunction(ir.module,"_moduleDtor");
    new llvm::CallInst(fn,"",bb);

    // return
    new llvm::ReturnInst(call,bb);
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::FunctionType* DtoBaseFunctionType(FuncDeclaration* fdecl)
{
    Dsymbol* parent = fdecl->toParent();
    ClassDeclaration* cd = parent->isClassDeclaration();
    assert(cd);

    FuncDeclaration* f = fdecl;

    while (cd)
    {
        ClassDeclaration* base = cd->baseClass;
        if (!base)
            break;
        FuncDeclaration* f2 = base->findFunc(fdecl->ident, (TypeFunction*)fdecl->type);
        if (f2) {
            f = f2;
            cd = base;
        }
        else
            break;
    }

    DtoResolveDsymbol(f);
    return llvm::cast<llvm::FunctionType>(DtoType(f->type));
}

//////////////////////////////////////////////////////////////////////////////////////////
