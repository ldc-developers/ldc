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
#include "gen/dvalue.h"

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
            rettype = getPtrToType(DtoType(rt));
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
        //Logger::cout() << "returning through pointer parameter: " << *rettype << '\n';
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
        assert(ti->irStruct->constInit);
        std::vector<const llvm::Type*> types;
        types.push_back(DtoSize_t());
        types.push_back(getPtrToType(getPtrToType(ti->irStruct->constInit->getType())));
        const llvm::Type* t1 = llvm::StructType::get(types);
        paramvec.push_back(getPtrToType(t1));
        paramvec.push_back(getPtrToType(llvm::Type::Int8Ty));
    }

    size_t n = Argument::dim(f->parameters);

    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = DtoDType(arg->type);
        assert(argT);

        const llvm::Type* at = DtoType(argT);
        if (isaStruct(at)) {
            Logger::println("struct param");
            paramvec.push_back(getPtrToType(at));
        }
        else if (isaArray(at)) {
            Logger::println("sarray param");
            assert(argT->ty == Tsarray);
            //paramvec.push_back(getPtrToType(at->getContainedType(0)));
            paramvec.push_back(getPtrToType(at));
        }
        else if (llvm::isa<llvm::OpaqueType>(at)) {
            Logger::println("opaque param");
            assert(argT->ty == Tstruct || argT->ty == Tclass);
            paramvec.push_back(getPtrToType(at));
        }
        else {
            if ((arg->storageClass & STCref) || (arg->storageClass & STCout)) {
                Logger::println("by ref param");
                at = getPtrToType(at);
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

    const llvm::PointerType* i8pty = getPtrToType(llvm::Type::Int8Ty);
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

    // unittest has null type, just build it manually
    /*if (fdecl->isUnitTestDeclaration()) {
        std::vector<const llvm::Type*> args;
        return llvm::FunctionType::get(llvm::Type::VoidTy, args, false);
    }*/

    // type has already been resolved
    if (fdecl->type->llvmType != 0) {
        return llvm::cast<llvm::FunctionType>(fdecl->type->llvmType->get());
    }

    const llvm::Type* thisty = NULL;
    if (fdecl->needThis()) {
        if (AggregateDeclaration* ad = fdecl->isMember2()) {
            Logger::println("isMember = this is: %s", ad->type->toChars());
            thisty = DtoType(ad->type);
            //Logger::cout() << "this llvm type: " << *thisty << '\n';
            if (isaStruct(thisty) || (!gIR->structs.empty() && thisty == gIR->topstruct()->recty.get()))
                thisty = getPtrToType(thisty);
        }
        else {
            Logger::println("chars: %s type: %s kind: %s", fdecl->toChars(), fdecl->type->toChars(), fdecl->kind());
            assert(0);
        }
    }
    else if (fdecl->isNested()) {
        thisty = getPtrToType(llvm::Type::Int8Ty);
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
    fdecl->irFunc->func = func;
    return func;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoResolveFunction(FuncDeclaration* fdecl)
{
    if (!global.params.useUnitTests && fdecl->isUnitTestDeclaration()) {
        return; // ignore declaration completely
    }

    // is imported and we don't have access?
    if (fdecl->getModule() != gIR->dmodule)
    {
        if (fdecl->prot() == PROTprivate)
            return;
    }

    if (fdecl->llvmResolved) return;
    fdecl->llvmResolved = true;

    Logger::println("DtoResolveFunction(%s): %s", fdecl->toPrettyChars(), fdecl->loc.toChars());
    LOG_SCOPE;

    if (fdecl->runTimeHack) {
        gIR->declareList.push_back(fdecl);
        TypeFunction* tf = (TypeFunction*)fdecl->type;
        tf->llvmRetInPtr = DtoIsPassedByRef(tf->next);
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

    Logger::println("DtoDeclareFunction(%s): %s", fdecl->toPrettyChars(), fdecl->loc.toChars());
    LOG_SCOPE;

    assert(!fdecl->isAbstract());

    // intrinsic sanity check
    if (fdecl->llvmInternal == LLVMintrinsic && fdecl->fbody) {
        error(fdecl->loc, "intrinsics cannot have function bodies");
        fatal();
    }

    if (fdecl->runTimeHack) {
        Logger::println("runtime hack func chars: %s", fdecl->toChars());
        if (!fdecl->irFunc) {
            fdecl->irFunc = new IrFunction(fdecl);
            fdecl->irFunc->func = LLVM_D_GetRuntimeFunction(gIR->module, fdecl->toChars());
        }
        return;
    }

    bool declareOnly = false;
    bool templInst = fdecl->parent && DtoIsTemplateInstance(fdecl->parent);
    if (!templInst && fdecl->getModule() != gIR->dmodule)
    {
        Logger::println("not template instance, and not in this module. declare only!");
        Logger::println("current module: %s", gIR->dmodule->ident->toChars());
        Logger::println("func module: %s", fdecl->getModule()->ident->toChars());
        declareOnly = true;
    }
    else if (fdecl->llvmInternal == LLVMva_start)
        declareOnly = true;

    if (!fdecl->irFunc) {
        fdecl->irFunc = new IrFunction(fdecl);
    }

    // mangled name
    char* mangled_name;
    if (fdecl->llvmInternal == LLVMintrinsic)
        mangled_name = fdecl->llvmInternal1;
    else
        mangled_name = fdecl->mangle();

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
        func = new llvm::Function(functype, DtoLinkage(fdecl), mangled_name, gIR->module);
    else
        assert(func->getFunctionType() == functype);

    // add func to IRFunc
    fdecl->irFunc->func = func;

    // calling convention
    if (!vafunc && fdecl->llvmInternal != LLVMintrinsic)
        func->setCallingConv(DtoCallingConv(f->linkage));
    else // fall back to C, it should be the right thing to do
        func->setCallingConv(llvm::CallingConv::C);

    fdecl->irFunc->func = func;
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
        fdecl->irFunc->retArg = iarg;
        ++iarg;
    }
    if (f->llvmUsesThis) {
        iarg->setName("this");
        fdecl->irFunc->thisVar = iarg;
        assert(fdecl->irFunc->thisVar);
        ++iarg;
    }

    if (f->linkage == LINKd && f->varargs == 1) {
        iarg->setName("_arguments");
        fdecl->irFunc->_arguments = iarg;
        ++iarg;
        iarg->setName("_argptr");
        fdecl->irFunc->_argptr = iarg;
        ++iarg;
    }

    for (; iarg != func->arg_end(); ++iarg)
    {
        Argument* arg = Argument::getNth(f->parameters, k++);
        //arg->llvmValue = iarg;
        //Logger::println("identifier: '%s' %p\n", arg->ident->toChars(), arg->ident);
        if (arg && arg->ident != 0) {
            if (arg->vardecl) {
                assert(!arg->vardecl->irLocal);
                arg->vardecl->irLocal = new IrLocal(arg->vardecl);
                arg->vardecl->irLocal->value = iarg;
            }
            iarg->setName(arg->ident->toChars());
        }
        else {
            iarg->setName("unnamed");
        }
    }

    if (fdecl->isUnitTestDeclaration())
        gIR->unitTests.push_back(fdecl);

    if (!declareOnly)
        gIR->defineList.push_back(fdecl);
    else
        assert(func->getLinkage() != llvm::GlobalValue::InternalLinkage);

    Logger::cout() << "func decl: " << *func << '\n';
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineFunc(FuncDeclaration* fd)
{
    if (fd->llvmDefined) return;
    fd->llvmDefined = true;

    assert(fd->llvmDeclared);

    Logger::println("DtoDefineFunc(%s): %s", fd->toPrettyChars(), fd->loc.toChars());
    LOG_SCOPE;

    // debug info
    if (global.params.symdebug) {
        Module* mo = fd->getModule();
        fd->irFunc->dwarfSubProg = DtoDwarfSubProgram(fd, DtoDwarfCompileUnit(mo));
    }

    Type* t = DtoDType(fd->type);
    TypeFunction* f = (TypeFunction*)t;
    assert(f->llvmType);

    llvm::Function* func = fd->irFunc->func;
    const llvm::FunctionType* functype = func->getFunctionType();

    // only members of the current module or template instances maybe be defined
    if (fd->getModule() == gIR->dmodule || DtoIsTemplateInstance(fd->parent))
    {
        fd->llvmDModule = gIR->dmodule;

        // function definition
        if (fd->fbody != 0)
        {
            Logger::println("Doing function body for: %s", fd->toChars());
            assert(fd->irFunc);
            gIR->functions.push_back(fd->irFunc);

            if (fd->isMain())
                gIR->emitMain = true;

            llvm::BasicBlock* beginbb = new llvm::BasicBlock("entry",func);
            llvm::BasicBlock* endbb = new llvm::BasicBlock("endentry",func);

            //assert(gIR->scopes.empty());
            gIR->scopes.push_back(IRScope(beginbb, endbb));

                // create alloca point
                llvm::Instruction* allocaPoint = new llvm::BitCastInst(llvm::ConstantInt::get(llvm::Type::Int32Ty,0,false),llvm::Type::Int32Ty,"alloca point",gIR->scopebb());
                gIR->func()->allocapoint = allocaPoint;

                // need result variable? (not nested)
                if (fd->vresult && !fd->vresult->nestedref) {
                    Logger::println("non-nested vresult value");
                    fd->vresult->irLocal = new IrLocal(fd->vresult);
                    fd->vresult->irLocal->value = new llvm::AllocaInst(DtoType(fd->vresult->type),"function_vresult",allocaPoint);
                }

                // give arguments storage
                size_t n = Argument::dim(f->parameters);
                for (int i=0; i < n; ++i) {
                    Argument* arg = Argument::getNth(f->parameters, i);
                    if (arg && arg->vardecl) {
                        VarDeclaration* vd = arg->vardecl;
                        if (!vd->needsStorage || vd->nestedref || vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type))
                            continue;
                        llvm::Value* a = vd->irLocal->value;
                        assert(a);
                        std::string s(a->getName());
                        Logger::println("giving argument '%s' storage", s.c_str());
                        s.append("_storage");
                        llvm::Value* v = new llvm::AllocaInst(a->getType(),s,allocaPoint);
                        gIR->ir->CreateStore(a,v);
                        vd->irLocal->value = v;
                    }
                    else {
                        Logger::attention(fd->loc, "some unknown argument: %s", arg ? arg->toChars() : 0);
                    }
                }

                // debug info
                if (global.params.symdebug) DtoDwarfFuncStart(fd);

                llvm::Value* parentNested = NULL;
                if (FuncDeclaration* fd2 = fd->toParent2()->isFuncDeclaration()) {
                    if (!fd->isStatic()) // huh?
                        parentNested = fd2->irFunc->nestedVar;
                }

                // need result variable? (nested)
                if (fd->vresult && fd->vresult->nestedref) {
                    Logger::println("nested vresult value: %s", fd->vresult->toChars());
                    fd->nestedVars.insert(fd->vresult);
                }

                // construct nested variables struct
                if (!fd->nestedVars.empty() || parentNested) {
                    std::vector<const llvm::Type*> nestTypes;
                    int j = 0;
                    if (parentNested) {
                        nestTypes.push_back(parentNested->getType());
                        j++;
                    }
                    for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i) {
                        VarDeclaration* vd = *i;
                        Logger::println("referenced nested variable %s", vd->toChars());
                        if (!vd->irLocal)
                            vd->irLocal = new IrLocal(vd);
                        vd->irLocal->nestedIndex = j++;
                        if (vd->isParameter()) {
                            if (!vd->irLocal->value) {
                                assert(vd == fd->vthis);
                                vd->irLocal->value = fd->irFunc->thisVar;
                            }
                            assert(vd->irLocal->value);
                            nestTypes.push_back(vd->irLocal->value->getType());
                        }
                        else {
                            nestTypes.push_back(DtoType(vd->type));
                        }
                    }
                    const llvm::StructType* nestSType = llvm::StructType::get(nestTypes);
                    Logger::cout() << "nested var struct has type:" << *nestSType << '\n';
                    fd->irFunc->nestedVar = new llvm::AllocaInst(nestSType,"nestedvars",allocaPoint);
                    if (parentNested) {
                        assert(fd->irFunc->thisVar);
                        llvm::Value* ptr = gIR->ir->CreateBitCast(fd->irFunc->thisVar, parentNested->getType(), "tmp");
                        gIR->ir->CreateStore(ptr, DtoGEPi(fd->irFunc->nestedVar, 0,0, "tmp"));
                    }
                    for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i) {
                        VarDeclaration* vd = *i;
                        if (vd->isParameter()) {
                            assert(vd->irLocal);
                            gIR->ir->CreateStore(vd->irLocal->value, DtoGEPi(fd->irFunc->nestedVar, 0, vd->irLocal->nestedIndex, "tmp"));
                            vd->irLocal->value = fd->irFunc->nestedVar;
                        }
                    }
                }

                // copy _argptr to a memory location
                if (f->linkage == LINKd && f->varargs == 1)
                {
                    llvm::Value* argptrmem = new llvm::AllocaInst(fd->irFunc->_argptr->getType(), "_argptrmem", gIR->topallocapoint());
                    new llvm::StoreInst(fd->irFunc->_argptr, argptrmem, gIR->scopebb());
                    fd->irFunc->_argptr = argptrmem;
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
                allocaPoint->eraseFromParent();
                allocaPoint = 0;
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
    const llvm::Type* chPtrType = (const llvm::Type*)getPtrToType(llvm::Type::Int8Ty);
    pvec.push_back((const llvm::Type*)getPtrToType(chPtrType));
    pvec.push_back((const llvm::Type*)getPtrToType(chPtrType));
    const llvm::Type* rettype = (const llvm::Type*)llvm::Type::Int32Ty;

    llvm::FunctionType* functype = llvm::FunctionType::get(rettype, pvec, false);
    llvm::Function* func = new llvm::Function(functype,llvm::GlobalValue::ExternalLinkage,"main",ir.module);

    llvm::BasicBlock* bb = new llvm::BasicBlock("entry",func);

    // call static ctors
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(ir.module,"_moduleCtor");
    llvm::Instruction* apt = new llvm::CallInst(fn,"",bb);

    // run unit tests if -unittest is provided
    if (global.params.useUnitTests) {
        fn = LLVM_D_GetRuntimeFunction(ir.module,"_moduleUnitTests");
        llvm::Instruction* apt = new llvm::CallInst(fn,"",bb);
    }

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

DValue* DtoArgument(Argument* fnarg, Expression* argexp)
{
    Logger::println("DtoArgument");
    LOG_SCOPE;

    DValue* arg = argexp->toElem(gIR);

    // ref/out arg
    if (fnarg && ((fnarg->storageClass & STCref) || (fnarg->storageClass & STCout)))
    {
        if (arg->isVar() || arg->isLRValue())
            arg = new DImValue(argexp->type, arg->getLVal(), false);
        else
            arg = new DImValue(argexp->type, arg->getRVal(), false);
    }
    // aggregate arg
    else if (DtoIsPassedByRef(argexp->type))
    {
        llvm::Value* alloc = new llvm::AllocaInst(DtoType(argexp->type), "tmpparam", gIR->topallocapoint());
        DVarValue* vv = new DVarValue(argexp->type, alloc, true);
        DtoAssign(vv, arg);
        arg = vv;
    }
    // normal arg (basic/value type)
    else
    {
        // nothing to do
    }

    return arg;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoVariadicArgument(Expression* argexp, llvm::Value* dst)
{
    Logger::println("DtoVariadicArgument");
    LOG_SCOPE;
    DVarValue* vv = new DVarValue(argexp->type, dst, true);
    DtoAssign(vv, argexp->toElem(gIR));
}

//////////////////////////////////////////////////////////////////////////////////////////
