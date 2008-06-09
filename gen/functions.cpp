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
#include "gen/llvmhelpers.h"
#include "gen/runtime.h"
#include "gen/arrays.h"
#include "gen/logger.h"
#include "gen/functions.h"
#include "gen/todebug.h"
#include "gen/classes.h"
#include "gen/dvalue.h"

const llvm::FunctionType* DtoFunctionType(Type* type, const LLType* thistype, bool ismain)
{
    TypeFunction* f = (TypeFunction*)type;
    assert(f != 0);

    if (type->ir.type != NULL) {
        return llvm::cast<llvm::FunctionType>(type->ir.type->get());
    }

    bool typesafeVararg = false;
    bool arrayVararg = false;
    if (f->linkage == LINKd)
    {
        if (f->varargs == 1)
            typesafeVararg = true;
        else if (f->varargs == 2)
            arrayVararg = true;
    }

    // return value type
    const LLType* rettype;
    const LLType* actualRettype;
    Type* rt = f->next;
    bool retinptr = false;
    bool usesthis = false;

    // parameter types
    std::vector<const LLType*> paramvec;

    if (ismain)
    {
        rettype = LLType::Int32Ty;
        actualRettype = rettype;
        if (Argument::dim(f->parameters) == 0)
        {
        const LLType* arrTy = DtoArrayType(LLType::Int8Ty);
        const LLType* arrArrTy = DtoArrayType(arrTy);
        paramvec.push_back(getPtrToType(arrArrTy));
        }
    }
    else{
        assert(rt);
        Type* rtfin = DtoDType(rt);
        if (DtoIsReturnedInArg(rt)) {
            rettype = getPtrToType(DtoType(rt));
            actualRettype = LLType::VoidTy;
            f->llvmRetInPtr = retinptr = true;
        }
        else {
            rettype = DtoType(rt);
            actualRettype = rettype;
        }
    }

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
        assert(ti->ir.irStruct->constInit);
        std::vector<const LLType*> types;
        types.push_back(DtoSize_t());
        types.push_back(getPtrToType(getPtrToType(ti->ir.irStruct->constInit->getType())));
        const LLType* t1 = llvm::StructType::get(types);
        paramvec.push_back(getPtrToType(t1));
        paramvec.push_back(getPtrToType(LLType::Int8Ty));
    }
    else if (arrayVararg)
    {
        // do nothing?
    }

    size_t n = Argument::dim(f->parameters);

    int nbyval = 0;

    llvm::PAListPtr palist;

    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = DtoDType(arg->type);
        assert(argT);

        bool refOrOut = ((arg->storageClass & STCref) || (arg->storageClass & STCout));

        const LLType* at = DtoType(argT);
        if (isaStruct(at)) {
            Logger::println("struct param");
            paramvec.push_back(getPtrToType(at));
            arg->llvmByVal = !refOrOut;
        }
        else if (isaArray(at)) {
            Logger::println("sarray param");
            assert(argT->ty == Tsarray);
            //paramvec.push_back(getPtrToType(at->getContainedType(0)));
            paramvec.push_back(getPtrToType(at));
            //arg->llvmByVal = !refOrOut; // static array are passed by reference
        }
        else if (llvm::isa<llvm::OpaqueType>(at)) {
            Logger::println("opaque param");
            assert(argT->ty == Tstruct || argT->ty == Tclass);
            paramvec.push_back(getPtrToType(at));
        }
        else {
            if (refOrOut) {
                Logger::println("by ref param");
                at = getPtrToType(at);
            }
            else {
                Logger::println("in param");
            }
            paramvec.push_back(at);
        }

        if (arg->llvmByVal)
            nbyval++;
    }

    //warning("set %d byval args for type: %s", nbyval, f->toChars());

    // construct function type
    bool isvararg = !(typesafeVararg || arrayVararg) && f->varargs;
    llvm::FunctionType* functype = llvm::FunctionType::get(actualRettype, paramvec, isvararg);

    f->llvmRetInPtr = retinptr;
    f->llvmUsesThis = usesthis;

    f->ir.type = new llvm::PATypeHolder(functype);

    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static const llvm::FunctionType* DtoVaFunctionType(FuncDeclaration* fdecl)
{
    // type has already been resolved
    if (fdecl->type->ir.type != 0) {
        return llvm::cast<llvm::FunctionType>(fdecl->type->ir.type->get());
    }

    TypeFunction* f = (TypeFunction*)fdecl->type;
    assert(f != 0);

    const llvm::PointerType* i8pty = getPtrToType(LLType::Int8Ty);
    std::vector<const LLType*> args;

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

    const llvm::FunctionType* fty = llvm::FunctionType::get(LLType::VoidTy, args, false);

    f->ir.type = new llvm::PATypeHolder(fty);

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
        std::vector<const LLType*> args;
        return llvm::FunctionType::get(LLType::VoidTy, args, false);
    }*/

    // type has already been resolved
    if (fdecl->type->ir.type != 0) {
        return llvm::cast<llvm::FunctionType>(fdecl->type->ir.type->get());
    }

    const LLType* thisty = NULL;
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
        thisty = getPtrToType(LLType::Int8Ty);
    }

    const llvm::FunctionType* functype = DtoFunctionType(fdecl->type, thisty, fdecl->isMain());

    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* DtoDeclareVaFunction(FuncDeclaration* fdecl)
{
    TypeFunction* f = (TypeFunction*)DtoDType(fdecl->type);
    const llvm::FunctionType* fty = DtoVaFunctionType(fdecl);
    LLConstant* fn = 0;

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
    fdecl->ir.irFunc->func = func;
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

    if (fdecl->ir.resolved) return;
    fdecl->ir.resolved = true;

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
            fdecl->ir.declared = true;
            fdecl->ir.initialized = true;
            fdecl->ir.defined = true;
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

static void set_param_attrs(TypeFunction* f, llvm::Function* func, FuncDeclaration* fdecl)
{
    assert(f->parameters);

    int llidx = 1;
    if (f->llvmRetInPtr) ++llidx;
    if (f->llvmUsesThis) ++llidx;
    if (f->linkage == LINKd && f->varargs == 1)
        llidx += 2;

    int funcNumArgs = func->getArgumentList().size();
    std::vector<llvm::ParamAttrsWithIndex> attrs;
    int k = 0;

    int nbyval = 0;

    if (fdecl->isMain() && Argument::dim(f->parameters) == 0)
    {
        llvm::ParamAttrsWithIndex PAWI;
        PAWI.Index = llidx;
        PAWI.Attrs = llvm::ParamAttr::ByVal;
        attrs.push_back(PAWI);
        llidx++;
        nbyval++;
    }

    for (; llidx <= funcNumArgs && f->parameters->dim > k; ++llidx,++k)
    {
        Argument* fnarg = (Argument*)f->parameters->data[k];
        assert(fnarg);
        if (fnarg->llvmByVal)
        {
            llvm::ParamAttrsWithIndex PAWI;
            PAWI.Index = llidx;
            PAWI.Attrs = llvm::ParamAttr::ByVal;
            attrs.push_back(PAWI);
            nbyval++;
        }
    }

    if (nbyval) {
        llvm::PAListPtr palist = llvm::PAListPtr::get(attrs.begin(), attrs.end());
        func->setParamAttrs(palist);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareFunction(FuncDeclaration* fdecl)
{
    if (fdecl->ir.declared) return;
    fdecl->ir.declared = true;

    Logger::println("DtoDeclareFunction(%s): %s", fdecl->toPrettyChars(), fdecl->loc.toChars());
    LOG_SCOPE;

    assert(!fdecl->isAbstract());

    // intrinsic sanity check
    if (fdecl->llvmInternal == LLVMintrinsic && fdecl->fbody) {
        error(fdecl->loc, "intrinsics cannot have function bodies");
        fatal();
    }

    // get TypeFunction*
    Type* t = DtoDType(fdecl->type);
    TypeFunction* f = (TypeFunction*)t;

    // runtime function special handling
    if (fdecl->runTimeHack) {
        Logger::println("runtime hack func chars: %s", fdecl->toChars());
        if (!fdecl->ir.irFunc) {
            IrFunction* irfunc = new IrFunction(fdecl);
            llvm::Function* llfunc = LLVM_D_GetRuntimeFunction(gIR->module, fdecl->toChars());
            fdecl->ir.irFunc = irfunc;
            fdecl->ir.irFunc->func = llfunc;
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

    if (!fdecl->ir.irFunc) {
        fdecl->ir.irFunc = new IrFunction(fdecl);
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

    // construct function
    const llvm::FunctionType* functype = DtoFunctionType(fdecl);
    llvm::Function* func = vafunc ? vafunc : gIR->module->getFunction(mangled_name);
    if (!func)
        func = llvm::Function::Create(functype, DtoLinkage(fdecl), mangled_name, gIR->module);
    else
        assert(func->getFunctionType() == functype);

    // add func to IRFunc
    fdecl->ir.irFunc->func = func;

    // calling convention
    if (!vafunc && fdecl->llvmInternal != LLVMintrinsic)
        func->setCallingConv(DtoCallingConv(f->linkage));
    else // fall back to C, it should be the right thing to do
        func->setCallingConv(llvm::CallingConv::C);

    fdecl->ir.irFunc->func = func;
    assert(llvm::isa<llvm::FunctionType>(f->ir.type->get()));

    // parameter attributes
    if (f->parameters) {
        set_param_attrs(f, func, fdecl);
    }

    // main
    if (fdecl->isMain()) {
        gIR->mainFunc = func;
    }

    // static ctor
    if (fdecl->isStaticCtorDeclaration() && fdecl->getModule() == gIR->dmodule) {
        gIR->ctors.push_back(fdecl);
    }
    // static dtor
    else if (fdecl->isStaticDtorDeclaration() && fdecl->getModule() == gIR->dmodule) {
        gIR->dtors.push_back(fdecl);
    }

    // we never reference parameters of function prototypes
    if (!declareOnly)
    {
        // name parameters
        llvm::Function::arg_iterator iarg = func->arg_begin();
        int k = 0;
        if (f->llvmRetInPtr) {
            iarg->setName("retval");
            fdecl->ir.irFunc->retArg = iarg;
            ++iarg;
        }
        if (f->llvmUsesThis) {
            iarg->setName("this");
            fdecl->ir.irFunc->thisVar = iarg;
            assert(fdecl->ir.irFunc->thisVar);
            ++iarg;
        }

        if (f->linkage == LINKd && f->varargs == 1) {
            iarg->setName("_arguments");
            fdecl->ir.irFunc->_arguments = iarg;
            ++iarg;
            iarg->setName("_argptr");
            fdecl->ir.irFunc->_argptr = iarg;
            ++iarg;
        }

        for (; iarg != func->arg_end(); ++iarg)
        {
            if (fdecl->parameters && fdecl->parameters->dim > k)
            {
                Dsymbol* argsym = (Dsymbol*)fdecl->parameters->data[k++];
                VarDeclaration* argvd = argsym->isVarDeclaration();
                assert(argvd);
                assert(!argvd->ir.irLocal);
                argvd->ir.irLocal = new IrLocal(argvd);
                argvd->ir.irLocal->value = iarg;
                iarg->setName(argvd->ident->toChars());
            }
            else
            {
                iarg->setName("unnamed");
            }
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
    if (fd->ir.defined) return;
    fd->ir.defined = true;

    assert(fd->ir.declared);

    Logger::println("DtoDefineFunc(%s): %s", fd->toPrettyChars(), fd->loc.toChars());
    LOG_SCOPE;

    // debug info
    if (global.params.symdebug) {
        Module* mo = fd->getModule();
        fd->ir.irFunc->dwarfSubProg = DtoDwarfSubProgram(fd, DtoDwarfCompileUnit(mo));
    }

    Type* t = DtoDType(fd->type);
    TypeFunction* f = (TypeFunction*)t;
    assert(f->ir.type);

    llvm::Function* func = fd->ir.irFunc->func;
    const llvm::FunctionType* functype = func->getFunctionType();

    // only members of the current module or template instances maybe be defined
    if (!(fd->getModule() == gIR->dmodule || DtoIsTemplateInstance(fd->parent)))
        return;

    // set module owner
    fd->ir.DModule = gIR->dmodule;

    // is there a body?
    if (fd->fbody == NULL)
        return;

    Logger::println("Doing function body for: %s", fd->toChars());
    assert(fd->ir.irFunc);
    gIR->functions.push_back(fd->ir.irFunc);

    if (fd->isMain())
        gIR->emitMain = true;

    std::string entryname("entry_");
    entryname.append(fd->toPrettyChars());

    llvm::BasicBlock* beginbb = llvm::BasicBlock::Create(entryname,func);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endentry",func);

    //assert(gIR->scopes.empty());
    gIR->scopes.push_back(IRScope(beginbb, endbb));

    // create alloca point
    llvm::Instruction* allocaPoint = new llvm::AllocaInst(LLType::Int32Ty, "alloca point", beginbb);
    gIR->func()->allocapoint = allocaPoint;

    // need result variable? (not nested)
    if (fd->vresult && !fd->vresult->nestedref) {
        Logger::println("non-nested vresult value");
        fd->vresult->ir.irLocal = new IrLocal(fd->vresult);
        fd->vresult->ir.irLocal->value = new llvm::AllocaInst(DtoType(fd->vresult->type),"function_vresult",allocaPoint);
    }

    // give arguments storage
    if (fd->parameters)
    {
        size_t n = fd->parameters->dim;
        for (int i=0; i < n; ++i)
        {
            Dsymbol* argsym = (Dsymbol*)fd->parameters->data[i];
            VarDeclaration* vd = argsym->isVarDeclaration();
            assert(vd);

            if (!vd->needsStorage || vd->nestedref || vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type))
                continue;

            LLValue* a = vd->ir.irLocal->value;
            assert(a);
            std::string s(a->getName());
            Logger::println("giving argument '%s' storage", s.c_str());
            s.append("_storage");

            LLValue* v = new llvm::AllocaInst(a->getType(),s,allocaPoint);
            gIR->ir->CreateStore(a,v);
            vd->ir.irLocal->value = v;
        }
    }

    // debug info
    if (global.params.symdebug) DtoDwarfFuncStart(fd);

    LLValue* parentNested = NULL;
    if (FuncDeclaration* fd2 = fd->toParent2()->isFuncDeclaration()) {
        if (!fd->isStatic()) // huh?
            parentNested = fd2->ir.irFunc->nestedVar;
    }

    // need result variable? (nested)
    if (fd->vresult && fd->vresult->nestedref) {
        Logger::println("nested vresult value: %s", fd->vresult->toChars());
        fd->nestedVars.insert(fd->vresult);
    }

    // construct nested variables struct
    if (!fd->nestedVars.empty() || parentNested) {
        std::vector<const LLType*> nestTypes;
        int j = 0;
        if (parentNested) {
            nestTypes.push_back(parentNested->getType());
            j++;
        }
        for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i) {
            VarDeclaration* vd = *i;
            Logger::println("referenced nested variable %s", vd->toChars());
            if (!vd->ir.irLocal)
                vd->ir.irLocal = new IrLocal(vd);
            vd->ir.irLocal->nestedIndex = j++;
            if (vd->isParameter()) {
                if (!vd->ir.irLocal->value) {
                    assert(vd == fd->vthis);
                    vd->ir.irLocal->value = fd->ir.irFunc->thisVar;
                }
                assert(vd->ir.irLocal->value);
                nestTypes.push_back(vd->ir.irLocal->value->getType());
            }
            else {
                nestTypes.push_back(DtoType(vd->type));
            }
        }
        const llvm::StructType* nestSType = llvm::StructType::get(nestTypes);
        Logger::cout() << "nested var struct has type:" << *nestSType << '\n';
        fd->ir.irFunc->nestedVar = new llvm::AllocaInst(nestSType,"nestedvars",allocaPoint);
        if (parentNested) {
            assert(fd->ir.irFunc->thisVar);
            LLValue* ptr = gIR->ir->CreateBitCast(fd->ir.irFunc->thisVar, parentNested->getType(), "tmp");
            gIR->ir->CreateStore(ptr, DtoGEPi(fd->ir.irFunc->nestedVar, 0,0, "tmp"));
        }
        for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i) {
            VarDeclaration* vd = *i;
            if (vd->isParameter()) {
                assert(vd->ir.irLocal);
                gIR->ir->CreateStore(vd->ir.irLocal->value, DtoGEPi(fd->ir.irFunc->nestedVar, 0, vd->ir.irLocal->nestedIndex, "tmp"));
                vd->ir.irLocal->value = fd->ir.irFunc->nestedVar;
            }
        }
    }

    // copy _argptr to a memory location
    if (f->linkage == LINKd && f->varargs == 1)
    {
        LLValue* argptrmem = new llvm::AllocaInst(fd->ir.irFunc->_argptr->getType(), "_argptrmem", gIR->topallocapoint());
        new llvm::StoreInst(fd->ir.irFunc->_argptr, argptrmem, gIR->scopebb());
        fd->ir.irFunc->_argptr = argptrmem;
    }

    // output function body
    fd->fbody->toIR(gIR);

    // llvm requires all basic blocks to end with a TerminatorInst but DMD does not put a return statement
    // in automatically, so we do it here.
    if (!fd->isMain()) {
        if (!gIR->scopereturned()) {
            // pass the previous block into this block
            if (global.params.symdebug) DtoDwarfFuncEnd(fd);
            if (func->getReturnType() == LLType::VoidTy) {
                llvm::ReturnInst::Create(gIR->scopebb());
            }
            else {
                llvm::ReturnInst::Create(llvm::UndefValue::get(func->getReturnType()), gIR->scopebb());
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
            /*if (func->getReturnType() == LLType::VoidTy) {
                llvm::ReturnInst::Create(lastbb);
            }
            else {
                llvm::ReturnInst::Create(llvm::UndefValue::get(func->getReturnType()), lastbb);
            }*/
        }
    }

    // if the last block is not terminated we return a null value or void
    // for some unknown reason this is needed when a void main() has a inline asm block ...
    // this should be harmless for well formed code!
    lastbb = &func->getBasicBlockList().back();
    if (!lastbb->getTerminator())
    {
        Logger::println("adding missing return statement");
        if (func->getReturnType() == LLType::VoidTy)
            llvm::ReturnInst::Create(lastbb);
        else
            llvm::ReturnInst::Create(llvm::Constant::getNullValue(func->getReturnType()), lastbb);
    }

    gIR->functions.pop_back();
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
    // byval arg, but expr has no storage yet
    else if (DtoIsPassedByRef(argexp->type) && (arg->isSlice() || arg->isComplex() || arg->isNull()))
    {
        LLValue* alloc = new llvm::AllocaInst(DtoType(argexp->type), "tmpparam", gIR->topallocapoint());
        DVarValue* vv = new DVarValue(argexp->type, alloc, true);
        DtoAssign(vv, arg);
        arg = vv;
    }

    return arg;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoVariadicArgument(Expression* argexp, LLValue* dst)
{
    Logger::println("DtoVariadicArgument");
    LOG_SCOPE;
    DVarValue* vv = new DVarValue(argexp->type, dst, true);
    gIR->exps.push_back(IRExp(NULL, argexp, vv));
    DtoAssign(vv, argexp->toElem(gIR));
    gIR->exps.pop_back();
}

//////////////////////////////////////////////////////////////////////////////////////////
