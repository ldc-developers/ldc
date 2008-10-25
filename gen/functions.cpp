#include "gen/llvm.h"
#include "llvm/Support/CFG.h"
#include "llvm/Intrinsics.h"

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

#include <algorithm>

const llvm::FunctionType* DtoFunctionType(Type* type, const LLType* thistype, const LLType* nesttype, bool ismain)
{
    assert(type->ty == Tfunction);
    TypeFunction* f = (TypeFunction*)type;

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
    bool usesnest = false;

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
        paramvec.push_back(arrArrTy);
        }
    }
    else{
        assert(rt);
        if (DtoIsReturnedInArg(rt)) {
            rettype = getPtrToType(DtoType(rt));
            actualRettype = LLType::VoidTy;
            f->retInPtr = retinptr = true;
        }
        else {
            rettype = DtoType(rt);
            actualRettype = rettype;
        }

        if (unsigned ea = DtoShouldExtend(rt))
        {
            f->retAttrs |= ea;
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
    else if (nesttype) {
        paramvec.push_back(nesttype);
        usesnest = true;
    }

    if (typesafeVararg) {
        ClassDeclaration* ti = Type::typeinfo;
        ti->toObjFile(0); // TODO: multiobj
        DtoForceConstInitDsymbol(ti);
        assert(ti->ir.irStruct->constInit);
        std::vector<const LLType*> types;
        types.push_back(DtoSize_t());
        types.push_back(getPtrToType(getPtrToType(ti->ir.irStruct->constInit->getType())));
        const LLType* t1 = llvm::StructType::get(types);
        paramvec.push_back(t1);
        paramvec.push_back(getPtrToType(LLType::Int8Ty));
    }

    // number of formal params
    size_t n = Argument::dim(f->parameters);

#if X86_REVERSE_PARAMS
    // on x86 we need to reverse the formal params in some cases to match the ABI
    if (global.params.cpu == ARCHx86)
    {
        // more than one formal arg,
        // extern(D) linkage
        // not a D-style vararg
        if (n > 1 && f->linkage == LINKd && !typesafeVararg)
        {
            f->reverseParams = true;
            f->reverseIndex = paramvec.size();
        }
    }
#endif // X86_REVERSE_PARAMS


    for (int i=0; i < n; ++i) {
        Argument* arg = Argument::getNth(f->parameters, i);
        // ensure scalar
        Type* argT = arg->type->toBasetype();
        assert(argT);

        bool refOrOut = ((arg->storageClass & STCref) || (arg->storageClass & STCout));

        const LLType* at = DtoType(argT);

        // handle lazy args
        if (arg->storageClass & STClazy)
        {
            Logger::println("lazy param");
            TypeFunction *ltf = new TypeFunction(NULL, arg->type, 0, LINKd);
            TypeDelegate *ltd = new TypeDelegate(ltf);
            at = DtoType(ltd);
            paramvec.push_back(at);
        }
        // opaque types need special handling
        else if (llvm::isa<llvm::OpaqueType>(at)) {
            Logger::println("opaque param");
            assert(argT->ty == Tstruct || argT->ty == Tclass);
            paramvec.push_back(getPtrToType(at));
        }
        // structs are passed as a reference, but by value
        else if (argT->ty == Tstruct) {
            Logger::println("struct param");
            if (!refOrOut)
                arg->llvmAttrs |= llvm::Attribute::ByVal;
            paramvec.push_back(getPtrToType(at));
        }
        // static arrays are passed directly by reference
        else if (argT->ty == Tsarray)
        {
            Logger::println("static array param");
            at = getPtrToType(at);
            paramvec.push_back(at);
        }
        // firstclass ' ref/out ' parameter
        else if (refOrOut) {
            Logger::println("ref/out param");
            at = getPtrToType(at);
            paramvec.push_back(at);
        }
        // firstclass ' in ' parameter
        else {
            Logger::println("in param");
            if (unsigned ea = DtoShouldExtend(argT))
                arg->llvmAttrs |= ea;
            paramvec.push_back(at);
        }
    }

    // reverse params?
    if (f->reverseParams)
    {
        std::reverse(paramvec.begin() + f->reverseIndex, paramvec.end());
    }

    // construct function type
    bool isvararg = !(typesafeVararg || arrayVararg) && f->varargs;
    llvm::FunctionType* functype = llvm::FunctionType::get(actualRettype, paramvec, isvararg);

#if X86_PASS_IN_EAX
    // tell first param to be passed in a register if we can
    // ONLY extern(D) functions !
    if ((n > 0 || usesthis || usesnest) && f->linkage == LINKd)
    {
        // FIXME: Only x86 right now ...
        if (global.params.cpu == ARCHx86)
        {
            // pass first param in EAX if it fits, is not floating point and is not a 3 byte struct.
            // FIXME: struct are not passed in EAX yet

            // if there is a implicit context parameter, pass it in EAX
            if (usesthis || usesnest)
            {
                f->thisAttrs |= llvm::Attribute::InReg;
            }
            // otherwise check the first formal parameter
            else
            {
                int inreg = f->reverseParams ? n - 1 : 0;
                Argument* arg = Argument::getNth(f->parameters, inreg);
                Type* t = arg->type->toBasetype();

                // 32bit ints, pointers, classes, static arrays and AAs
                // are candidate for being passed in EAX
                if ((arg->storageClass & STCin) &&
                    ((t->isscalar() && !t->isfloating()) ||
                     t->ty == Tclass || t->ty == Tsarray || t->ty == Taarray) &&
                    (t->size() <= PTRSIZE))
                {
                    arg->llvmAttrs |= llvm::Attribute::InReg;
                }
            }
        }
    }
#endif // X86_PASS_IN_EAX

    // done
    f->retInPtr = retinptr;
    f->usesThis = usesthis;
    f->usesNest = usesnest;

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
    const llvm::FunctionType* fty = 0;

    if (fdecl->llvmInternal == LLVMva_start)
        fty = GET_INTRINSIC_DECL(vastart)->getFunctionType();
    else if (fdecl->llvmInternal == LLVMva_copy)
        fty = GET_INTRINSIC_DECL(vacopy)->getFunctionType();
    else if (fdecl->llvmInternal == LLVMva_end)
        fty = GET_INTRINSIC_DECL(vaend)->getFunctionType();
    assert(fty);

    f->ir.type = new llvm::PATypeHolder(fty);
    return fty;
}

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl)
{
    // handle for C vararg intrinsics
    if (fdecl->isVaIntrinsic())
        return DtoVaFunctionType(fdecl);

    // type has already been resolved
    if (fdecl->type->ir.type != 0)
        return llvm::cast<llvm::FunctionType>(fdecl->type->ir.type->get());

    const LLType* thisty = 0;
    const LLType* nestty = 0;

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
        nestty = getPtrToType(LLType::Int8Ty);
    }

    const llvm::FunctionType* functype = DtoFunctionType(fdecl->type, thisty, nestty, fdecl->isMain());

    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* DtoDeclareVaFunction(FuncDeclaration* fdecl)
{
    TypeFunction* f = (TypeFunction*)fdecl->type->toBasetype();
    const llvm::FunctionType* fty = DtoVaFunctionType(fdecl);
    llvm::Function* func = 0;

    if (fdecl->llvmInternal == LLVMva_start)
        func = GET_INTRINSIC_DECL(vastart);
    else if (fdecl->llvmInternal == LLVMva_copy)
        func = GET_INTRINSIC_DECL(vacopy);
    else if (fdecl->llvmInternal == LLVMva_end)
        func = GET_INTRINSIC_DECL(vaend);
    assert(func);

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

    //printf("resolve function: %s\n", fdecl->toPrettyChars());

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
        else if (tempdecl->llvmInternal == LLVMintrinsic)
        {
            Logger::println("overloaded intrinsic found");
            fdecl->llvmInternal = LLVMintrinsic;
            DtoOverloadedIntrinsicName(tinst, tempdecl, fdecl->intrinsicName);
            fdecl->linkage = LINKintrinsic;
            ((TypeFunction*)fdecl->type)->linkage = LINKintrinsic;
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
    int llidx = 0;
    if (f->retInPtr) ++llidx;
    if (f->usesThis) ++llidx;
    else if (f->usesNest) ++llidx;
    if (f->linkage == LINKd && f->varargs == 1)
        llidx += 2;

    int funcNumArgs = func->getArgumentList().size();

    LLSmallVector<llvm::AttributeWithIndex, 9> attrs;
    llvm::AttributeWithIndex PAWI;

    // set return value attrs if any
    if (f->retAttrs)
    {
        PAWI.Index = 0;
        PAWI.Attrs = f->retAttrs;
        attrs.push_back(PAWI);
    }

    // set sret param
    if (f->retInPtr)
    {
        PAWI.Index = 1;
        PAWI.Attrs = llvm::Attribute::StructRet;
        attrs.push_back(PAWI);
    }

    // set this/nest param attrs
    if (f->thisAttrs)
    {
        PAWI.Index = f->retInPtr ? 2 : 1;
        PAWI.Attrs = f->thisAttrs;
        attrs.push_back(PAWI);
    }

    // set attrs on the rest of the arguments
    size_t n = Argument::dim(f->parameters);
    assert(funcNumArgs >= n); // main might mismatch, for the implicit char[][] arg

    LLSmallVector<unsigned,8> attrptr(n, 0);

    for (size_t k = 0; k < n; ++k)
    {
        Argument* fnarg = Argument::getNth(f->parameters, k);
        assert(fnarg);

        attrptr[k] = fnarg->llvmAttrs;
    }

    // reverse params?
    if (f->reverseParams)
    {
        std::reverse(attrptr.begin(), attrptr.end());
    }

    // build rest of attrs list
    for (int i = 0; i < n; i++)
    {
        if (attrptr[i])
        {
            PAWI.Index = llidx+i+1;
            PAWI.Attrs = attrptr[i];
            attrs.push_back(PAWI);
        }
    }

    llvm::AttrListPtr attrlist = llvm::AttrListPtr::get(attrs.begin(), attrs.end());
    func->setAttributes(attrlist);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareFunction(FuncDeclaration* fdecl)
{
    if (fdecl->ir.declared) return;
    fdecl->ir.declared = true;

    Logger::println("DtoDeclareFunction(%s): %s", fdecl->toPrettyChars(), fdecl->loc.toChars());
    LOG_SCOPE;

    //printf("declare function: %s\n", fdecl->toPrettyChars());

    // intrinsic sanity check
    if (fdecl->llvmInternal == LLVMintrinsic && fdecl->fbody) {
        error(fdecl->loc, "intrinsics cannot have function bodies");
        fatal();
    }

    // get TypeFunction*
    Type* t = fdecl->type->toBasetype();
    TypeFunction* f = (TypeFunction*)t;

    bool declareOnly = false;
    bool templInst = fdecl->parent && DtoIsTemplateInstance(fdecl->parent);
    if (!templInst && fdecl->getModule() != gIR->dmodule)
    {
        Logger::println("not template instance, and not in this module. declare only!");
        Logger::println("current module: %s", gIR->dmodule->ident->toChars());
        if(fdecl->getModule())
            Logger::println("func module: %s", fdecl->getModule()->ident->toChars());
        else {
            Logger::println("func not in a module, is runtime");
        }
        declareOnly = true;
    }
    else if (fdecl->llvmInternal == LLVMva_start)
        declareOnly = true;

    if (!fdecl->ir.irFunc) {
        fdecl->ir.irFunc = new IrFunction(fdecl);
    }

    // mangled name
    const char* mangled_name;
    if (fdecl->llvmInternal == LLVMintrinsic)
        mangled_name = fdecl->intrinsicName.c_str();
    else
        mangled_name = fdecl->mangle();

    llvm::Function* vafunc = 0;
    if (fdecl->isVaIntrinsic())
        vafunc = DtoDeclareVaFunction(fdecl);

    // construct function
    const llvm::FunctionType* functype = DtoFunctionType(fdecl);
    llvm::Function* func = vafunc ? vafunc : gIR->module->getFunction(mangled_name);
    if (!func)
        func = llvm::Function::Create(functype, DtoLinkage(fdecl), mangled_name, gIR->module);

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
    if (!fdecl->isIntrinsic()) {
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

        if (f->retInPtr) {
            iarg->setName(".sretarg");
            fdecl->ir.irFunc->retArg = iarg;
            ++iarg;
        }

        if (f->usesThis) {
            iarg->setName("this");
            fdecl->ir.irFunc->thisArg = iarg;
            assert(fdecl->ir.irFunc->thisArg);
            ++iarg;
        }
        else if (f->usesNest) {
            iarg->setName(".nest");
            fdecl->ir.irFunc->nestArg = iarg;
            assert(fdecl->ir.irFunc->nestArg);
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

        int k = 0;

        for (; iarg != func->arg_end(); ++iarg)
        {
            if (fdecl->parameters && fdecl->parameters->dim > k)
            {
                Dsymbol* argsym;
                if (f->reverseParams)
                    argsym = (Dsymbol*)fdecl->parameters->data[fdecl->parameters->dim-k-1];
                else
                    argsym = (Dsymbol*)fdecl->parameters->data[k];

                VarDeclaration* argvd = argsym->isVarDeclaration();
                assert(argvd);
                assert(!argvd->ir.irLocal);
                argvd->ir.irLocal = new IrLocal(argvd);
                argvd->ir.irLocal->value = iarg;
                iarg->setName(argvd->ident->toChars());

                k++;
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

    if (Logger::enabled())
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

    // error on naked
    if (fd->naked)
    {
        fd->error("naked is not supported");
        fatal();
    }

    // debug info
    if (global.params.symdebug) {
        Module* mo = fd->getModule();
        fd->ir.irFunc->dwarfSubProg = DtoDwarfSubProgram(fd);
    }

    Type* t = fd->type->toBasetype();
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
    IrFunction* irfunction = fd->ir.irFunc;
    gIR->functions.push_back(irfunction);

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
    irfunction->allocapoint = allocaPoint;

    // debug info - after all allocas, but before any llvm.dbg.declare etc
    if (global.params.symdebug) DtoDwarfFuncStart(fd);

    // need result variable?
    if (fd->vresult) {
        Logger::println("vresult value");
        fd->vresult->ir.irLocal = new IrLocal(fd->vresult);
        fd->vresult->ir.irLocal->value = DtoAlloca(DtoType(fd->vresult->type), "function_vresult");
    }
    
    // this hack makes sure the frame pointer elimination optimization is disabled.
    // this this eliminates a bunch of inline asm related issues.
    // naked must always eliminate the framepointer however...
    if (fd->inlineAsm && !fd->naked)
    {
        // emit a call to llvm_eh_unwind_init
        LLFunction* hack = GET_INTRINSIC_DECL(eh_unwind_init);
        gIR->ir->CreateCall(hack, "");
    }

    // give the 'this' argument storage and debug info
    if (f->usesThis)
    {
        LLValue* thisvar = irfunction->thisArg;
        assert(thisvar);

        LLValue* thismem = DtoAlloca(thisvar->getType(), ".this");
        DtoStore(thisvar, thismem);
        irfunction->thisArg = thismem;
        
        assert(!fd->vthis->ir.irLocal);
        fd->vthis->ir.irLocal = new IrLocal(fd->vthis);
        fd->vthis->ir.irLocal->value = thismem;

        if (global.params.symdebug)
            DtoDwarfLocalVariable(thismem, fd->vthis);
        
        if (fd->vthis->nestedref)
            fd->nestedVars.insert(fd->vthis);
    }

    // give arguments storage
    // and debug info
    if (fd->parameters)
    {
        size_t n = fd->parameters->dim;
        for (int i=0; i < n; ++i)
        {
            Dsymbol* argsym = (Dsymbol*)fd->parameters->data[i];
            VarDeclaration* vd = argsym->isVarDeclaration();
            assert(vd);
            
            if (vd->nestedref)
                fd->nestedVars.insert(vd);

            IrLocal* irloc = vd->ir.irLocal;
            assert(irloc);

            bool refout = vd->storage_class & (STCref | STCout);
            bool lazy = vd->storage_class & STClazy;

            if (refout)
            {
                continue;
            }
            else if (!lazy && DtoIsPassedByRef(vd->type))
            {
                LLValue* vdirval = irloc->value;
                if (global.params.symdebug && !(isaArgument(vdirval) && !isaArgument(vdirval)->hasByValAttr()))
                    DtoDwarfLocalVariable(vdirval, vd);
                continue;
            }

            LLValue* a = irloc->value;
            LLValue* v = DtoAlloca(a->getType(), "."+a->getName());
            DtoStore(a,v);
            irloc->value = v;
        }
    }

    // need result variable? (nested)
    if (fd->vresult && fd->vresult->nestedref) {
        Logger::println("nested vresult value: %s", fd->vresult->toChars());
        fd->nestedVars.insert(fd->vresult);
    }

    // construct nested variables array
    if (!fd->nestedVars.empty())
    {
        Logger::println("has nested frame");
        // start with add all enclosing parent frames
        int nparelems = 0;
        Dsymbol* par = fd->toParent2();
        while (par)
        {
            if (FuncDeclaration* parfd = par->isFuncDeclaration())
            {
                nparelems += parfd->nestedVars.size();
            }
            else if (ClassDeclaration* parcd = par->isClassDeclaration())
            {
                // nothing needed
            }
            else
            {
                break;
            }
            par = par->toParent2();
        }
        int nelems = fd->nestedVars.size() + nparelems;
        
        // make array type for nested vars
        const LLType* nestedVarsTy = LLArrayType::get(getVoidPtrType(), nelems);
    
        // alloca it
        LLValue* nestedVars = DtoAlloca(nestedVarsTy, ".nested_vars");
        
        // copy parent frame into beginning
        if (nparelems)
        {
            LLValue* src = irfunction->nestArg;
            if (!src)
            {
                assert(irfunction->thisArg);
                assert(fd->isMember2());
                LLValue* thisval = DtoLoad(irfunction->thisArg);
                ClassDeclaration* cd = fd->isMember2()->isClassDeclaration();
                assert(cd);
                assert(cd->vthis);
                src = DtoLoad(DtoGEPi(thisval, 0,2+cd->vthis->ir.irField->index, ".vthis"));
            }
            DtoMemCpy(nestedVars, src, DtoConstSize_t(nparelems*PTRSIZE));
        }
        
        // store in IrFunction
        irfunction->nestedVar = nestedVars;
        
        // go through all nested vars and assign indices
        int idx = nparelems;
        for (std::set<VarDeclaration*>::iterator i=fd->nestedVars.begin(); i!=fd->nestedVars.end(); ++i)
        {
            VarDeclaration* vd = *i;
            if (!vd->ir.irLocal)
                vd->ir.irLocal = new IrLocal(vd);

            if (vd->isParameter())
            {
                Logger::println("nested param: %s", vd->toChars());
                LLValue* gep = DtoGEPi(nestedVars, 0, idx);
                LLValue* val = DtoBitCast(vd->ir.irLocal->value, getVoidPtrType());
                DtoStore(val, gep);
            }
            else
            {
                Logger::println("nested var:   %s", vd->toChars());
            }

            vd->ir.irLocal->nestedIndex = idx++;
        }
        
        // fixup nested result variable
        if (fd->vresult && fd->vresult->nestedref) {
            Logger::println("nested vresult value: %s", fd->vresult->toChars());
            LLValue* gep = DtoGEPi(nestedVars, 0, fd->vresult->ir.irLocal->nestedIndex);
            LLValue* val = DtoBitCast(fd->vresult->ir.irLocal->value, getVoidPtrType());
            DtoStore(val, gep);
        }
    }

    // copy _argptr and _arguments to a memory location
    if (f->linkage == LINKd && f->varargs == 1)
    {
        // _argptr
        LLValue* argptrmem = DtoAlloca(fd->ir.irFunc->_argptr->getType(), "_argptr_mem");
        new llvm::StoreInst(fd->ir.irFunc->_argptr, argptrmem, gIR->scopebb());
        fd->ir.irFunc->_argptr = argptrmem;

        // _arguments
        LLValue* argumentsmem = DtoAlloca(fd->ir.irFunc->_arguments->getType(), "_arguments_mem");
        new llvm::StoreInst(fd->ir.irFunc->_arguments, argumentsmem, gIR->scopebb());
        fd->ir.irFunc->_arguments = argumentsmem;
    }

    // output function body
    fd->fbody->toIR(gIR);

    // llvm requires all basic blocks to end with a TerminatorInst but DMD does not put a return statement
    // in automatically, so we do it here.
    if (!gIR->scopereturned()) {
        // pass the previous block into this block
        if (global.params.symdebug) DtoDwarfFuncEnd(fd);
        if (func->getReturnType() == LLType::VoidTy) {
            llvm::ReturnInst::Create(gIR->scopebb());
        }
        else {
            if (!fd->isMain())
                llvm::ReturnInst::Create(llvm::UndefValue::get(func->getReturnType()), gIR->scopebb());
            else
                llvm::ReturnInst::Create(llvm::Constant::getNullValue(func->getReturnType()), gIR->scopebb());
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
    if (lastbb->empty())
    {
        new llvm::UnreachableInst(lastbb);
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
    if (fnarg && (fnarg->storageClass & (STCref | STCout)))
    {
        if (arg->isVar() || arg->isLRValue())
            arg = new DImValue(argexp->type, arg->getLVal());
        else
            arg = new DImValue(argexp->type, arg->getRVal());
    }
    // lazy arg
    else if (fnarg && (fnarg->storageClass & STClazy))
    {
        assert(argexp->type->toBasetype()->ty == Tdelegate);
        assert(!arg->isLVal());
        return arg;
    }
    // byval arg, but expr has no storage yet
    else if (DtoIsPassedByRef(argexp->type) && (arg->isSlice() || arg->isNull()))
    {
        LLValue* alloc = DtoAlloca(DtoType(argexp->type), ".tmp_arg");
        DVarValue* vv = new DVarValue(argexp->type, alloc);
        DtoAssign(argexp->loc, vv, arg);
        arg = vv;
    }

    return arg;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoVariadicArgument(Expression* argexp, LLValue* dst)
{
    Logger::println("DtoVariadicArgument");
    LOG_SCOPE;
    DVarValue vv(argexp->type, dst);
    DtoAssign(argexp->loc, &vv, argexp->toElem(gIR));
}

//////////////////////////////////////////////////////////////////////////////////////////

bool FuncDeclaration::isIntrinsic()
{
    return (llvmInternal == LLVMintrinsic || isVaIntrinsic());
}

bool FuncDeclaration::isVaIntrinsic()
{
    return (llvmInternal == LLVMva_start ||
            llvmInternal == LLVMva_copy ||
            llvmInternal == LLVMva_end);
}
