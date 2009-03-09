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
#include "gen/abi.h"

const llvm::FunctionType* DtoFunctionType(Type* type, Type* thistype, Type* nesttype, bool ismain)
{
    if (Logger::enabled())
        Logger::println("DtoFunctionType(%s)", type->toChars());
    LOG_SCOPE
    // sanity check
    assert(type->ty == Tfunction);
    TypeFunction* f = (TypeFunction*)type;

    // already built ?
    if (type->ir.type != NULL) {
        //assert(f->fty != NULL);
        return llvm::cast<llvm::FunctionType>(type->ir.type->get());
    }

    if (f->linkage != LINKintrinsic) {
        // Tell the ABI we're resolving a new function type
        gABI->newFunctionType(f);
    }

    // start new ir funcTy
    f->fty.reset();

    // llvm idx counter
    size_t lidx = 0;

    // main needs a little special handling
    if (ismain)
    {
        f->fty.ret = new IrFuncTyArg(Type::tint32, false);
    }
    // sane return value
    else
    {
        Type* rt = f->next;
        unsigned a = 0;
        // sret return
        if (f->linkage != LINKintrinsic)
            if (gABI->returnInArg(f))
            {
                f->fty.arg_sret = new IrFuncTyArg(rt, true, llvm::Attribute::StructRet);
                rt = Type::tvoid;
                lidx++;
            }
            // sext/zext return
            else if (unsigned se = DtoShouldExtend(rt))
            {
                a = se;
            }
        f->fty.ret = new IrFuncTyArg(rt, false, a);
    }
    lidx++;

    // member functions
    if (thistype)
    {
        bool toref = (thistype->toBasetype()->ty == Tstruct);
        f->fty.arg_this = new IrFuncTyArg(thistype, toref);
        lidx++;
    }

    // and nested functions
    else if (nesttype)
    {
        f->fty.arg_nest = new IrFuncTyArg(nesttype, false);
        lidx++;
    }

    // vararg functions are special too
    if (f->varargs)
    {
        if (f->linkage == LINKd)
        {
            // d style with hidden args
            // 2 (array) is handled by the frontend
            if (f->varargs == 1)
            {
                // _arguments
                f->fty.arg_arguments = new IrFuncTyArg(Type::typeinfo->type->arrayOf(), false);
                lidx++;
                // _argptr
                f->fty.arg_argptr = new IrFuncTyArg(Type::tvoid->pointerTo(), false);
                lidx++;
            }
        }
        else if (f->linkage == LINKc)
        {
            f->fty.c_vararg = true;
        }
        else
        {
            type->error(0, "invalid linkage for variadic function");
            fatal();
        }
    }

    // if this _Dmain() doesn't have an argument, we force it to have one
    int nargs = Argument::dim(f->parameters);

    if (ismain && nargs == 0)
    {
        Type* mainargs = Type::tchar->arrayOf()->arrayOf();
        f->fty.args.push_back(new IrFuncTyArg(mainargs, false));
        lidx++;
    }
    // add explicit parameters
    else for (int i = 0; i < nargs; i++)
    {
        // get argument
        Argument* arg = Argument::getNth(f->parameters, i);

        // reference semantics? ref, out and static arrays are
        bool byref = (arg->storageClass & (STCref|STCout)) || (arg->type->toBasetype()->ty == Tsarray);

        Type* argtype = arg->type;
        unsigned a = 0;

        // handle lazy args
        if (arg->storageClass & STClazy)
        {
            Logger::println("lazy param");
            TypeFunction *ltf = new TypeFunction(NULL, arg->type, 0, LINKd);
            TypeDelegate *ltd = new TypeDelegate(ltf);
            argtype = ltd;
        }
        // byval
        else if (f->linkage != LINKintrinsic
                && gABI->passByVal(argtype))
        {
            if (!byref) a |= llvm::Attribute::ByVal;
            byref = true;
        }
        // sext/zext
        else if (!byref)
        {
            a |= DtoShouldExtend(argtype);
        }

        f->fty.args.push_back(new IrFuncTyArg(argtype, byref, a));
        lidx++;
    }

    if (f->linkage != LINKintrinsic) {
        // let the abi rewrite the types as necesary
        gABI->rewriteFunctionType(f);

        // Tell the ABI we're done with this function type
        gABI->doneWithFunctionType();
    }

    // build the function type
    std::vector<const LLType*> argtypes;
    argtypes.reserve(lidx);

    if (f->fty.arg_sret) argtypes.push_back(f->fty.arg_sret->ltype);
    if (f->fty.arg_this) argtypes.push_back(f->fty.arg_this->ltype);
    if (f->fty.arg_nest) argtypes.push_back(f->fty.arg_nest->ltype);
    if (f->fty.arg_arguments) argtypes.push_back(f->fty.arg_arguments->ltype);
    if (f->fty.arg_argptr) argtypes.push_back(f->fty.arg_argptr->ltype);

    size_t beg = argtypes.size();
    size_t nargs2 = f->fty.args.size();
    for (size_t i = 0; i < nargs2; i++)
    {
        argtypes.push_back(f->fty.args[i]->ltype);
    }

    // reverse params?
    if (f->fty.reverseParams && nargs2 > 1)
    {
        std::reverse(argtypes.begin() + beg, argtypes.end());
    }

    llvm::FunctionType* functype = llvm::FunctionType::get(f->fty.ret->ltype, argtypes, f->fty.c_vararg);
    f->ir.type = new llvm::PATypeHolder(functype);

    Logger::cout() << "Final function type: " << *functype << "\n";

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

    // create new ir funcTy
    f->fty.reset();
    f->fty.ret = new IrFuncTyArg(Type::tvoid, false);

    f->fty.args.push_back(new IrFuncTyArg(Type::tvoid->pointerTo(), false));

    if (fdecl->llvmInternal == LLVMva_start)
        fty = GET_INTRINSIC_DECL(vastart)->getFunctionType();
    else if (fdecl->llvmInternal == LLVMva_copy) {
        fty = GET_INTRINSIC_DECL(vacopy)->getFunctionType();
        f->fty.args.push_back(new IrFuncTyArg(Type::tvoid->pointerTo(), false));
    }
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

    Type *dthis=0, *dnest=0;

    if (fdecl->needThis()) {
        if (AggregateDeclaration* ad = fdecl->isMember2()) {
            Logger::println("isMember = this is: %s", ad->type->toChars());
            dthis = ad->type;
            const LLType* thisty = DtoType(dthis);
            //Logger::cout() << "this llvm type: " << *thisty << '\n';
            if (isaStruct(thisty) || (!gIR->structs.empty() && thisty == gIR->topstruct()->type->ir.type->get()))
                thisty = getPtrToType(thisty);
        }
        else {
            Logger::println("chars: %s type: %s kind: %s", fdecl->toChars(), fdecl->type->toChars(), fdecl->kind());
            assert(0);
        }
    }
    else if (fdecl->isNested()) {
        dnest = Type::tvoid->pointerTo();
    }

    const llvm::FunctionType* functype = DtoFunctionType(fdecl->type, dthis, dnest, fdecl->isMain());

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
    int funcNumArgs = func->getArgumentList().size();

    LLSmallVector<llvm::AttributeWithIndex, 9> attrs;
    llvm::AttributeWithIndex PAWI;

    int idx = 0;

    // handle implicit args
    #define ADD_PA(X) \
    if (f->fty.X) { \
        if (f->fty.X->attrs) { \
            PAWI.Index = idx; \
            PAWI.Attrs = f->fty.X->attrs; \
            attrs.push_back(PAWI); \
        } \
        idx++; \
    }

    ADD_PA(ret)
    ADD_PA(arg_sret)
    ADD_PA(arg_this)
    ADD_PA(arg_nest)
    ADD_PA(arg_arguments)
    ADD_PA(arg_argptr)

    #undef ADD_PA

    // set attrs on the rest of the arguments
    size_t n = Argument::dim(f->parameters);
    LLSmallVector<unsigned,8> attrptr(n, 0);

    for (size_t k = 0; k < n; ++k)
    {
        Argument* fnarg = Argument::getNth(f->parameters, k);
        assert(fnarg);

        attrptr[k] = f->fty.args[k]->attrs;
    }

    // reverse params?
    if (f->fty.reverseParams)
    {
        std::reverse(attrptr.begin(), attrptr.end());
    }

    // build rest of attrs list
    for (int i = 0; i < n; i++)
    {
        if (attrptr[i])
        {
            PAWI.Index = idx+i;
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

    bool declareOnly = !mustDefineSymbol(fdecl);

    if (fdecl->llvmInternal == LLVMva_start)
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
        func->setCallingConv(DtoCallingConv(fdecl->loc, f->linkage));
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
    if (fdecl->isStaticCtorDeclaration()) {
        if (mustDefineSymbol(fdecl)) {
            gIR->ctors.push_back(fdecl);
        }
    }
    // static dtor
    else if (fdecl->isStaticDtorDeclaration()) {
        if (mustDefineSymbol(fdecl)) {
            gIR->dtors.push_back(fdecl);
        }
    }

    // we never reference parameters of function prototypes
    std::string str;
    if (!declareOnly)
    {
        // name parameters
        llvm::Function::arg_iterator iarg = func->arg_begin();

        if (f->fty.arg_sret) {
            iarg->setName(".sret_arg");
            fdecl->ir.irFunc->retArg = iarg;
            ++iarg;
        }

        if (f->fty.arg_this) {
            iarg->setName(".this_arg");
            fdecl->ir.irFunc->thisArg = iarg;
            assert(fdecl->ir.irFunc->thisArg);
            ++iarg;
        }
        else if (f->fty.arg_nest) {
            iarg->setName(".nest_arg");
            fdecl->ir.irFunc->nestArg = iarg;
            assert(fdecl->ir.irFunc->nestArg);
            ++iarg;
        }

        if (f->fty.arg_argptr) {
            iarg->setName("._arguments");
            fdecl->ir.irFunc->_arguments = iarg;
            ++iarg;
            iarg->setName("._argptr");
            fdecl->ir.irFunc->_argptr = iarg;
            ++iarg;
        }

        int k = 0;

        for (; iarg != func->arg_end(); ++iarg)
        {
            if (fdecl->parameters && fdecl->parameters->dim > k)
            {
                Dsymbol* argsym;
                if (f->fty.reverseParams)
                    argsym = (Dsymbol*)fdecl->parameters->data[fdecl->parameters->dim-k-1];
                else
                    argsym = (Dsymbol*)fdecl->parameters->data[k];

                VarDeclaration* argvd = argsym->isVarDeclaration();
                assert(argvd);
                assert(!argvd->ir.irLocal);
                argvd->ir.irLocal = new IrLocal(argvd);
                argvd->ir.irLocal->value = iarg;

                str = argvd->ident->toChars();
                str.append("_arg");
                iarg->setName(str);

                k++;
            }
            else
            {
                iarg->setName("unnamed");
            }
        }
    }

    if (fdecl->isUnitTestDeclaration() && !declareOnly)
        gIR->unitTests.push_back(fdecl);

    if (!declareOnly)
        gIR->defineList.push_back(fdecl);
    else
        assert(func->getLinkage() != llvm::GlobalValue::InternalLinkage);

    if (Logger::enabled())
        Logger::cout() << "func decl: " << *func << '\n';
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this isn't too pretty!

void DtoDefineFunction(FuncDeclaration* fd)
{
    if (fd->ir.defined) return;
    fd->ir.defined = true;

    assert(fd->ir.declared);

    if (Logger::enabled())
        Logger::println("DtoDefineFunc(%s): %s", fd->toPrettyChars(), fd->loc.toChars());
    LOG_SCOPE;

    // if this function is naked, we take over right away! no standard processing!
    if (fd->naked)
    {
        DtoDefineNakedFunction(fd);
        return;
    }

    // debug info
    if (global.params.symdebug) {
        fd->ir.irFunc->diSubprogram = DtoDwarfSubProgram(fd);
    }

    Type* t = fd->type->toBasetype();
    TypeFunction* f = (TypeFunction*)t;
    assert(f->ir.type);

    llvm::Function* func = fd->ir.irFunc->func;
    const llvm::FunctionType* functype = func->getFunctionType();

    // sanity check
    assert(mustDefineSymbol(fd));

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

    std::string entryname("entry");

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
    if (fd->inlineAsm)
    {
        // emit a call to llvm_eh_unwind_init
        LLFunction* hack = GET_INTRINSIC_DECL(eh_unwind_init);
        gIR->ir->CreateCall(hack, "");
    }

    // give the 'this' argument storage and debug info
    if (f->fty.arg_this)
    {
        LLValue* thisvar = irfunction->thisArg;
        assert(thisvar);

        LLValue* thismem = DtoAlloca(thisvar->getType(), "this");
        DtoStore(thisvar, thismem);
        irfunction->thisArg = thismem;
        
        assert(!fd->vthis->ir.irLocal);
        fd->vthis->ir.irLocal = new IrLocal(fd->vthis);
        fd->vthis->ir.irLocal->value = thismem;

        if (global.params.symdebug)
            DtoDwarfLocalVariable(thismem, fd->vthis);

    #if DMDV2
        if (fd->vthis->nestedrefs.dim)
    #else
        if (fd->vthis->nestedref)
    #endif
        {
            fd->nestedVars.insert(fd->vthis);
        }
    }

    // give arguments storage
    // and debug info
    if (fd->parameters)
    {
        size_t n = f->fty.args.size();
        assert(n == fd->parameters->dim);
        for (int i=0; i < n; ++i)
        {
            Dsymbol* argsym = (Dsymbol*)fd->parameters->data[i];
            VarDeclaration* vd = argsym->isVarDeclaration();
            assert(vd);

            IrLocal* irloc = vd->ir.irLocal;
            assert(irloc);

        #if DMDV2
            if (vd->nestedrefs.dim)
        #else
            if (vd->nestedref)
        #endif
            {
                fd->nestedVars.insert(vd);
            }

            bool refout = vd->storage_class & (STCref | STCout);
            bool lazy = vd->storage_class & STClazy;

            if (!refout && (!f->fty.args[i]->byref || lazy))
            {
                // alloca a stack slot for this first class value arg
                const LLType* argt;
                if (lazy)
                    argt = irloc->value->getType();
                else
                    argt = DtoType(vd->type);
                LLValue* mem = DtoAlloca(argt, vd->ident->toChars());

                // let the abi transform the argument back first
                DImValue arg_dval(vd->type, irloc->value);
                f->fty.getParam(vd->type, i, &arg_dval, mem);

                // set the arg var value to the alloca
                irloc->value = mem;
            }

            if (global.params.symdebug && !(isaArgument(irloc->value) && !isaArgument(irloc->value)->hasByValAttr()) && !refout)
                DtoDwarfLocalVariable(irloc->value, vd);
        }
    }

// need result variable? (nested)
#if DMDV2
    if (fd->vresult && fd->vresult->nestedrefs.dim) {
#else
    if (fd->vresult && fd->vresult->nestedref) {
#endif
        Logger::println("nested vresult value: %s", fd->vresult->toChars());
        fd->nestedVars.insert(fd->vresult);
    }

    // construct nested variables array
    if (!fd->nestedVars.empty())
    {
        Logger::println("has nested frame");
        // start with adding all enclosing parent frames until a static parent is reached
        int nparelems = 0;
        if (!fd->isStatic())
        {
            Dsymbol* par = fd->toParent2();
            while (par)
            {
                if (FuncDeclaration* parfd = par->isFuncDeclaration())
                {
                    nparelems += parfd->nestedVars.size();
                    // stop at first static
                    if (parfd->isStatic())
                        break;
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
                src = DtoLoad(DtoGEPi(thisval, 0,cd->vthis->ir.irField->index, ".vthis"));
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
    #if DMDV2
        if (fd->vresult && fd->vresult->nestedrefs.dim) {
    #else
        if (fd->vresult && fd->vresult->nestedref) {
    #endif
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

    // TODO: clean up this mess

//     std::cout << *func << std::endl;

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
            {
                AsmBlockStatement* asmb = fd->fbody->endsWithAsm();
                if (asmb) {
                    assert(asmb->abiret);
                    llvm::ReturnInst::Create(asmb->abiret, gIR->scopebb());
                }
                else {
                    llvm::ReturnInst::Create(llvm::UndefValue::get(func->getReturnType()), gIR->scopebb());
                }
            }
            else
                llvm::ReturnInst::Create(llvm::Constant::getNullValue(func->getReturnType()), gIR->scopebb());
        }
    }

//     std::cout << *func << std::endl;

    // erase alloca point
    allocaPoint->eraseFromParent();
    allocaPoint = 0;
    gIR->func()->allocapoint = 0;

    gIR->scopes.pop_back();

    // get rid of the endentry block, it's never used
    assert(!func->getBasicBlockList().empty());
    func->getBasicBlockList().pop_back();

    gIR->functions.pop_back();

//     std::cout << *func << std::endl;
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
