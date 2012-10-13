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
#include "id.h"

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
#include "gen/nested.h"
#include "gen/pragma.h"

#if LDC_LLVM_VER < 302
using namespace llvm::Attribute;
#endif

llvm::FunctionType* DtoFunctionType(Type* type, Type* thistype, Type* nesttype, bool ismain)
{
    if (Logger::enabled())
        Logger::println("DtoFunctionType(%s)", type->toChars());
    LOG_SCOPE

    // sanity check
    assert(type->ty == Tfunction);
    TypeFunction* f = static_cast<TypeFunction*>(type);

    TargetABI* abi = (f->linkage == LINKintrinsic ? TargetABI::getIntrinsic() : gABI);
    // Tell the ABI we're resolving a new function type
    abi->newFunctionType(f);

    // Do not modify f->fty yet; this function may be called recursively if any
    // of the argument types refer to this type.
    IrFuncTy fty;

    // llvm idx counter
    size_t lidx = 0;

    // main needs a little special handling
    if (ismain)
    {
        fty.ret = new IrFuncTyArg(Type::tint32, false);
    }
    // sane return value
    else
    {
        Type* rt = f->next;
#if LDC_LLVM_VER >= 302
        llvm::Attributes a;
#else
        llvm::Attributes a = None;
#endif

        // sret return
        if (abi->returnInArg(f))
        {
#if LDC_LLVM_VER >= 302
            fty.arg_sret = new IrFuncTyArg(rt, true, llvm::Attributes::get(llvm::Attributes::Builder().addAttribute(llvm::Attributes::StructRet)
                                                                                                      .addAttribute(llvm::Attributes::NoAlias)
                                                                                                      .addAttribute(llvm::Attributes::NoCapture)));
#else
            fty.arg_sret = new IrFuncTyArg(rt, true, StructRet | NoAlias | NoCapture);
#endif
            rt = Type::tvoid;
            lidx++;
        }
        // sext/zext return
        else
        {
            Type *t = rt;
#if DMDV2
            if (f->isref)
                t = t->pointerTo();
#endif
            if (llvm::Attributes atts = DtoShouldExtend(t))
                a = atts;
        }
#if DMDV2
        fty.ret = new IrFuncTyArg(rt, f->isref, a);
#else
        fty.ret = new IrFuncTyArg(rt, false, a);
#endif
    }
    lidx++;

    // member functions
    if (thistype)
    {
        fty.arg_this = new IrFuncTyArg(thistype, thistype->toBasetype()->ty == Tstruct);
        lidx++;
    }

    // and nested functions
    else if (nesttype)
    {
        fty.arg_nest = new IrFuncTyArg(nesttype, false);
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
                fty.arg_arguments = new IrFuncTyArg(Type::typeinfo->type->arrayOf(), false);
                lidx++;
                // _argptr
#if LDC_LLVM_VER >= 302
                fty.arg_argptr = new IrFuncTyArg(Type::tvoid->pointerTo(), false,
                                                 llvm::Attributes::get(llvm::Attributes::Builder().addAttribute(llvm::Attributes::NoAlias)
                                                                                                  .addAttribute(llvm::Attributes::NoCapture)));
#else
                fty.arg_argptr = new IrFuncTyArg(Type::tvoid->pointerTo(), false, NoAlias | NoCapture);
#endif
                lidx++;
            }
        }
        else
        {
            // Default to C-style varargs for non-extern(D) variadic functions.
            // This seems to be what DMD does.
            fty.c_vararg = true;
        }
    }

    // if this _Dmain() doesn't have an argument, we force it to have one
    int nargs = Parameter::dim(f->parameters);

    if (ismain && nargs == 0)
    {
        Type* mainargs = Type::tchar->arrayOf()->arrayOf();
        fty.args.push_back(new IrFuncTyArg(mainargs, false));
        lidx++;
    }
    // add explicit parameters
    else for (int i = 0; i < nargs; i++)
    {
        // get argument
        Parameter* arg = Parameter::getNth(f->parameters, i);

        // reference semantics? ref, out and d1 static arrays are
        bool byref = arg->storageClass & (STCref|STCout);
#if !SARRAYVALUE
        byref = byref || (arg->type->toBasetype()->ty == Tsarray);
#endif

        Type* argtype = arg->type;
#if LDC_LLVM_VER >= 302
        llvm::Attributes a;
#else
        llvm::Attributes a = None;
#endif

        // handle lazy args
        if (arg->storageClass & STClazy)
        {
            Logger::println("lazy param");
            TypeFunction *ltf = new TypeFunction(NULL, arg->type, 0, LINKd);
            TypeDelegate *ltd = new TypeDelegate(ltf);
            argtype = ltd;
        }
        // byval
        else if (abi->passByVal(byref ? argtype->pointerTo() : argtype))
        {
#if LDC_LLVM_VER >= 302
            if (!byref) a = llvm::Attributes::get(llvm::Attributes::Builder(a).addAttribute(llvm::Attributes::ByVal));
#else
            if (!byref) a |= llvm::Attribute::ByVal;
#endif
            // set byref, because byval requires a pointed LLVM type
            byref = true;
        }
        // sext/zext
        else if (!byref)
        {
            a |= DtoShouldExtend(argtype);
        }

        fty.args.push_back(new IrFuncTyArg(argtype, byref, a));
        lidx++;
    }

    // Now we can modify f->fty safely.
    f->fty = fty;

    // let the abi rewrite the types as necesary
    abi->rewriteFunctionType(f);

    // Tell the ABI we're done with this function type
    abi->doneWithFunctionType();

    // build the function type
    std::vector<LLType*> argtypes;
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

    LLFunctionType* functype = LLFunctionType::get(f->fty.ret->ltype, argtypes, f->fty.c_vararg);

    Logger::cout() << "Final function type: " << *functype << "\n";

    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::FunctionType* DtoVaFunctionType(FuncDeclaration* fdecl)
{
    TypeFunction* f = static_cast<TypeFunction*>(fdecl->type);
    LLFunctionType* fty = 0;

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

    return fty;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl)
{
    // handle for C vararg intrinsics
    if (fdecl->isVaIntrinsic())
        return DtoVaFunctionType(fdecl);

    Type *dthis=0, *dnest=0;

#if DMDV2
    if (fdecl->ident == Id::ensure || fdecl->ident == Id::require) {
        FuncDeclaration *p = fdecl->parent->isFuncDeclaration();
        assert(p);
        AggregateDeclaration *ad = p->isMember2();
        assert(ad);
        dnest = Type::tvoid->pointerTo();
    } else
#endif
    if (fdecl->needThis()) {
        if (AggregateDeclaration* ad = fdecl->isMember2()) {
            Logger::println("isMember = this is: %s", ad->type->toChars());
            dthis = ad->type;
            LLType* thisty = DtoType(dthis);
            //Logger::cout() << "this llvm type: " << *thisty << '\n';
            if (ad->isStructDeclaration())
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

    LLFunctionType* functype = DtoFunctionType(fdecl->type, dthis, dnest, fdecl->isMain());

    return functype;
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::Function* DtoDeclareVaFunction(FuncDeclaration* fdecl)
{
    DtoVaFunctionType(fdecl);
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
    if ((!global.params.useUnitTests || !fdecl->type) && fdecl->isUnitTestDeclaration()) {
        Logger::println("Ignoring unittest %s", fdecl->toPrettyChars());
        return; // ignore declaration completely
    }

    if (fdecl->ir.resolved) return;
    fdecl->ir.resolved = true;

    Type *type = fdecl->type;
    // If errors occurred compiling it, such as bugzilla 6118
    if (type && type->ty == Tfunction) {
        Type *next = static_cast<TypeFunction *>(type)->next;
        if (!next || next->ty == Terror)
            return;
    }

    //printf("resolve function: %s\n", fdecl->toPrettyChars());

    if (fdecl->parent)
    if (TemplateInstance* tinst = fdecl->parent->isTemplateInstance())
    {
        TemplateDeclaration* tempdecl = tinst->tempdecl;
        if (tempdecl->llvmInternal == LLVMva_arg)
        {
            Logger::println("magic va_arg found");
            fdecl->llvmInternal = LLVMva_arg;
            fdecl->ir.resolved = true;
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
            static_cast<TypeFunction*>(fdecl->type)->linkage = LINKintrinsic;
        }
        else if (tempdecl->llvmInternal == LLVMinline_asm)
        {
            Logger::println("magic inline asm found");
            TypeFunction* tf = static_cast<TypeFunction*>(fdecl->type);
            if (tf->varargs != 1 || (fdecl->parameters && fdecl->parameters->dim != 0))
            {
                error("invalid __asm declaration, must be a D style variadic with no explicit parameters");
                fatal();
            }
            fdecl->llvmInternal = LLVMinline_asm;
            fdecl->ir.resolved = true;
            fdecl->ir.declared = true;
            fdecl->ir.initialized = true;
            fdecl->ir.defined = true;
            return; // this gets mapped to a special inline asm call, no point in going on.
        }
    }

    DtoType(fdecl->type);

    Logger::println("DtoResolveFunction(%s): %s", fdecl->toPrettyChars(), fdecl->loc.toChars());
    LOG_SCOPE;

    // queue declaration unless the function is abstract without body
    if (!fdecl->isAbstract() || fdecl->fbody)
    {
        DtoDeclareFunction(fdecl);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

static void set_param_attrs(TypeFunction* f, llvm::Function* func, FuncDeclaration* fdecl)
{
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
    size_t n = Parameter::dim(f->parameters);
#if LDC_LLVM_VER >= 302
    LLSmallVector<llvm::Attributes, 8> attrptr(n, llvm::Attributes());
#else
    LLSmallVector<llvm::Attributes, 8> attrptr(n, None);
#endif

    for (size_t k = 0; k < n; ++k)
    {
        Parameter* fnarg = Parameter::getNth(f->parameters, k);
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

#if LDC_LLVM_VER >= 302
    llvm::AttrListPtr attrlist = llvm::AttrListPtr::get(llvm::ArrayRef<llvm::AttributeWithIndex>(attrs));
#else
    llvm::AttrListPtr attrlist = llvm::AttrListPtr::get(attrs.begin(), attrs.end());
#endif
    func->setAttributes(attrlist);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareFunction(FuncDeclaration* fdecl)
{
    DtoResolveFunction(fdecl);

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
    TypeFunction* f = static_cast<TypeFunction*>(t);

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

    LLFunction* vafunc = 0;
    if (fdecl->isVaIntrinsic())
        vafunc = DtoDeclareVaFunction(fdecl);

    // construct function
    LLFunctionType* functype = DtoFunctionType(fdecl);
    LLFunction* func = vafunc ? vafunc : gIR->module->getFunction(mangled_name);
    if (!func) {
        func = LLFunction::Create(functype, DtoLinkage(fdecl), mangled_name, gIR->module);
    } else if (func->getFunctionType() != functype) {
        error(fdecl->loc, "Function type does not match previously declared function with the same mangled name: %s", fdecl->mangle());
    }

    if (Logger::enabled())
        Logger::cout() << "func = " << *func << std::endl;

    // add func to IRFunc
    fdecl->ir.irFunc->func = func;

    // calling convention
    if (!vafunc && fdecl->llvmInternal != LLVMintrinsic)
        func->setCallingConv(DtoCallingConv(fdecl->loc, f->linkage));
    else // fall back to C, it should be the right thing to do
        func->setCallingConv(llvm::CallingConv::C);

    // parameter attributes
    if (!fdecl->isIntrinsic()) {
        set_param_attrs(f, func, fdecl);
        if (global.params.disableRedZone) {
#if LDC_LLVM_VER >= 302
            func->addFnAttr(llvm::Attributes::NoRedZone);
#else
            func->addFnAttr(NoRedZone);
#endif
        }
    }

    // main
    if (fdecl->isMain()) {
        // Detect multiple main functions, which is disallowed. DMD checks this
        // in the glue code, so we need to do it here as well.
        if (gIR->mainFunc) {
            error(fdecl->loc, "only one main function allowed");
        }
        gIR->mainFunc = func;
    }

#if DMDV2
    // shared static ctor
    if (fdecl->isSharedStaticCtorDeclaration()) {
        if (mustDefineSymbol(fdecl)) {
            gIR->sharedCtors.push_back(fdecl);
        }
    }
    // shared static dtor
    else if (StaticDtorDeclaration *dtorDecl = fdecl->isSharedStaticDtorDeclaration()) {
        if (mustDefineSymbol(fdecl)) {
            gIR->sharedDtors.push_front(fdecl);
            if (dtorDecl->vgate)
                gIR->sharedGates.push_front(dtorDecl->vgate);
        }
    } else
#endif
    // static ctor
    if (fdecl->isStaticCtorDeclaration()) {
        if (mustDefineSymbol(fdecl)) {
            gIR->ctors.push_back(fdecl);
        }
    }
    // static dtor
    else if (StaticDtorDeclaration *dtorDecl = fdecl->isStaticDtorDeclaration()) {
        if (mustDefineSymbol(fdecl)) {
            gIR->dtors.push_front(fdecl);
#if DMDV2
            if (dtorDecl->vgate)
                gIR->gates.push_front(dtorDecl->vgate);
#endif
        }
    }

    // we never reference parameters of function prototypes
    std::string str;
   // if (!declareOnly)
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
                int paramIndex = f->fty.reverseParams ? fdecl->parameters->dim-k-1 : k;
                Dsymbol* argsym = static_cast<Dsymbol*>(fdecl->parameters->data[paramIndex]);

                VarDeclaration* argvd = argsym->isVarDeclaration();
                assert(argvd);
                assert(!argvd->ir.irLocal);
                argvd->ir.irParam = new IrParameter(argvd);
                argvd->ir.irParam->value = iarg;
                argvd->ir.irParam->arg = f->fty.args[paramIndex];

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
        Type::sir->addFunctionBody(fdecl->ir.irFunc);
    else
        assert(func->getLinkage() != llvm::GlobalValue::InternalLinkage);
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this isn't too pretty!

void DtoDefineFunction(FuncDeclaration* fd)
{
    DtoDeclareFunction(fd);

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
    fd->ir.irFunc->diSubprogram = DtoDwarfSubProgram(fd);

    Type* t = fd->type->toBasetype();
    TypeFunction* f = static_cast<TypeFunction*>(t);
    // assert(f->irtype);

    llvm::Function* func = fd->ir.irFunc->func;

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

    llvm::BasicBlock* beginbb = llvm::BasicBlock::Create(gIR->context(), entryname,func);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create(gIR->context(), "endentry",func);

    //assert(gIR->scopes.empty());
    gIR->scopes.push_back(IRScope(beginbb, endbb));

    // create alloca point
    // this gets erased when the function is complete, so alignment etc does not matter at all
    llvm::Instruction* allocaPoint = new llvm::AllocaInst(LLType::getInt32Ty(gIR->context()), "alloca point", beginbb);
    irfunction->allocapoint = allocaPoint;

    // debug info - after all allocas, but before any llvm.dbg.declare etc
    DtoDwarfFuncStart(fd);

    // this hack makes sure the frame pointer elimination optimization is disabled.
    // this this eliminates a bunch of inline asm related issues.
    if (fd->hasReturnExp & 8) // has inline asm
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

        LLValue* thismem = thisvar;
    #if STRUCTTHISREF
        if (!f->fty.arg_this->byref)
    #endif
        {
            thismem = DtoRawAlloca(thisvar->getType(), 0, "this"); // FIXME: align?
            DtoStore(thisvar, thismem);
            irfunction->thisArg = thismem;
        }

        assert(!fd->vthis->ir.irParam);
        fd->vthis->ir.irParam = new IrParameter(fd->vthis);
        fd->vthis->ir.irParam->value = thismem;
        fd->vthis->ir.irParam->arg = f->fty.arg_this;
        fd->vthis->ir.irParam->isVthis = true;

        DtoDwarfLocalVariable(thismem, fd->vthis);
    }

    // give the 'nestArg' storage
    if (f->fty.arg_nest)
    {
        LLValue *nestArg = irfunction->nestArg;
        LLValue *val = DtoRawAlloca(nestArg->getType(), 0, "nestedFrame");
        DtoStore(nestArg, val);
        irfunction->nestArg = val;
    }

    // give arguments storage
    // and debug info
    if (fd->parameters)
    {
        size_t n = f->fty.args.size();
        assert(n == fd->parameters->dim);
        for (int i=0; i < n; ++i)
        {
            Dsymbol* argsym = static_cast<Dsymbol*>(fd->parameters->data[i]);
            VarDeclaration* vd = argsym->isVarDeclaration();
            assert(vd);

            IrParameter* irparam = vd->ir.irParam;
            assert(irparam);

        #if DMDV1
            if (vd->nestedref)
            {
                fd->nestedVars.insert(vd);
            }
        #endif

            bool refout = vd->storage_class & (STCref | STCout);
            bool lazy = vd->storage_class & STClazy;
            if (!refout && (!irparam->arg->byref || lazy))
            {
                // alloca a stack slot for this first class value arg
                LLType* argt;
                if (lazy)
                    argt = irparam->value->getType();
                else
                    argt = DtoType(vd->type);
                LLValue* mem = DtoRawAlloca(argt, 0, vd->ident->toChars());

                // let the abi transform the argument back first
                DImValue arg_dval(vd->type, irparam->value);
                f->fty.getParam(vd->type, i, &arg_dval, mem);

                // set the arg var value to the alloca
                irparam->value = mem;
            }

            if (global.params.symdebug && !(isaArgument(irparam->value) && isaArgument(irparam->value)->hasByValAttr()) && !refout)
                DtoDwarfLocalVariable(irparam->value, vd);
        }
    }


#if DMDV1
    // need result variable? (nested)
    if (fd->vresult && fd->vresult->nestedref) {
        Logger::println("nested vresult value: %s", fd->vresult->toChars());
        fd->nestedVars.insert(fd->vresult);
    }

    if (fd->vthis && fd->vthis->nestedref && !fd->nestedVars.empty()) {
        Logger::println("nested vthis value: %s", fd->vthis->toChars());
        fd->nestedVars.insert(fd->vthis);
    }
#endif

    FuncGen fg;
    irfunction->gen = &fg;

    DtoCreateNestedContext(fd);

    if (fd->vresult && !
#if DMDV2
        fd->vresult->nestedrefs.dim // FIXME: not sure here :/
#else
        fd->vresult->nestedref
#endif
    )
    {
        DtoVarDeclaration(fd->vresult);
    }

    // copy _argptr and _arguments to a memory location
    if (f->linkage == LINKd && f->varargs == 1)
    {
        // _argptr
        LLValue* argptrmem = DtoRawAlloca(fd->ir.irFunc->_argptr->getType(), 0, "_argptr_mem");
        new llvm::StoreInst(fd->ir.irFunc->_argptr, argptrmem, gIR->scopebb());
        fd->ir.irFunc->_argptr = argptrmem;

        // _arguments
        LLValue* argumentsmem = DtoRawAlloca(fd->ir.irFunc->_arguments->getType(), 0, "_arguments_mem");
        new llvm::StoreInst(fd->ir.irFunc->_arguments, argumentsmem, gIR->scopebb());
        fd->ir.irFunc->_arguments = argumentsmem;
    }

    // output function body
    fd->fbody->toIR(gIR);
    irfunction->gen = 0;

    // TODO: clean up this mess

//     std::cout << *func << std::endl;

    llvm::BasicBlock* bb = gIR->scopebb();
    if (pred_begin(bb) == pred_end(bb) && bb != &bb->getParent()->getEntryBlock()) {
        // This block is trivially unreachable, so just delete it.
        // (This is a common case because it happens when 'return'
        // is the last statement in a function)
        bb->eraseFromParent();
    } else if (!gIR->scopereturned()) {
        // llvm requires all basic blocks to end with a TerminatorInst but DMD does not put a return statement
        // in automatically, so we do it here.

        // pass the previous block into this block
        DtoDwarfFuncEnd(fd);
        if (func->getReturnType() == LLType::getVoidTy(gIR->context())) {
            llvm::ReturnInst::Create(gIR->context(), gIR->scopebb());
        }
        else if (!fd->isMain()) {
            AsmBlockStatement* asmb = fd->fbody->endsWithAsm();
            if (asmb) {
                assert(asmb->abiret);
                llvm::ReturnInst::Create(gIR->context(), asmb->abiret, bb);
            }
            else {
                llvm::ReturnInst::Create(gIR->context(), llvm::UndefValue::get(func->getReturnType()), bb);
            }
        }
        else
            llvm::ReturnInst::Create(gIR->context(), LLConstant::getNullValue(func->getReturnType()), bb);
    }

//     std::cout << *func << std::endl;

    // erase alloca point
    if (allocaPoint->getParent())
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

llvm::FunctionType* DtoBaseFunctionType(FuncDeclaration* fdecl)
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
        FuncDeclaration* f2 = base->findFunc(fdecl->ident, static_cast<TypeFunction*>(fdecl->type));
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

DValue* DtoArgument(Parameter* fnarg, Expression* argexp)
{
    Logger::println("DtoArgument");
    LOG_SCOPE;

    DValue* arg = argexp->toElem(gIR);

    // ref/out arg
    if (fnarg && (fnarg->storageClass & (STCref | STCout)))
    {
        Loc loc;
        arg = new DImValue(argexp->type, makeLValue(loc, arg));
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
        LLValue* alloc = DtoAlloca(argexp->type, ".tmp_arg");
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
