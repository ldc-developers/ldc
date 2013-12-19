//===-- functions.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/functions.h"
#include "aggregate.h"
#include "declaration.h"
#include "id.h"
#include "init.h"
#include "module.h"
#include "mtype.h"
#include "statement.h"
#include "template.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/Intrinsics.h"
#else
#include "llvm/Intrinsics.h"
#endif
#include "llvm/Support/CFG.h"
#include <iostream>

#if LDC_LLVM_VER < 302
using namespace llvm::Attribute;
#endif

llvm::FunctionType* DtoFunctionType(Type* type, IrFuncTy &irFty, Type* thistype, Type* nesttype, bool isMain, bool isCtor)
{
    if (Logger::enabled())
        Logger::println("DtoFunctionType(%s)", type->toChars());
    LOG_SCOPE

    // sanity check
    assert(type->ty == Tfunction);
    TypeFunction* f = static_cast<TypeFunction*>(type);
    assert(f->next && "Encountered function type with invalid return type; "
        "trying to codegen function ignored by the frontend?");

    // Return cached type if available
    if (irFty.funcType) return irFty.funcType;

    TargetABI* abi = (f->linkage == LINKintrinsic ? TargetABI::getIntrinsic() : gABI);
    // Tell the ABI we're resolving a new function type
    abi->newFunctionType(f);

    // Do not modify irFty yet; this function may be called recursively if any
    // of the argument types refer to this type.
    IrFuncTy newIrFty;

    // llvm idx counter
    size_t lidx = 0;

    // main needs a little special handling
    if (isMain)
    {
        newIrFty.ret = new IrFuncTyArg(Type::tint32, false);
    }
    // sane return value
    else
    {
        Type* rt = f->next;
#if LDC_LLVM_VER >= 302
        llvm::AttrBuilder attrBuilder;
#else
        llvm::Attributes a = None;
#endif

        // sret return
        if (abi->returnInArg(f))
        {
#if LDC_LLVM_VER >= 302
#if LDC_LLVM_VER >= 303
            newIrFty.arg_sret = new IrFuncTyArg(rt, true,
                llvm::AttrBuilder().addAttribute(llvm::Attribute::StructRet)
                .addAttribute(llvm::Attribute::NoAlias)
#else
            newIrFty.arg_sret = new IrFuncTyArg(rt, true, llvm::Attributes::get(gIR->context(),
                llvm::AttrBuilder().addAttribute(llvm::Attributes::StructRet)
                .addAttribute(llvm::Attributes::NoAlias)
#endif
#if LDC_LLVM_VER == 302
            )
#endif
            );
#else
            newIrFty.arg_sret = new IrFuncTyArg(rt, true, StructRet | NoAlias
            );
#endif
            rt = Type::tvoid;
            lidx++;
        }
        // sext/zext return
        else
        {
            Type *t = rt;
            if (f->isref)
                t = t->pointerTo();
#if LDC_LLVM_VER >= 303
            if (llvm::Attribute::AttrKind a = DtoShouldExtend(t))
                attrBuilder.addAttribute(a);
#elif LDC_LLVM_VER == 302
            if (llvm::Attributes::AttrVal a = DtoShouldExtend(t))
                attrBuilder.addAttribute(a);
#else
            a = DtoShouldExtend(t);
#endif
        }
#if LDC_LLVM_VER >= 303
        llvm::AttrBuilder a = attrBuilder;
#elif LDC_LLVM_VER == 302
        llvm::Attributes a = llvm::Attributes::get(gIR->context(), attrBuilder);
#endif
        newIrFty.ret = new IrFuncTyArg(rt, f->isref, a);
    }
    lidx++;

    // member functions
    if (thistype)
    {
#if LDC_LLVM_VER >= 303
        llvm::AttrBuilder attrBuilder;
        if (isCtor)
            attrBuilder.addAttribute(llvm::Attribute::Returned);
#endif
        newIrFty.arg_this = new IrFuncTyArg(thistype, thistype->toBasetype()->ty == Tstruct
#if LDC_LLVM_VER >= 303
                                       , attrBuilder
#endif
                                       );
        lidx++;
    }

    // and nested functions
    else if (nesttype)
    {
        newIrFty.arg_nest = new IrFuncTyArg(nesttype, false);
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
                newIrFty.arg_arguments = new IrFuncTyArg(Type::dtypeinfo->type->arrayOf(), false);
                lidx++;
                // _argptr
#if LDC_LLVM_VER >= 303
                newIrFty.arg_argptr = new IrFuncTyArg(Type::tvoid->pointerTo(), false,
                                                 llvm::AttrBuilder().addAttribute(llvm::Attribute::NoAlias)
                                                                    .addAttribute(llvm::Attribute::NoCapture));
#elif LDC_LLVM_VER == 302
                newIrFty.arg_argptr = new IrFuncTyArg(Type::tvoid->pointerTo(), false,
                                                 llvm::Attributes::get(gIR->context(), llvm::AttrBuilder().addAttribute(llvm::Attributes::NoAlias)
                                                                                                          .addAttribute(llvm::Attributes::NoCapture)));
#else
                newIrFty.arg_argptr = new IrFuncTyArg(Type::tvoid->pointerTo(), false, NoAlias | NoCapture);
#endif
                lidx++;
            }
        }
        else
        {
            // Default to C-style varargs for non-extern(D) variadic functions.
            // This seems to be what DMD does.
            newIrFty.c_vararg = true;
        }
    }

    // if this _Dmain() doesn't have an argument, we force it to have one
    int nargs = Parameter::dim(f->parameters);

    if (isMain && nargs == 0)
    {
        Type* mainargs = Type::tchar->arrayOf()->arrayOf();
        newIrFty.args.push_back(new IrFuncTyArg(mainargs, false));
        lidx++;
    }
    // add explicit parameters
    else for (int i = 0; i < nargs; i++)
    {
        // get argument
        Parameter* arg = Parameter::getNth(f->parameters, i);

        // reference semantics? ref, out and d1 static arrays are
        bool byref = arg->storageClass & (STCref|STCout);

        Type* argtype = arg->type;
#if LDC_LLVM_VER >= 302
        llvm::AttrBuilder attrBuilder;
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
#if LDC_LLVM_VER >= 303
            if (!byref) attrBuilder.addAttribute(llvm::Attribute::ByVal);
#elif LDC_LLVM_VER == 302
            if (!byref) attrBuilder.addAttribute(llvm::Attributes::ByVal);
#else
            if (!byref) a |= llvm::Attribute::ByVal;
#endif
            // set byref, because byval requires a pointed LLVM type
            byref = true;
        }
        // sext/zext
        else if (!byref)
        {
#if LDC_LLVM_VER >= 303
            if (llvm::Attribute::AttrKind a = DtoShouldExtend(argtype))
                attrBuilder.addAttribute(a);
#elif LDC_LLVM_VER == 302
            if (llvm::Attributes::AttrVal a = DtoShouldExtend(argtype))
                attrBuilder.addAttribute(a);
#else
            a |= DtoShouldExtend(argtype);
#endif
        }
#if LDC_LLVM_VER >= 303
        llvm::AttrBuilder a = attrBuilder;
#elif LDC_LLVM_VER == 302
        llvm::Attributes a = llvm::Attributes::get(gIR->context(), attrBuilder);
#endif
        newIrFty.args.push_back(new IrFuncTyArg(argtype, byref, a));
        lidx++;
    }

    // Now we can modify irFty safely.
    irFty = newIrFty;

    // let the abi rewrite the types as necesary
    abi->rewriteFunctionType(f, irFty);

    // Tell the ABI we're done with this function type
    abi->doneWithFunctionType();

    // build the function type
    std::vector<LLType*> argtypes;
    argtypes.reserve(lidx);

    if (irFty.arg_sret) argtypes.push_back(irFty.arg_sret->ltype);
    if (irFty.arg_this) argtypes.push_back(irFty.arg_this->ltype);
    if (irFty.arg_nest) argtypes.push_back(irFty.arg_nest->ltype);
    if (irFty.arg_arguments) argtypes.push_back(irFty.arg_arguments->ltype);
    if (irFty.arg_argptr) argtypes.push_back(irFty.arg_argptr->ltype);

    size_t beg = argtypes.size();
    size_t nargs2 = irFty.args.size();
    for (size_t i = 0; i < nargs2; i++)
    {
        argtypes.push_back(irFty.args[i]->ltype);
    }

    // reverse params?
    if (irFty.reverseParams && nargs2 > 1)
    {
        std::reverse(argtypes.begin() + beg, argtypes.end());
    }

    irFty.funcType = LLFunctionType::get(irFty.ret->ltype, argtypes, irFty.c_vararg);

    Logger::cout() << "Final function type: " << *irFty.funcType << "\n";

    return irFty.funcType;
}

//////////////////////////////////////////////////////////////////////////////////////////

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Assembly/Parser.h"

LLFunction* DtoInlineIRFunction(FuncDeclaration* fdecl)
{
    const char* mangled_name = fdecl->mangleExact();
    TemplateInstance* tinst = fdecl->parent->isTemplateInstance();
    assert(tinst);

    Objects& objs = tinst->tdtypes;
    assert(objs.dim == 3);

    Expression* a0 = isExpression(objs[0]);
    assert(a0);
    StringExp* strexp = a0->toString();
    assert(strexp);
    assert(strexp->sz == 1);
    std::string code(static_cast<char*>(strexp->string), strexp->len);

    Type* ret = isType(objs[1]);
    assert(ret);

    Tuple* a2 = isTuple(objs[2]);
    assert(a2);
    Objects& arg_types = a2->objects;

    std::string str;
    llvm::raw_string_ostream stream(str);
    stream << "define " << *DtoType(ret) << " @" << mangled_name << "(";

    for(size_t i = 0; ;)
    {
        Type* ty = isType(arg_types[i]);
        //assert(ty);
        if(!ty)
        {
            error(tinst->loc,
                "All parameters of a template defined with pragma llvm_inline_ir, except for the first one, should be types");
            fatal();
        }
        stream << *DtoType(ty);

        i++;
        if(i >= arg_types.dim)
            break;

        stream << ", ";
    }

    if(ret->ty == Tvoid)
        code.append("\nret void");

    stream << ")\n{\n" << code <<  "\n}";

    llvm::SMDiagnostic err;
    llvm::ParseAssemblyString(stream.str().c_str(), gIR->module, err, gIR->context());
    std::string errstr = err.getMessage();
    if(errstr != "")
        error(tinst->loc,
            "can't parse inline LLVM IR:\n%s\n%s\n%s\nThe input string was: \n%s",
#if LDC_LLVM_VER >= 303
            err.getLineContents().str().c_str(),
#else
            err.getLineContents().c_str(),
#endif
            (std::string(err.getColumnNo(), ' ') + '^').c_str(),
            errstr.c_str(), stream.str().c_str());

    LLFunction* fun = gIR->module->getFunction(mangled_name);
    fun->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
#if LDC_LLVM_VER >= 303
    fun->addFnAttr(llvm::Attribute::AlwaysInline);
#elif LDC_LLVM_VER == 302
    fun->addFnAttr(llvm::Attributes::AlwaysInline);
#else
    fun->addFnAttr(AlwaysInline);
#endif
    return fun;
}

//////////////////////////////////////////////////////////////////////////////////////////

static llvm::FunctionType* DtoVaFunctionType(FuncDeclaration* fdecl)
{
    IrFuncTy &irFty = fdecl->irFty;
    LLFunctionType* type = 0;

    // create new ir funcTy
    irFty.reset();
    irFty.ret = new IrFuncTyArg(Type::tvoid, false);

    irFty.args.push_back(new IrFuncTyArg(Type::tvoid->pointerTo(), false));

    if (fdecl->llvmInternal == LLVMva_start)
        type = GET_INTRINSIC_DECL(vastart)->getFunctionType();
    else if (fdecl->llvmInternal == LLVMva_copy) {
        type = GET_INTRINSIC_DECL(vacopy)->getFunctionType();
        irFty.args.push_back(new IrFuncTyArg(Type::tvoid->pointerTo(), false));
    }
    else if (fdecl->llvmInternal == LLVMva_end)
        type = GET_INTRINSIC_DECL(vaend)->getFunctionType();
    assert(type);

    return type;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl)
{
    // handle for C vararg intrinsics
    if (fdecl->isVaIntrinsic())
        return DtoVaFunctionType(fdecl);

    Type *dthis=0, *dnest=0;

    if (fdecl->ident == Id::ensure || fdecl->ident == Id::require) {
        FuncDeclaration *p = fdecl->parent->isFuncDeclaration();
        assert(p);
        AggregateDeclaration *ad = p->isMember2();
        assert(ad);
        dnest = Type::tvoid->pointerTo();
    } else
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
            llvm_unreachable("needThis, but invalid parent declaration.");
        }
    }
    else if (fdecl->isNested()) {
        dnest = Type::tvoid->pointerTo();
    }

    LLFunctionType* functype = DtoFunctionType(fdecl->type, fdecl->irFty, dthis, dnest, fdecl->isMain(), fdecl->isCtorDeclaration());

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
        if (TemplateDeclaration* tempdecl = tinst->tempdecl->isTemplateDeclaration())
        {
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
            else if (tempdecl->llvmInternal == LLVMinline_ir)
            {
                fdecl->llvmInternal = LLVMinline_ir;
                fdecl->linkage = LINKc;
                fdecl->ir.defined = true;
                Type* type = fdecl->type;
                assert(type->ty == Tfunction);
                static_cast<TypeFunction*>(type)->linkage = LINKc;
            }
        }
    }

    DtoFunctionType(fdecl);

    Logger::println("DtoResolveFunction(%s): %s", fdecl->toPrettyChars(), fdecl->loc.toChars());
    LOG_SCOPE;

    // queue declaration unless the function is abstract without body
    if (!fdecl->isAbstract() || fdecl->fbody)
    {
        DtoDeclareFunction(fdecl);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

#if LDC_LLVM_VER >= 303
static void set_param_attrs(TypeFunction* f, llvm::Function* func, FuncDeclaration* fdecl)
{
    IrFuncTy &irFty = fdecl->irFty;
    llvm::AttributeSet old = func->getAttributes();
    llvm::AttributeSet existingAttrs[] = { old.getFnAttributes(), old.getRetAttributes() };
    llvm::AttributeSet newAttrs = llvm::AttributeSet::get(gIR->context(), existingAttrs);

    int idx = 0;

    // handle implicit args
    #define ADD_PA(X) \
    if (irFty.X) { \
        if (irFty.X->attrs.hasAttributes()) { \
            llvm::AttributeSet a = llvm::AttributeSet::get(gIR->context(), idx, irFty.X->attrs); \
            newAttrs = newAttrs.addAttributes(gIR->context(), idx, a); \
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
    for (size_t k = 0; k < n; k++)
    {
        Parameter* fnarg = Parameter::getNth(f->parameters, k);
        assert(fnarg);

        llvm::AttrBuilder a = irFty.args[k]->attrs;
        if (a.hasAttributes())
        {
            unsigned i = idx + (irFty.reverseParams ? n-k-1 : k);
            llvm::AttributeSet as = llvm::AttributeSet::get(gIR->context(), i, a);
            newAttrs = newAttrs.addAttributes(gIR->context(), i, as);
        }
    }

    // Store the final attribute set
    func->setAttributes(newAttrs);
}
#else
static void set_param_attrs(TypeFunction* f, llvm::Function* func, FuncDeclaration* fdecl)
{
    IrFuncTy &irFty = fdecl->irFty;
    LLSmallVector<llvm::AttributeWithIndex, 9> attrs;

    int idx = 0;

    // handle implicit args
    #define ADD_PA(X) \
    if (irFty.X) { \
        if (HAS_ATTRIBUTES(irFty.X->attrs)) { \
            attrs.push_back(llvm::AttributeWithIndex::get(idx, irFty.X->attrs)); \
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
#if LDC_LLVM_VER == 302
    LLSmallVector<llvm::Attributes, 8> attrptr(n, llvm::Attributes());
#else
    LLSmallVector<llvm::Attributes, 8> attrptr(n, None);
#endif

    for (size_t k = 0; k < n; ++k)
    {
        Parameter* fnarg = Parameter::getNth(f->parameters, k);
        assert(fnarg);

        attrptr[k] = irFty.args[k]->attrs;
    }

    // reverse params?
    if (irFty.reverseParams)
    {
        std::reverse(attrptr.begin(), attrptr.end());
    }

    // build rest of attrs list
    for (size_t i = 0; i < n; i++)
    {
        if (HAS_ATTRIBUTES(attrptr[i]))
        {
            attrs.push_back(llvm::AttributeWithIndex::get(idx + i, attrptr[i]));
        }
    }

    // Merge in any old attributes (attributes for the function itself are
    // also stored in a list slot).
    const size_t newSize = attrs.size();
    llvm::AttrListPtr oldAttrs = func->getAttributes();
    for (size_t i = 0; i < oldAttrs.getNumSlots(); ++i) {
        llvm::AttributeWithIndex curr = oldAttrs.getSlot(i);

        bool found = false;
        for (size_t j = 0; j < newSize; ++j) {
            if (attrs[j].Index == curr.Index) {
#if LDC_LLVM_VER == 302
                attrs[j].Attrs = llvm::Attributes::get(
                    gIR->context(),
                    llvm::AttrBuilder(attrs[j].Attrs).addAttributes(curr.Attrs));
#else
                attrs[j].Attrs |= curr.Attrs;
#endif
                found = true;
                break;
            }
        }

        if (!found) {
            attrs.push_back(curr);
        }
    }

#if LDC_LLVM_VER >= 302
	llvm::AttrListPtr attrlist = llvm::AttrListPtr::get(gIR->context(),
        llvm::ArrayRef<llvm::AttributeWithIndex>(attrs));
#else
    llvm::AttrListPtr attrlist = llvm::AttrListPtr::get(attrs.begin(), attrs.end());
#endif
    func->setAttributes(attrlist);
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareFunction(FuncDeclaration* fdecl)
{
    DtoResolveFunction(fdecl);

    if (fdecl->ir.declared) return;
    fdecl->ir.declared = true;

    Logger::println("DtoDeclareFunction(%s): %s", fdecl->toPrettyChars(), fdecl->loc.toChars());
    LOG_SCOPE;

    if (fdecl->isUnitTestDeclaration() && !global.params.useUnitTests)
    {
        Logger::println("unit tests not enabled");
        return;
    }

    //printf("declare function: %s\n", fdecl->toPrettyChars());

    // intrinsic sanity check
    if (fdecl->llvmInternal == LLVMintrinsic && fdecl->fbody) {
        error(fdecl->loc, "intrinsics cannot have function bodies");
        fatal();
    }

    // get TypeFunction*
    Type* t = fdecl->type->toBasetype();
    TypeFunction* f = static_cast<TypeFunction*>(t);

    if (!fdecl->ir.irFunc) {
        fdecl->ir.irFunc = new IrFunction(fdecl);
    }

    LLFunction* vafunc = 0;
    if (fdecl->isVaIntrinsic())
        vafunc = DtoDeclareVaFunction(fdecl);

    // calling convention
    LINK link = f->linkage;
    if (vafunc || fdecl->llvmInternal == LLVMintrinsic
        // DMD treats _Dmain as having C calling convention and this has been
        // hardcoded into druntime, even if the frontend type has D linkage.
        // See Bugzilla issue 9028.
        || fdecl->isMain()
    )
    {
        link = LINKc;
    }

    // mangled name
    std::string mangledName;
    if (fdecl->llvmInternal == LLVMintrinsic)
        mangledName = fdecl->intrinsicName;
    else
        mangledName = fdecl->mangleExact();
    mangledName = gABI->mangleForLLVM(mangledName, link);

    // construct function
    LLFunctionType* functype = DtoFunctionType(fdecl);
    LLFunction* func = vafunc ? vafunc : gIR->module->getFunction(mangledName);
    if (!func) {
        if(fdecl->llvmInternal == LLVMinline_ir)
        {
            func = DtoInlineIRFunction(fdecl);
        }
        else
        {
            // All function declarations are "external" - any other linkage type
            // is set when actually defining the function.
            func = LLFunction::Create(functype,
                llvm::GlobalValue::ExternalLinkage, mangledName, gIR->module);
        }
    } else if (func->getFunctionType() != functype) {
        error(fdecl->loc, "Function type does not match previously declared function with the same mangled name: %s", fdecl->mangleExact());
        fatal();
    }

    func->setCallingConv(gABI->callingConv(link));

    if (Logger::enabled())
        Logger::cout() << "func = " << *func << std::endl;

    // add func to IRFunc
    fdecl->ir.irFunc->func = func;

    // parameter attributes
    if (!fdecl->isIntrinsic()) {
        set_param_attrs(f, func, fdecl);
        if (global.params.disableRedZone) {
#if LDC_LLVM_VER >= 303
            func->addFnAttr(llvm::Attribute::NoRedZone);
#elif LDC_LLVM_VER == 302
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

    if (fdecl->neverInline)
    {
        fdecl->ir.irFunc->setNeverInline();
    }

    if (fdecl->llvmInternal == LLVMglobal_crt_ctor || fdecl->llvmInternal == LLVMglobal_crt_dtor)
    {
        AppendFunctionToLLVMGlobalCtorsDtors(func, fdecl->priority, fdecl->llvmInternal == LLVMglobal_crt_ctor);
    }

    IrFuncTy &irFty = fdecl->irFty;

    // if (!declareOnly)
    {
        // name parameters
        llvm::Function::arg_iterator iarg = func->arg_begin();

        if (irFty.arg_sret) {
            iarg->setName(".sret_arg");
            fdecl->ir.irFunc->retArg = iarg;
            ++iarg;
        }

        if (irFty.arg_this) {
            iarg->setName(".this_arg");
            fdecl->ir.irFunc->thisArg = iarg;

            VarDeclaration* v = fdecl->vthis;
            if (v) {
                // We already build the this argument here if we will need it
                // later for codegen'ing the function, just as normal
                // parameters below, because it can be referred to in nested
                // context types. Will be given storage in DtoDefineFunction.
                assert(!v->ir.irParam);
                v->ir.irParam = new IrParameter(v, iarg, irFty.arg_this, true);
            }

            ++iarg;
        }
        else if (irFty.arg_nest) {
            iarg->setName(".nest_arg");
            fdecl->ir.irFunc->nestArg = iarg;
            assert(fdecl->ir.irFunc->nestArg);
            ++iarg;
        }

        if (irFty.arg_argptr) {
            iarg->setName("._arguments");
            fdecl->ir.irFunc->_arguments = iarg;
            ++iarg;
            iarg->setName("._argptr");
            fdecl->ir.irFunc->_argptr = iarg;
            ++iarg;
        }

        // we never reference parameters of function prototypes
        unsigned int k = 0;
        for (; iarg != func->arg_end(); ++iarg)
        {
            if (fdecl->parameters && fdecl->parameters->dim > k)
            {
                int paramIndex = irFty.reverseParams ? fdecl->parameters->dim-k-1 : k;
                Dsymbol* argsym = static_cast<Dsymbol*>(fdecl->parameters->data[paramIndex]);

                VarDeclaration* argvd = argsym->isVarDeclaration();
                assert(argvd);
                assert(!argvd->ir.irLocal);
                std::string str(argvd->ident->toChars());
                str.append("_arg");
                iarg->setName(str);
                argvd->ir.irParam = new IrParameter(argvd, iarg, irFty.args[paramIndex]);

                k++;
            }
            else
            {
                iarg->setName("unnamed");
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// FIXME: this isn't too pretty!

void DtoDefineFunction(FuncDeclaration* fd)
{
    if (Logger::enabled())
        Logger::println("DtoDefineFunction(%s): %s", fd->toPrettyChars(), fd->loc.toChars());
    LOG_SCOPE;

    if (fd->ir.defined) return;

    // Skip generating code if this part of a TemplateInstance that is instantiated
    // only by non-root modules (i.e. modules not listed on the command line).
    // See DMD's FuncDeclaration::toObjFile. Check this before calling
    // DtoDeclareFunction DtoDeclareFunction to avoid touching unanalyzed code.
    TemplateInstance *ti = fd->inTemplateInstance();
    if (!global.params.useUnitTests &&
        ti && ti->instantiatingModule && !ti->instantiatingModule->isRoot())
    {
        Module *mi = ti->instantiatingModule;

        // If mi imports any root modules, we still need to generate the code.
        for (size_t i = 0; i < Module::amodules.dim; ++i)
        {
            Module *m = Module::amodules[i];
            m->insearch = 0;
        }
        bool importsRoot = false;
        for (size_t i = 0; i < Module::amodules.dim; ++i)
        {
            Module *m = Module::amodules[i];
            if (m->isRoot() && mi->imports(m))
            {
                importsRoot = true;
                break;
            }
        }
        for (size_t i = 0; i < Module::amodules.dim; ++i)
        {
            Module *m = Module::amodules[i];
            m->insearch = 0;
        }
        if (!importsRoot)
        {
            IF_LOG Logger::println("Ignoring; already instantiated in %s (%s)", ti->instantiatingModule->toChars(), ti->toChars());
            fd->ir.defined = true;
            return;
        }
    }

    DtoDeclareFunction(fd);
    assert(fd->ir.declared);

    // DtoResolveFunction might also set the defined flag for functions we
    // should not touch.
    if (fd->ir.defined) return;
    fd->ir.defined = true;

    // We cannot emit nested functions with parents that have not gone through
    // semantic analysis. This can happen as DMD leaks some template instances
    // from constraints into the module member list. DMD gets away with being
    // sloppy as functions in template contraints obviously never need to access
    // data from the template function itself, but it would still mess up our
    // nested context creation code.
    FuncDeclaration* parent = fd;
    while ((parent = getParentFunc(parent, true)))
    {
        if (parent->semanticRun != PASSsemantic3done)
        {
            Logger::println("Ignoring nested function with unanalyzed parent.");
            return;
        }
    }

    if (fd->isUnitTestDeclaration()) {
        if (global.params.useUnitTests)
            gIR->unitTests.push_back(fd);
        else
            return;
    } else if (fd->isSharedStaticCtorDeclaration()) {
        gIR->sharedCtors.push_back(fd);
    } else if (StaticDtorDeclaration *dtorDecl = fd->isSharedStaticDtorDeclaration()) {
        gIR->sharedDtors.push_front(fd);
        if (dtorDecl->vgate)
            gIR->sharedGates.push_front(dtorDecl->vgate);
    } else if (fd->isStaticCtorDeclaration()) {
        gIR->ctors.push_back(fd);
    } else if (StaticDtorDeclaration *dtorDecl = fd->isStaticDtorDeclaration()) {
        gIR->dtors.push_front(fd);
        if (dtorDecl->vgate)
            gIR->gates.push_front(dtorDecl->vgate);
    }


    // if this function is naked, we take over right away! no standard processing!
    if (fd->naked)
    {
        DtoDefineNakedFunction(fd);
        return;
    }

    IrFuncTy &irFty = fd->irFty;

    // debug info
    fd->ir.irFunc->diSubprogram = gIR->DBuilder.EmitSubProgram(fd);

    Type* t = fd->type->toBasetype();
    TypeFunction* f = static_cast<TypeFunction*>(t);
    // assert(f->irtype);

    llvm::Function* func = fd->ir.irFunc->func;

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

    func->setLinkage(DtoLinkage(fd));

    // On x86_64, always set 'uwtable' for System V ABI compatibility.
    // TODO: Find a better place for this.
    if (global.params.targetTriple.getArch() == llvm::Triple::x86_64)
    {
#if LDC_LLVM_VER >= 303
        func->addFnAttr(llvm::Attribute::UWTable);
#elif LDC_LLVM_VER == 302
        func->addFnAttr(llvm::Attributes::UWTable);
#else
        func->addFnAttr(UWTable);
#endif
    }

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
    gIR->DBuilder.EmitFuncStart(fd);

    // this hack makes sure the frame pointer elimination optimization is disabled.
    // this this eliminates a bunch of inline asm related issues.
    if (fd->hasReturnExp & 8) // has inline asm
    {
        // emit a call to llvm_eh_unwind_init
        LLFunction* hack = GET_INTRINSIC_DECL(eh_unwind_init);
        gIR->ir->CreateCall(hack, "");
    }

    // give the 'this' argument storage and debug info
    if (irFty.arg_this)
    {
        LLValue* thisvar = irfunction->thisArg;
        assert(thisvar);

        LLValue* thismem = thisvar;
        if (!irFty.arg_this->byref)
        {
            thismem = DtoRawAlloca(thisvar->getType(), 0, "this"); // FIXME: align?
            DtoStore(thisvar, thismem);
            irfunction->thisArg = thismem;
        }

        assert(fd->vthis->ir.irParam->value == thisvar);
        fd->vthis->ir.irParam->value = thismem;

        gIR->DBuilder.EmitLocalVariable(thismem, fd->vthis);
    }

    // give the 'nestArg' storage
    if (irFty.arg_nest)
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
        size_t n = irFty.args.size();
        assert(n == fd->parameters->dim);
        for (size_t i=0; i < n; ++i)
        {
            Dsymbol* argsym = static_cast<Dsymbol*>(fd->parameters->data[i]);
            VarDeclaration* vd = argsym->isVarDeclaration();
            assert(vd);

            IrParameter* irparam = vd->ir.irParam;
            assert(irparam);

            bool refout = vd->storage_class & (STCref | STCout);
            bool lazy = vd->storage_class & STClazy;
            if (!refout && (!irparam->arg->byref || lazy))
            {
                // alloca a stack slot for this first class value arg
                LLType* argt;
                if (lazy)
                    argt = irparam->value->getType();
                else
                    argt = i1ToI8(DtoType(vd->type));
                LLValue* mem = DtoRawAlloca(argt, 0, vd->ident->toChars());

                // let the abi transform the argument back first
                DImValue arg_dval(vd->type, irparam->value);
                irFty.getParam(vd->type, i, &arg_dval, mem);

                // set the arg var value to the alloca
                irparam->value = mem;
            }

            if (global.params.symdebug && !(isaArgument(irparam->value) && isaArgument(irparam->value)->hasByValAttr()) && !refout)
                gIR->DBuilder.EmitLocalVariable(irparam->value, vd);
        }
    }

    FuncGen fg;
    irfunction->gen = &fg;

    DtoCreateNestedContext(fd);

    if (fd->vresult && !
        fd->vresult->nestedrefs.dim // FIXME: not sure here :/
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
        gIR->DBuilder.EmitFuncEnd(fd);
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

    DtoResolveFunction(f);
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
