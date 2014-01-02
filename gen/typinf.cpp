//===-- typinf.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file mostly consists of code under the BSD-style LDC license, but some
// parts have been derived from DMD as noted below. See the LICENSE file for
// details.
//
//===----------------------------------------------------------------------===//

// Copyright (c) 1999-2004 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

// Modifications for LDC:
// Copyright (c) 2007 by Tomas Lindquist Olsen
// tomas at famolsen dk

#include "aggregate.h"
#include "attrib.h"
#include "declaration.h"
#include "enum.h"
#include "expression.h"
#include "id.h"
#include "import.h"
#include "init.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "scope.h"
#include "template.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/metadata.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "ir/irtype.h"
#include "ir/irvar.h"
#include <cassert>
#include <cstdio>
#include <ir/irtypeclass.h>

/*******************************************
 * Get a canonicalized form of the TypeInfo for use with the internal
 * runtime library routines. Canonicalized in that static arrays are
 * represented as dynamic arrays, enums are represented by their
 * underlying type, etc. This reduces the number of TypeInfo's needed,
 * so we can use the custom internal ones more.
 */

Expression *Type::getInternalTypeInfo(Scope *sc)
{   TypeInfoDeclaration *tid;
    Expression *e;
    Type *t;
    static TypeInfoDeclaration *internalTI[TMAX];

    //printf("Type::getInternalTypeInfo() %s\n", toChars());
    t = toBasetype();
    switch (t->ty)
    {
    case Tsarray:
#if 0
        // convert to corresponding dynamic array type
        t = t->nextOf()->mutableOf()->arrayOf();
#endif
        break;

    case Tclass:
        if (((TypeClass *)t)->sym->isInterfaceDeclaration())
            break;
        goto Linternal;

    case Tarray:
        // convert to corresponding dynamic array type
        t = t->nextOf()->mutableOf()->arrayOf();
        if (t->nextOf()->ty != Tclass)
            break;
        goto Linternal;

    case Tfunction:
    case Tdelegate:
    case Tpointer:
    Linternal:
        tid = internalTI[t->ty];
        if (!tid)
        {   tid = new TypeInfoDeclaration(t, 1);
            internalTI[t->ty] = tid;
        }
        e = new VarExp(Loc(), tid);
        e = e->addressOf(sc);
        e->type = tid->type;        // do this so we don't get redundant dereference
        return e;

    default:
        break;
    }
    //printf("\tcalling getTypeInfo() %s\n", t->toChars());
    return t->getTypeInfo(sc);
}


bool inNonRoot(Dsymbol *s);
FuncDeclaration *search_toHash(StructDeclaration *sd);
FuncDeclaration *search_toString(StructDeclaration *sd);

/****************************************************
 * Get the exact TypeInfo.
 */

Expression *Type::getTypeInfo(Scope *sc)
{
    IF_LOG Logger::println("Type::getTypeInfo(): %s", toChars());
    LOG_SCOPE

    if (!Type::dtypeinfo)
    {
        error(Loc(), "TypeInfo not found. object.d may be incorrectly installed or corrupt, compile with -v switch");
        fatal();
    }

    Type *t = merge2(); // do this since not all Type's are merge'd
    if (!t->vtinfo)
    {
        if (t->isShared())      // does both 'shared' and 'shared const'
            t->vtinfo = new TypeInfoSharedDeclaration(t);
        else if (t->isConst())
            t->vtinfo = new TypeInfoConstDeclaration(t);
        else if (t->isImmutable())
            t->vtinfo = new TypeInfoInvariantDeclaration(t);
        else if (t->isWild())
            t->vtinfo = new TypeInfoWildDeclaration(t);
        else
            t->vtinfo = t->getTypeInfoDeclaration();
        assert(t->vtinfo);
        vtinfo = t->vtinfo;

        /* If this has a custom implementation in std/typeinfo, then
         * do not generate a COMDAT for it.
         */
        if (!t->builtinTypeInfo())
        {   // Generate COMDAT
            if (sc)                     // if in semantic() pass
            {
                // Find module that will go all the way to an object file
                Module *m = sc->module->importedFrom;
                m->members->push(t->vtinfo);

                if (ty == Tstruct)
                {
                    StructDeclaration *sd = ((TypeStruct *)this)->sym;
                    if ((sd->xeq && sd->xeq != sd->xerreq ||
                         sd->xcmp && sd->xcmp != sd->xerrcmp ||
                         search_toHash(sd) ||
                         search_toString(sd)
                        ) && inNonRoot(sd))
                    {
                        Module::addDeferredSemantic3(sd);
                    }
                }
            }
            else                        // if in obj generation pass
            {
                t->vtinfo->codegen(gIR);
            }
        }
    }
    if (!vtinfo)
        vtinfo = t->vtinfo;     // Types aren't merged, but we can share the vtinfo's
    Expression *e = new VarExp(Loc(), t->vtinfo);
    e = e->addressOf(sc);
    e->type = t->vtinfo->type;          // do this so we don't get redundant dereference
    return e;
}

enum RET TypeFunction::retStyle()
{
    return RETstack;
}

TypeInfoDeclaration *Type::getTypeInfoDeclaration()
{
    //printf("Type::getTypeInfoDeclaration() %s\n", toChars());
    return new TypeInfoDeclaration(this, 0);
}

TypeInfoDeclaration *TypeTypedef::getTypeInfoDeclaration()
{
    return new TypeInfoTypedefDeclaration(this);
}

TypeInfoDeclaration *TypePointer::getTypeInfoDeclaration()
{
    return new TypeInfoPointerDeclaration(this);
}

TypeInfoDeclaration *TypeDArray::getTypeInfoDeclaration()
{
    return new TypeInfoArrayDeclaration(this);
}

TypeInfoDeclaration *TypeSArray::getTypeInfoDeclaration()
{
    return new TypeInfoStaticArrayDeclaration(this);
}

TypeInfoDeclaration *TypeAArray::getTypeInfoDeclaration()
{
    return new TypeInfoAssociativeArrayDeclaration(this);
}

TypeInfoDeclaration *TypeStruct::getTypeInfoDeclaration()
{
    return new TypeInfoStructDeclaration(this);
}

TypeInfoDeclaration *TypeClass::getTypeInfoDeclaration()
{
    if (sym->isInterfaceDeclaration())
        return new TypeInfoInterfaceDeclaration(this);
    else
        return new TypeInfoClassDeclaration(this);
}

TypeInfoDeclaration *TypeVector::getTypeInfoDeclaration()
{
    return new TypeInfoVectorDeclaration(this);
}

TypeInfoDeclaration *TypeEnum::getTypeInfoDeclaration()
{
    return new TypeInfoEnumDeclaration(this);
}

TypeInfoDeclaration *TypeFunction::getTypeInfoDeclaration()
{
    return new TypeInfoFunctionDeclaration(this);
}

TypeInfoDeclaration *TypeDelegate::getTypeInfoDeclaration()
{
    return new TypeInfoDelegateDeclaration(this);
}

TypeInfoDeclaration *TypeTuple::getTypeInfoDeclaration()
{
    return new TypeInfoTupleDeclaration(this);
}

/* ========================================================================= */

/* These decide if there's an instance for them already in std.typeinfo,
 * because then the compiler doesn't need to build one.
 */

int Type::builtinTypeInfo()
{
    return 0;
}

int TypeBasic::builtinTypeInfo()
{
    return mod ? 0 : 1;
}

int TypeDArray::builtinTypeInfo()
{
    return !mod && ((next->isTypeBasic() != NULL && !next->mod) ||
        // strings are so common, make them builtin
        (next->ty == Tchar && next->mod == MODimmutable) ||
        (next->ty == Tchar && next->mod == MODconst));
}

int TypeClass::builtinTypeInfo()
{
    /* This is statically put out with the ClassInfo, so
     * claim it is built in so it isn't regenerated by each module.
     */
#if IN_DMD
    return mod ? 0 : 1;
#elif IN_LLVM
    // FIXME if I enable this, the way LDC does typeinfo will cause a bunch
    // of linker errors to missing class typeinfo definitions.
    return 0;
#endif
}

/* ========================================================================= */

//////////////////////////////////////////////////////////////////////////////
//                             MAGIC   PLACE
//                                (wut?)
//////////////////////////////////////////////////////////////////////////////

static void emitTypeMetadata(TypeInfoDeclaration *tid)
{
    // We don't want to generate metadata for non-concrete types (such as tuple
    // types, slice types, typeof(expr), etc.), void and function types (without
    // an indirection), as there must be a valid LLVM undef value of that type.
    // As those types cannot appear as LLVM values, they are not interesting for
    // the optimizer passes anyway.
    Type* t = tid->tinfo->toBasetype();
    if (t->ty < Terror && t->ty != Tvoid && t->ty != Tfunction && t->ty != Tident) {
        // Add some metadata for use by optimization passes.
        std::string metaname(TD_PREFIX);
        metaname += tid->mangle();
        llvm::NamedMDNode* meta = gIR->module->getNamedMetadata(metaname);

        if (!meta) {
            // Construct the fields
            MDNodeField* mdVals[TD_NumFields];
            mdVals[TD_TypeInfo] = llvm::cast<MDNodeField>(tid->ir.irGlobal->value);
            mdVals[TD_Type] = llvm::UndefValue::get(DtoType(tid->tinfo));

            // Construct the metadata and insert it into the module.
            llvm::NamedMDNode* node = gIR->module->getOrInsertNamedMetadata(metaname);
            node->addOperand(llvm::MDNode::get(gIR->context(),
                llvm::makeArrayRef(mdVals, TD_NumFields)));
        }
    }
}

void DtoResolveTypeInfo(TypeInfoDeclaration* tid)
{
    if (tid->ir.resolved) return;
    tid->ir.resolved = true;

    // TypeInfo instances (except ClassInfo ones) are always emitted as weak
    // symbols when they are used.
    tid->codegen(gIR);
}

void TypeInfoDeclaration::codegen(IRState* p)
{
    Logger::println("TypeInfoDeclaration::codegen(%s)", toPrettyChars());
    LOG_SCOPE;

    if (ir.defined) return;
    ir.defined = true;

    std::string mangled(mangle());
    if (Logger::enabled())
    {
        Logger::println("type = '%s'", tinfo->toChars());
        Logger::println("typeinfo mangle: %s", mangled.c_str());
    }

    IrGlobal* irg = new IrGlobal(this);
    ir.irGlobal = irg;
    irg->value = gIR->module->getGlobalVariable(mangled);
    if (irg->value) {
        irg->type = irg->value->getType()->getContainedType(0);
        assert(irg->type->isStructTy());
    } else {
        if (tinfo->builtinTypeInfo()) // this is a declaration of a builtin __initZ var
            irg->type = Type::dtypeinfo->type->irtype->isClass()->getMemoryLLType();
        else
            irg->type = LLStructType::create(gIR->context(), toPrettyChars());
        irg->value = new llvm::GlobalVariable(*gIR->module, irg->type, true,
            llvm::GlobalValue::ExternalLinkage, NULL, mangled);
    }

    emitTypeMetadata(this);

    // this is a declaration of a builtin __initZ var
    if (tinfo->builtinTypeInfo()) {
        LLGlobalVariable* g = isaGlobalVar(irg->value);
        g->setLinkage(llvm::GlobalValue::ExternalLinkage);
        return;
    }

    // define custom typedef
    llvmDefine();
}

/* ========================================================================= */

void TypeInfoDeclaration::llvmDefine()
{
    Logger::println("TypeInfoDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::dtypeinfo);
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoTypedefDeclaration::llvmDefine()
{
    Logger::println("TypeInfoTypedefDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfotypedef);

    assert(tinfo->ty == Ttypedef);
    TypeTypedef *tc = static_cast<TypeTypedef *>(tinfo);
    TypedefDeclaration *sd = tc->sym;

    // TypeInfo base
    sd->basetype = sd->basetype->merge(); // dmd does it ... why?
    b.push_typeinfo(sd->basetype);

    // char[] name
    b.push_string(sd->toPrettyChars());

    // void[] init
    // emit null array if we should use the basetype, or if the basetype
    // uses default initialization.
    if (tinfo->isZeroInit(Loc()) || !sd->init)
    {
        b.push_null_void_array();
    }
    // otherwise emit a void[] with the default initializer
    else
    {
        LLConstant* C = DtoConstInitializer(sd->loc, sd->basetype, sd->init);
        b.push_void_array(C, sd->basetype, sd);
    }

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoEnumDeclaration::llvmDefine()
{
    Logger::println("TypeInfoEnumDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfoenum);

    assert(tinfo->ty == Tenum);
    TypeEnum *tc = static_cast<TypeEnum *>(tinfo);
    EnumDeclaration *sd = tc->sym;

    // TypeInfo base
    b.push_typeinfo(sd->memtype);

    // char[] name
    b.push_string(sd->toPrettyChars());

    // void[] init
    // emit void[] with the default initialier, the array is null if the default
    // initializer is zero
    if (!sd->members || tinfo->isZeroInit(loc))
    {
        b.push_null_void_array();
    }
    // otherwise emit a void[] with the default initializer
    else
    {
        Type *memtype = sd->memtype;
        LLType *memty = DtoType(memtype);
        LLConstant *C;
        Expression *defaultval = sd->getDefaultValue(loc);
        if (memtype->isintegral())
            C = LLConstantInt::get(memty, defaultval->toInteger(), !isLLVMUnsigned(memtype));
        else if (memtype->isString())
            C = DtoConstString(static_cast<const char *>(defaultval->toString()->string));
        else
            llvm_unreachable("Unsupported type");

        b.push_void_array(C, memtype, sd);
    }

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoPointerDeclaration::llvmDefine()
{
    Logger::println("TypeInfoPointerDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfopointer);
    // TypeInfo base
    b.push_typeinfo(tinfo->nextOf());
    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoArrayDeclaration::llvmDefine()
{
    Logger::println("TypeInfoArrayDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfoarray);
    // TypeInfo base
    b.push_typeinfo(tinfo->nextOf());
    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoStaticArrayDeclaration::llvmDefine()
{
    Logger::println("TypeInfoStaticArrayDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Tsarray);
    TypeSArray *tc = static_cast<TypeSArray *>(tinfo);

    RTTIBuilder b(Type::typeinfostaticarray);

    // value typeinfo
    b.push_typeinfo(tc->nextOf());

    // length
    b.push(DtoConstSize_t(static_cast<size_t>(tc->dim->toUInteger())));

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoAssociativeArrayDeclaration::llvmDefine()
{
    Logger::println("TypeInfoAssociativeArrayDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Taarray);
    TypeAArray *tc = static_cast<TypeAArray *>(tinfo);

    RTTIBuilder b(Type::typeinfoassociativearray);

    // value typeinfo
    b.push_typeinfo(tc->nextOf());

    // key typeinfo
    b.push_typeinfo(tc->index);

    // impl typeinfo
    b.push_typeinfo(tc->getImpl()->type);

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoFunctionDeclaration::llvmDefine()
{
    Logger::println("TypeInfoFunctionDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfofunction);
    // TypeInfo base
    b.push_typeinfo(tinfo->nextOf());
    // string deco
    b.push_string(tinfo->deco);
    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoDelegateDeclaration::llvmDefine()
{
    Logger::println("TypeInfoDelegateDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Tdelegate);
    Type* ret_type = tinfo->nextOf()->nextOf();

    RTTIBuilder b(Type::typeinfodelegate);
    // TypeInfo base
    b.push_typeinfo(ret_type);
    // string deco
    b.push_string(tinfo->deco);
    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoStructDeclaration::llvmDefine()
{
    Logger::println("TypeInfoStructDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // make sure struct is resolved
    assert(tinfo->ty == Tstruct);
    TypeStruct *tc = static_cast<TypeStruct *>(tinfo);
    StructDeclaration *sd = tc->sym;

    // handle opaque structs
    if (!sd->members) {
        RTTIBuilder b(Type::typeinfostruct);
        b.finalize(ir.irGlobal);
        return;
    }

    // can't emit typeinfo for forward declarations
    if (sd->sizeok != SIZEOKdone)
    {
        sd->error("cannot emit TypeInfo for forward declaration");
        fatal();
    }

    DtoResolveStruct(sd);
    IrAggr* iraggr = sd->ir.irAggr;

    RTTIBuilder b(Type::typeinfostruct);

    // char[] name
    b.push_string(sd->toPrettyChars());

    // void[] init
    // The protocol is to write a null pointer for zero-initialized arrays. The
    // length field is always needed for tsize().
    llvm::Constant *initPtr;
    if (tc->isZeroInit(Loc()))
        initPtr = getNullValue(getVoidPtrType());
    else
        initPtr = iraggr->getInitSymbol();
    b.push_void_array(getTypeStoreSize(DtoType(tc)), initPtr);

    // well use this module for all overload lookups

    // toHash
    FuncDeclaration* fd = search_toHash(sd);
    b.push_funcptr(fd);

    // opEquals
    fd = sd->xeq;
    b.push_funcptr(fd);

    // opCmp
    fd = sd->xcmp;
    b.push_funcptr(fd);

    // toString
    fd = search_toString(sd);
    b.push_funcptr(fd);

    // uint m_flags;
    unsigned hasptrs = tc->hasPointers() ? 1 : 0;
    b.push_uint(hasptrs);

    ClassDeclaration* tscd = Type::typeinfostruct;

    // On x86_64, class TypeInfo_Struct contains 2 additional fields
    // (m_arg1/m_arg2) which are used for the X86_64 System V ABI varargs 
    // implementation. They are not present on any other cpu/os.
    assert((global.params.targetTriple.getArch() != llvm::Triple::x86_64 && tscd->fields.dim == 11) ||
           (global.params.targetTriple.getArch() == llvm::Triple::x86_64 && tscd->fields.dim == 13));

    //void function(void*)                    xdtor;
    b.push_funcptr(sd->dtor);

    //void function(void*)                    xpostblit;
    FuncDeclaration *xpostblit = sd->postblit;
    if (xpostblit && sd->postblit->storage_class & STCdisable)
        xpostblit = 0;
    b.push_funcptr(xpostblit);

    //uint m_align;
    b.push_uint(tc->alignsize());

    if (global.params.is64bit)
    {
        // TypeInfo m_arg1;
        // TypeInfo m_arg2;
        TypeTuple *tup = tc->toArgTypes();
        assert(tup->arguments->dim <= 2);
        for (unsigned i = 0; i < 2; i++)
        {
            if (i < tup->arguments->dim)
            {
                Type *targ = static_cast<Parameter *>(tup->arguments->data[i])->type;
                targ = targ->merge();
                b.push_typeinfo(targ);
            }
            else
                b.push_null(Type::dtypeinfo->type);
        }
    }

    // immutable(void)* m_RTInfo;
    // The cases where getRTInfo is null are not quite here, but the code is
    // modelled after what DMD does.
    if (sd->getRTInfo)
        b.push(sd->getRTInfo->toConstElem(gIR));
    else if (!tc->hasPointers())
        b.push_size_as_vp(0);       // no pointers
    else
        b.push_size_as_vp(1);       // has pointers

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoClassDeclaration::codegen(IRState *p)
{
    // For classes, the TypeInfo is in fact a ClassInfo instance and emitted
    // as a __ClassZ symbol. For interfaces, the __InterfaceZ symbol is
    // referenced as "info" member in a (normal) TypeInfo_Interface instance.
    IrGlobal *irg = new IrGlobal(this);
    ir.irGlobal = irg;

    assert(tinfo->ty == Tclass);
    TypeClass *tc = static_cast<TypeClass *>(tinfo);
    DtoResolveClass(tc->sym);

    irg->value = tc->sym->ir.irAggr->getClassInfoSymbol();
    irg->type = irg->value->getType()->getContainedType(0);

    if (!tc->sym->isInterfaceDeclaration())
    {
        emitTypeMetadata(this);
    }
}

void TypeInfoClassDeclaration::llvmDefine()
{
    llvm_unreachable("TypeInfoClassDeclaration::llvmDefine() should not be called, "
        "as a custom Dsymbol::codegen() override is used");
}

/* ========================================================================= */

void TypeInfoInterfaceDeclaration::llvmDefine()
{
    Logger::println("TypeInfoInterfaceDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // make sure interface is resolved
    assert(tinfo->ty == Tclass);
    TypeClass *tc = static_cast<TypeClass *>(tinfo);
    DtoResolveClass(tc->sym);

    RTTIBuilder b(Type::typeinfointerface);

    // TypeInfo base
    b.push_classinfo(tc->sym);

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoTupleDeclaration::llvmDefine()
{
    Logger::println("TypeInfoTupleDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // create elements array
    assert(tinfo->ty == Ttuple);
    TypeTuple *tu = static_cast<TypeTuple *>(tinfo);

    size_t dim = tu->arguments->dim;
    std::vector<LLConstant*> arrInits;
    arrInits.reserve(dim);

    LLType* tiTy = DtoType(Type::dtypeinfo->type);

    for (size_t i = 0; i < dim; i++)
    {
        Parameter *arg = static_cast<Parameter *>(tu->arguments->data[i]);
        arrInits.push_back(DtoTypeInfoOf(arg->type, true));
    }

    // build array
    LLArrayType* arrTy = LLArrayType::get(tiTy, dim);
    LLConstant* arrC = LLConstantArray::get(arrTy, arrInits);

    RTTIBuilder b(Type::typeinfotypelist);

    // push TypeInfo[]
    b.push_array(arrC, dim, Type::dtypeinfo->type, NULL);

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoConstDeclaration::llvmDefine()
{
    Logger::println("TypeInfoConstDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfoconst);
    // TypeInfo base
    b.push_typeinfo(tinfo->mutableOf()->merge());
    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoInvariantDeclaration::llvmDefine()
{
    Logger::println("TypeInfoInvariantDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfoinvariant);
    // TypeInfo base
    b.push_typeinfo(tinfo->mutableOf()->merge());
    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoSharedDeclaration::llvmDefine()
{
    Logger::println("TypeInfoSharedDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfoshared);
    // TypeInfo base
    b.push_typeinfo(tinfo->unSharedOf()->merge());
    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoWildDeclaration::llvmDefine()
{
    Logger::println("TypeInfoWildDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfowild);
    // TypeInfo base
    b.push_typeinfo(tinfo->mutableOf()->merge());
    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoVectorDeclaration::llvmDefine()
{
    Logger::println("TypeInfoVectorDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Tvector);
    TypeVector *tv = static_cast<TypeVector *>(tinfo);

    RTTIBuilder b(Type::typeinfovector);
    // TypeInfo base
    b.push_typeinfo(tv->basetype);
    // finish
    b.finalize(ir.irGlobal);
}
