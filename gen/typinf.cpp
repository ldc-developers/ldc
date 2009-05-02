

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

#include <cstdio>
#include <cassert>

#include "gen/llvm.h"

#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "scope.h"
#include "init.h"
#include "expression.h"
#include "attrib.h"
#include "declaration.h"
#include "template.h"
#include "id.h"
#include "enum.h"
#include "import.h"
#include "aggregate.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/arrays.h"
#include "gen/structs.h"
#include "gen/classes.h"
#include "gen/linkage.h"
#include "gen/metadata.h"

#include "ir/irvar.h"

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
    #if DMDV2
        // convert to corresponding dynamic array type
        t = t->nextOf()->mutableOf()->arrayOf();
    #endif
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
        e = new VarExp(0, tid);
        e = e->addressOf(sc);
        e->type = tid->type;    // do this so we don't get redundant dereference
        return e;

    default:
        break;
    }
    //printf("\tcalling getTypeInfo() %s\n", t->toChars());
    return t->getTypeInfo(sc);
}

/****************************************************
 * Get the exact TypeInfo.
 */

Expression *Type::getTypeInfo(Scope *sc)
{
    Expression *e;
    Type *t;

    //printf("Type::getTypeInfo() %p, %s\n", this, toChars());
    t = merge();    // do this since not all Type's are merge'd
    if (!t->vtinfo)
    {
#if DMDV2
    if (t->isConst())
        t->vtinfo = new TypeInfoConstDeclaration(t);
    else if (t->isInvariant())
        t->vtinfo = new TypeInfoInvariantDeclaration(t);
    else
#endif
        t->vtinfo = t->getTypeInfoDeclaration();
    assert(t->vtinfo);

    /* If this has a custom implementation in std/typeinfo, then
     * do not generate a COMDAT for it.
     */
    if (!t->builtinTypeInfo())
    {   // Generate COMDAT
        if (sc)         // if in semantic() pass
        {   // Find module that will go all the way to an object file
        Module *m = sc->module->importedFrom;
        m->members->push(t->vtinfo);
        }
        else            // if in obj generation pass
        {
#if IN_DMD
        t->vtinfo->toObjFile(0); // TODO: multiobj
#else
        t->vtinfo->codegen(sir);
#endif
        }
    }
    }
    e = new VarExp(0, t->vtinfo);
    e = e->addressOf(sc);
    e->type = t->vtinfo->type;      // do this so we don't get redundant dereference
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
#if DMDV2
    return !mod;
#else
    return 1;
#endif
}

int TypeDArray::builtinTypeInfo()
{
#if DMDV2
    return !mod && next->isTypeBasic() != NULL && !next->mod;
#else
    return next->isTypeBasic() != NULL;
#endif
}

/* ========================================================================= */

//////////////////////////////////////////////////////////////////////////////
//                             MAGIC   PLACE
//////////////////////////////////////////////////////////////////////////////

void DtoResolveTypeInfo(TypeInfoDeclaration* tid);
void DtoDeclareTypeInfo(TypeInfoDeclaration* tid);
void DtoConstInitTypeInfo(TypeInfoDeclaration* tid);

void TypeInfoDeclaration::codegen(Ir*)
{
    DtoResolveTypeInfo(this);
}

void DtoResolveTypeInfo(TypeInfoDeclaration* tid)
{
    if (tid->ir.resolved) return;
    tid->ir.resolved = true;

    Logger::println("DtoResolveTypeInfo(%s)", tid->toChars());
    LOG_SCOPE;

    IrGlobal* irg = new IrGlobal(tid);

    std::string mangle(tid->mangle());

    irg->value = gIR->module->getGlobalVariable(mangle);
    if (!irg->value)
        irg->value = new llvm::GlobalVariable(irg->type.get(), true,
        TYPEINFO_LINKAGE_TYPE, NULL, mangle, gIR->module);

    tid->ir.irGlobal = irg;

#ifdef USE_METADATA
    // Add some metadata for use by optimization passes.
    static std::string prefix = "llvm.ldc.typeinfo.";
    std::string metaname = prefix + mangle;
    LLGlobalVariable* meta = gIR->module->getGlobalVariable(metaname);
    // Don't generate metadata for non-concrete types
    // (such as tuple types, slice types, typeof(expr), etc.)
    if (!meta && tid->tinfo->toBasetype()->ty < Terror) {
        LLConstant* mdVals[] = {
            llvm::cast<LLConstant>(irg->value),
            llvm::UndefValue::get(DtoType(tid->tinfo))
        };
        llvm::MDNode* metadata =
            llvm::MDNode::get(mdVals, sizeof(mdVals) / sizeof(mdVals[0]));
        new llvm::GlobalVariable(metadata->getType(), true,
            METADATA_LINKAGE_TYPE, metadata, metaname, gIR->module);
    }
#endif

    DtoDeclareTypeInfo(tid);
}

void DtoDeclareTypeInfo(TypeInfoDeclaration* tid)
{
    DtoResolveTypeInfo(tid);

    if (tid->ir.declared) return;
    tid->ir.declared = true;

    Logger::println("DtoDeclareTypeInfo(%s)", tid->toChars());
    LOG_SCOPE;

    IrGlobal* irg = tid->ir.irGlobal;

    std::string mangled(tid->mangle());

    Logger::println("type = '%s'", tid->tinfo->toChars());
    Logger::println("typeinfo mangle: %s", mangled.c_str());

    assert(irg->value != NULL);

    // this is a declaration of a builtin __initZ var
    if (tid->tinfo->builtinTypeInfo()) {
        // fixup the global
        const llvm::Type* rty = Type::typeinfo->type->ir.type->get();
        llvm::cast<llvm::OpaqueType>(irg->type.get())->refineAbstractTypeTo(rty);
        LLGlobalVariable* g = isaGlobalVar(irg->value);
        g->setLinkage(llvm::GlobalValue::ExternalLinkage);
        return;
    }

    // custom typedef
    DtoConstInitTypeInfo(tid);
}

void DtoConstInitTypeInfo(TypeInfoDeclaration* tid)
{
    if (tid->ir.initialized) return;
    tid->ir.initialized = true;

    Logger::println("DtoConstInitTypeInfo(%s)", tid->toChars());
    LOG_SCOPE;

    tid->llvmDefine();
}

/* ========================================================================= */

void TypeInfoDeclaration::llvmDefine()
{
    assert(0 && "TypeInfoDeclaration::llvmDeclare");
}

/* ========================================================================= */

void TypeInfoTypedefDeclaration::llvmDefine()
{
    Logger::println("TypeInfoTypedefDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    ClassDeclaration* base = Type::typeinfotypedef;
    base->codegen(Type::sir);

    // vtbl
    std::vector<LLConstant*> sinits;
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(getNullPtr(getPtrToType(LLType::Int8Ty)));

    assert(tinfo->ty == Ttypedef);
    TypeTypedef *tc = (TypeTypedef *)tinfo;
    TypedefDeclaration *sd = tc->sym;

    // TypeInfo base
    sd->basetype = sd->basetype->merge(); // DMD does this!
    LLConstant* castbase = DtoTypeInfoOf(sd->basetype, true);
    sinits.push_back(castbase);

    // char[] name
    char *name = sd->toPrettyChars();
    sinits.push_back(DtoConstString(name));

    // void[] init
    const LLPointerType* initpt = getPtrToType(LLType::Int8Ty);
    if (tinfo->isZeroInit() || !sd->init) // 0 initializer, or the same as the base type
    {
        sinits.push_back(DtoConstSlice(DtoConstSize_t(0), getNullPtr(initpt)));
    }
    else
    {
        LLConstant* ci = DtoConstInitializer(sd->loc, sd->basetype, sd->init);
        std::string ciname(sd->mangle());
        ciname.append("__init");
        llvm::GlobalVariable* civar = new llvm::GlobalVariable(DtoType(sd->basetype),true,llvm::GlobalValue::InternalLinkage,ci,ciname,gIR->module);
        LLConstant* cicast = llvm::ConstantExpr::getBitCast(civar, initpt);
        size_t cisize = getTypeStoreSize(DtoType(sd->basetype));
        sinits.push_back(DtoConstSlice(DtoConstSize_t(cisize), cicast));
    }

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

void TypeInfoEnumDeclaration::llvmDefine()
{
    Logger::println("TypeInfoEnumDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    ClassDeclaration* base = Type::typeinfoenum;
    base->codegen(Type::sir);

    // vtbl
    std::vector<LLConstant*> sinits;
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    assert(tinfo->ty == Tenum);
    TypeEnum *tc = (TypeEnum *)tinfo;
    EnumDeclaration *sd = tc->sym;

    // TypeInfo base
    LLConstant* castbase = DtoTypeInfoOf(sd->memtype, true);
    sinits.push_back(castbase);

    // char[] name
    char *name = sd->toPrettyChars();
    sinits.push_back(DtoConstString(name));

    // void[] init
    const LLPointerType* initpt = getPtrToType(LLType::Int8Ty);
    if (tinfo->isZeroInit() || !sd->defaultval) // 0 initializer, or the same as the base type
    {
        sinits.push_back(DtoConstSlice(DtoConstSize_t(0), llvm::ConstantPointerNull::get(initpt)));
    }
    else
    {
    #if DMDV2
        assert(0 && "initializer not implemented");
    #else
        const LLType* memty = DtoType(sd->memtype);
        LLConstant* ci = llvm::ConstantInt::get(memty, sd->defaultval, !sd->memtype->isunsigned());
        std::string ciname(sd->mangle());
        ciname.append("__init");
        llvm::GlobalVariable* civar = new llvm::GlobalVariable(memty,true,llvm::GlobalValue::InternalLinkage,ci,ciname,gIR->module);
        LLConstant* cicast = llvm::ConstantExpr::getBitCast(civar, initpt);
        size_t cisize = getTypeStoreSize(memty);
        sinits.push_back(DtoConstSlice(DtoConstSize_t(cisize), cicast));
    #endif
    }

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

static void LLVM_D_Define_TypeInfoBase(Type* basetype, TypeInfoDeclaration* tid, ClassDeclaration* cd)
{
    ClassDeclaration* base = cd;
    base->codegen(Type::sir);

    // vtbl
    std::vector<LLConstant*> sinits;
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    // TypeInfo base
    LLConstant* castbase = DtoTypeInfoOf(basetype, true);
    sinits.push_back(castbase);

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(tid->ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(tid->ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

void TypeInfoPointerDeclaration::llvmDefine()
{
    Logger::println("TypeInfoPointerDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Tpointer);
    TypePointer *tc = (TypePointer *)tinfo;

    LLVM_D_Define_TypeInfoBase(tc->next, this, Type::typeinfopointer);
}

/* ========================================================================= */

void TypeInfoArrayDeclaration::llvmDefine()
{
    Logger::println("TypeInfoArrayDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Tarray);
    TypeDArray *tc = (TypeDArray *)tinfo;

    LLVM_D_Define_TypeInfoBase(tc->next, this, Type::typeinfoarray);
}

/* ========================================================================= */

void TypeInfoStaticArrayDeclaration::llvmDefine()
{
    Logger::println("TypeInfoStaticArrayDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // init typeinfo class
    ClassDeclaration* base = Type::typeinfostaticarray;
    base->codegen(Type::sir);

    // get type of typeinfo class
    const LLStructType* stype = isaStruct(base->type->ir.type->get());

    // initializer vector
    std::vector<LLConstant*> sinits;
    // first is always the vtable
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    // value typeinfo
    assert(tinfo->ty == Tsarray);
    TypeSArray *tc = (TypeSArray *)tinfo;
    LLConstant* castbase = DtoTypeInfoOf(tc->next, true);
    assert(castbase->getType() == stype->getElementType(2));
    sinits.push_back(castbase);

    // length
    sinits.push_back(DtoConstSize_t(tc->dim->toInteger()));

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

void TypeInfoAssociativeArrayDeclaration::llvmDefine()
{
    Logger::println("TypeInfoAssociativeArrayDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // init typeinfo class
    ClassDeclaration* base = Type::typeinfoassociativearray;
    base->codegen(Type::sir);

    // initializer vector
    std::vector<LLConstant*> sinits;
    // first is always the vtable
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    // get type
    assert(tinfo->ty == Taarray);
    TypeAArray *tc = (TypeAArray *)tinfo;

    // value typeinfo
    LLConstant* castbase = DtoTypeInfoOf(tc->next, true);
    sinits.push_back(castbase);

    // key typeinfo
    castbase = DtoTypeInfoOf(tc->index, true);
    sinits.push_back(castbase);

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

void TypeInfoFunctionDeclaration::llvmDefine()
{
    Logger::println("TypeInfoFunctionDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Tfunction);
    TypeFunction *tc = (TypeFunction *)tinfo;

    LLVM_D_Define_TypeInfoBase(tc->next, this, Type::typeinfofunction);
}

/* ========================================================================= */

void TypeInfoDelegateDeclaration::llvmDefine()
{
    Logger::println("TypeInfoDelegateDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Tdelegate);
    TypeDelegate *tc = (TypeDelegate *)tinfo;

    LLVM_D_Define_TypeInfoBase(tc->nextOf()->nextOf(), this, Type::typeinfodelegate);
}

/* ========================================================================= */

void TypeInfoStructDeclaration::llvmDefine()
{
    Logger::println("TypeInfoStructDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // make sure struct is resolved
    assert(tinfo->ty == Tstruct);
    TypeStruct *tc = (TypeStruct *)tinfo;
    StructDeclaration *sd = tc->sym;
    sd->codegen(Type::sir);

    ClassDeclaration* base = Type::typeinfostruct;
    base->codegen(Type::sir);

    const LLStructType* stype = isaStruct(base->type->ir.type->get());

    // vtbl
    std::vector<LLConstant*> sinits;
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    // char[] name
    char *name = sd->toPrettyChars();
    sinits.push_back(DtoConstString(name));
    //Logger::println("************** A");
    assert(sinits.back()->getType() == stype->getElementType(2));

    // void[] init
    const LLPointerType* initpt = getPtrToType(LLType::Int8Ty);
#if 0
    // the implementation of TypeInfo_Struct uses this to determine size. :/
    if (sd->zeroInit) // 0 initializer, or the same as the base type
    {
        sinits.push_back(DtoConstSlice(DtoConstSize_t(0), llvm::ConstantPointerNull::get(initpt)));
    }
    else
#endif
    {
        size_t cisize = getTypeStoreSize(tc->ir.type->get());
        LLConstant* cicast = llvm::ConstantExpr::getBitCast(sd->ir.irStruct->getInitSymbol(), initpt);
        sinits.push_back(DtoConstSlice(DtoConstSize_t(cisize), cicast));
    }

    // toX functions ground work
    FuncDeclaration *fd;
    FuncDeclaration *fdx;
    TypeFunction *tf;
    Type *ta;
    Dsymbol *s;

    static TypeFunction *tftohash;
    static TypeFunction *tftostring;

    if (!tftohash)
    {
    Scope sc;

    tftohash = new TypeFunction(NULL, Type::thash_t, 0, LINKd);
    tftohash = (TypeFunction *)tftohash->semantic(0, &sc);

    tftostring = new TypeFunction(NULL, Type::tchar->arrayOf(), 0, LINKd);
    tftostring = (TypeFunction *)tftostring->semantic(0, &sc);
    }

    TypeFunction *tfeqptr;
    {
    Scope sc;
    Arguments *arguments = new Arguments;
    Argument *arg = new Argument(STCin, tc->pointerTo(), NULL, NULL);

    arguments->push(arg);
    tfeqptr = new TypeFunction(arguments, Type::tint32, 0, LINKd);
    tfeqptr = (TypeFunction *)tfeqptr->semantic(0, &sc);
    }

#if 0
    TypeFunction *tfeq;
    {
    Scope sc;
    Array *arguments = new Array;
    Argument *arg = new Argument(In, tc, NULL, NULL);

    arguments->push(arg);
    tfeq = new TypeFunction(arguments, Type::tint32, 0, LINKd);
    tfeq = (TypeFunction *)tfeq->semantic(0, &sc);
    }
#endif

    //Logger::println("************** B");
    const LLPointerType* ptty = isaPointer(stype->getElementType(4));
    assert(ptty);

    s = search_function(sd, Id::tohash);
    fdx = s ? s->isFuncDeclaration() : NULL;
    if (fdx)
    {
        fd = fdx->overloadExactMatch(tftohash);
        if (fd) {
            fd->codegen(Type::sir);
            assert(fd->ir.irFunc->func != 0);
            LLConstant* c = isaConstant(fd->ir.irFunc->func);
            assert(c);
            c = llvm::ConstantExpr::getBitCast(c, ptty);
            sinits.push_back(c);
        }
        else {
            //fdx->error("must be declared as extern (D) uint toHash()");
            sinits.push_back(llvm::ConstantPointerNull::get(ptty));
        }
    }
    else {
        sinits.push_back(llvm::ConstantPointerNull::get(ptty));
    }

    s = search_function(sd, Id::eq);
    fdx = s ? s->isFuncDeclaration() : NULL;
    for (int i = 0; i < 2; i++)
    {
        //Logger::println("************** C %d", i);
        ptty = isaPointer(stype->getElementType(5+i));
        if (fdx)
        {
            fd = fdx->overloadExactMatch(tfeqptr);
            if (fd) {
                fd->codegen(Type::sir);
                assert(fd->ir.irFunc->func != 0);
                LLConstant* c = isaConstant(fd->ir.irFunc->func);
                assert(c);
                c = llvm::ConstantExpr::getBitCast(c, ptty);
                sinits.push_back(c);
            }
            else {
                //fdx->error("must be declared as extern (D) int %s(%s*)", fdx->toChars(), sd->toChars());
                sinits.push_back(llvm::ConstantPointerNull::get(ptty));
            }
        }
        else {
            sinits.push_back(llvm::ConstantPointerNull::get(ptty));
        }

        s = search_function(sd, Id::cmp);
        fdx = s ? s->isFuncDeclaration() : NULL;
    }

    //Logger::println("************** D");
    ptty = isaPointer(stype->getElementType(7));
    s = search_function(sd, Id::tostring);
    fdx = s ? s->isFuncDeclaration() : NULL;
    if (fdx)
    {
        fd = fdx->overloadExactMatch(tftostring);
        if (fd) {
            fd->codegen(Type::sir);
            assert(fd->ir.irFunc->func != 0);
            LLConstant* c = isaConstant(fd->ir.irFunc->func);
            assert(c);
            c = llvm::ConstantExpr::getBitCast(c, ptty);
            sinits.push_back(c);
        }
        else {
            //fdx->error("must be declared as extern (D) char[] toString()");
            sinits.push_back(llvm::ConstantPointerNull::get(ptty));
        }
    }
    else {
        sinits.push_back(llvm::ConstantPointerNull::get(ptty));
    }

    // uint m_flags;
    sinits.push_back(DtoConstUint(tc->hasPointers()));

#if DMDV2

    // const(MemberInfo[]) function(in char[]) xgetMembers;
    sinits.push_back(LLConstant::getNullValue(stype->getElementType(sinits.size())));

    //void function(void*)                    xdtor;
    sinits.push_back(LLConstant::getNullValue(stype->getElementType(sinits.size())));

    //void function(void*)                    xpostblit;
    sinits.push_back(LLConstant::getNullValue(stype->getElementType(sinits.size())));

#endif

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

void TypeInfoClassDeclaration::llvmDefine()
{
    Logger::println("TypeInfoClassDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // make sure class is resolved
    assert(tinfo->ty == Tclass);
    TypeClass *tc = (TypeClass *)tinfo;
    tc->sym->codegen(Type::sir);

    // init typeinfo class
    ClassDeclaration* base = Type::typeinfoclass;
    assert(base);
    base->codegen(Type::sir);

    // initializer vector
    std::vector<LLConstant*> sinits;
    // first is always the vtable
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    // get classinfo
    sinits.push_back(tc->sym->ir.irStruct->getClassInfoSymbol());

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

void TypeInfoInterfaceDeclaration::llvmDefine()
{
    Logger::println("TypeInfoInterfaceDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // make sure interface is resolved
    assert(tinfo->ty == Tclass);
    TypeClass *tc = (TypeClass *)tinfo;
    tc->sym->codegen(Type::sir);

    // init typeinfo class
    ClassDeclaration* base = Type::typeinfointerface;
    assert(base);
    base->codegen(Type::sir);

    // get type of typeinfo class
    const LLStructType* stype = isaStruct(base->type->ir.type->get());

    // initializer vector
    std::vector<LLConstant*> sinits;
    // first is always the vtable
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    // get classinfo
    sinits.push_back(tc->sym->ir.irStruct->getClassInfoSymbol());

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

void TypeInfoTupleDeclaration::llvmDefine()
{
    Logger::println("TypeInfoTupleDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // init typeinfo class
    ClassDeclaration* base = Type::typeinfotypelist;
    assert(base);
    base->codegen(Type::sir);

    // get type of typeinfo class
    const LLStructType* stype = isaStruct(base->type->ir.type->get());

    // initializer vector
    std::vector<LLConstant*> sinits;
    // first is always the vtable
    sinits.push_back(base->ir.irStruct->getVtblSymbol());

    // monitor
    sinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    // create elements array
    assert(tinfo->ty == Ttuple);
    TypeTuple *tu = (TypeTuple *)tinfo;

    size_t dim = tu->arguments->dim;
    std::vector<LLConstant*> arrInits;

    const LLType* tiTy = Type::typeinfo->type->ir.type->get();
    tiTy = getPtrToType(tiTy);

    for (size_t i = 0; i < dim; i++)
    {
        Argument *arg = (Argument *)tu->arguments->data[i];
        LLConstant* castbase = DtoTypeInfoOf(arg->type, true);
        assert(castbase->getType() == tiTy);
        arrInits.push_back(castbase);
    }

    // build array type
    const LLArrayType* arrTy = LLArrayType::get(tiTy, dim);
    LLConstant* arrC = llvm::ConstantArray::get(arrTy, arrInits);

    // need the pointer to the first element of arrC, so create a global for it
    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(arrTy,true,_linkage,arrC,".tupleelements",gIR->module);

    // get pointer to first element
    llvm::ConstantInt* zero = DtoConstSize_t(0);
    LLConstant* idxs[2] = { zero, zero };
    LLConstant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar, idxs, 2);

    // build the slice
    LLConstant* slice = DtoConstSlice(DtoConstSize_t(dim), arrptr);
    sinits.push_back(slice);

    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(sinits);

    // refine global type
    llvm::cast<llvm::OpaqueType>(ir.irGlobal->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(ir.irGlobal->value)->setInitializer(tiInit);
}

/* ========================================================================= */

#if DMDV2

void TypeInfoConstDeclaration::llvmDefine()
{
    Logger::println("TypeInfoConstDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    Type *tm = tinfo->mutableOf();
    tm = tm->merge();

    LLVM_D_Define_TypeInfoBase(tm, this, Type::typeinfoconst);
}

/* ========================================================================= */

void TypeInfoInvariantDeclaration::llvmDefine()
{
    Logger::println("TypeInfoInvariantDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    Type *tm = tinfo->mutableOf();
    tm = tm->merge();

    LLVM_D_Define_TypeInfoBase(tm, this, Type::typeinfoinvariant);
}

#endif
