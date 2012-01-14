

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
#include "gen/rttibuilder.h"

#include "ir/irvar.h"
#include "ir/irtype.h"

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
    //printf("Type::getTypeInfo() %p, %s\n", this, toChars());
    if (!Type::typeinfo)
    {
        error(0, "TypeInfo not found. object.d may be incorrectly installed or corrupt, compile with -v switch");
        fatal();
    }

    Expression *e = 0;
    Type *t = merge2(); // do this since not all Type's are merge'd

    if (!t->vtinfo)
    {
#if DMDV2
        if (t->isShared())
            t->vtinfo = new TypeInfoSharedDeclaration(t);
        else if (t->isConst())
            t->vtinfo = new TypeInfoConstDeclaration(t);
        else if (t->isImmutable())
            t->vtinfo = new TypeInfoInvariantDeclaration(t);
        else if (t->isWild())
            t->vtinfo = new TypeInfoWildDeclaration(t);
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
    return mod ? 0 : 1;
#else
    return 1;
#endif
}

int TypeDArray::builtinTypeInfo()
{
#if DMDV2
    return !mod && (next->isTypeBasic() != NULL && !next->mod ||
        // strings are so common, make them builtin
        next->ty == Tchar && next->mod == MODimmutable);
#else
    return next->isTypeBasic() != NULL;
#endif
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

void DtoResolveTypeInfo(TypeInfoDeclaration* tid);
void DtoDeclareTypeInfo(TypeInfoDeclaration* tid);

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

    std::string mangle(tid->mangle());

    IrGlobal* irg = new IrGlobal(tid);
    irg->value = gIR->module->getGlobalVariable(mangle);

    if (!irg->value) {
        if (tid->tinfo->builtinTypeInfo()) // this is a declaration of a builtin __initZ var
            irg->type = Type::typeinfo->type->irtype->getType();
        else
            irg->type = LLStructType::create(gIR->context(), tid->toPrettyChars());
        irg->value = new llvm::GlobalVariable(*gIR->module, irg->type, true,
                                              TYPEINFO_LINKAGE_TYPE, NULL, mangle);
    } else {
        irg->type = irg->value->getType()->getContainedType(0);
    }

    tid->ir.irGlobal = irg;

#if USE_METADATA
    // don't do this for void or llvm will crash
    if (tid->tinfo->ty != Tvoid) {
        // Add some metadata for use by optimization passes.
        std::string metaname = std::string(TD_PREFIX) + mangle;
        llvm::NamedMDNode* meta = gIR->module->getNamedMetadata(metaname);
        // Don't generate metadata for non-concrete types
        // (such as tuple types, slice types, typeof(expr), etc.)
        if (!meta && tid->tinfo->toBasetype()->ty < Terror) {
            // Construct the fields
            MDNodeField* mdVals[TD_NumFields];
            if (TD_Confirm >= 0)
                mdVals[TD_Confirm] = llvm::cast<MDNodeField>(irg->value);
            mdVals[TD_Type] = llvm::UndefValue::get(DtoType(tid->tinfo));
            // Construct the metadata
            llvm::MetadataBase* metadata = llvm::MDNode::get(gIR->context(), mdVals, TD_NumFields);
            // Insert it into the module
            llvm::NamedMDNode::Create(gIR->context(), metaname, &metadata, 1, gIR->module);
        }
    }
#endif // USE_METADATA

    DtoDeclareTypeInfo(tid);
}

void DtoDeclareTypeInfo(TypeInfoDeclaration* tid)
{
    DtoResolveTypeInfo(tid);

    if (tid->ir.declared) return;
    tid->ir.declared = true;

    Logger::println("DtoDeclareTypeInfo(%s)", tid->toChars());
    LOG_SCOPE;

    if (Logger::enabled())
    {
        std::string mangled(tid->mangle());
        Logger::println("type = '%s'", tid->tinfo->toChars());
        Logger::println("typeinfo mangle: %s", mangled.c_str());
    }

    IrGlobal* irg = tid->ir.irGlobal;
    assert(irg->value != NULL);

    // this is a declaration of a builtin __initZ var
    if (tid->tinfo->builtinTypeInfo()) {
        LLGlobalVariable* g = isaGlobalVar(irg->value);
        g->setLinkage(llvm::GlobalValue::ExternalLinkage);
        return;
    }

    // define custom typedef
    tid->llvmDefine();
}

/* ========================================================================= */

void TypeInfoDeclaration::llvmDefine()
{
    Logger::println("TypeInfoDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfo);
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoTypedefDeclaration::llvmDefine()
{
    Logger::println("TypeInfoTypedefDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    RTTIBuilder b(Type::typeinfotypedef);

    assert(tinfo->ty == Ttypedef);
    TypeTypedef *tc = (TypeTypedef *)tinfo;
    TypedefDeclaration *sd = tc->sym;

    // TypeInfo base
    sd->basetype = sd->basetype->merge(); // dmd does it ... why?
    b.push_typeinfo(sd->basetype);

    // char[] name
    b.push_string(sd->toPrettyChars());

    // void[] init
    // emit null array if we should use the basetype, or if the basetype
    // uses default initialization.
    if (tinfo->isZeroInit(0) || !sd->init)
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
    TypeEnum *tc = (TypeEnum *)tinfo;
    EnumDeclaration *sd = tc->sym;

    // TypeInfo base
    b.push_typeinfo(sd->memtype);

    // char[] name
    b.push_string(sd->toPrettyChars());

    // void[] init
    // emit void[] with the default initialier, the array is null if the default
    // initializer is zero
    if (!sd->defaultval || tinfo->isZeroInit(0))
    {
        b.push_null_void_array();
    }
    // otherwise emit a void[] with the default initializer
    else
    {
        LLType* memty = DtoType(sd->memtype);
#if DMDV2
        LLConstant* C = LLConstantInt::get(memty, sd->defaultval->toInteger(), !sd->memtype->isunsigned());
#else
        LLConstant* C = LLConstantInt::get(memty, sd->defaultval, !sd->memtype->isunsigned());
#endif
        b.push_void_array(C, sd->memtype, sd);
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
    TypeSArray *tc = (TypeSArray *)tinfo;

    RTTIBuilder b(Type::typeinfostaticarray);

    // value typeinfo
    b.push_typeinfo(tc->nextOf());

    // length
    b.push(DtoConstSize_t((size_t)tc->dim->toUInteger()));

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

void TypeInfoAssociativeArrayDeclaration::llvmDefine()
{
    Logger::println("TypeInfoAssociativeArrayDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    assert(tinfo->ty == Taarray);
    TypeAArray *tc = (TypeAArray *)tinfo;

    RTTIBuilder b(Type::typeinfoassociativearray);

    // value typeinfo
    b.push_typeinfo(tc->nextOf());

    // key typeinfo
    b.push_typeinfo(tc->index);

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

static FuncDeclaration* find_method_overload(AggregateDeclaration* ad, Identifier* id, TypeFunction* tf, Module* mod)
{
    Dsymbol *s = search_function(ad, id);
    FuncDeclaration *fdx = s ? s->isFuncDeclaration() : NULL;
    if (fdx)
    {
        FuncDeclaration *fd = fdx->overloadExactMatch(tf, mod);
        if (fd)
        {
            return fd;
        }
    }
    return NULL;
}

void TypeInfoStructDeclaration::llvmDefine()
{
    Logger::println("TypeInfoStructDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // make sure struct is resolved
    assert(tinfo->ty == Tstruct);
    TypeStruct *tc = (TypeStruct *)tinfo;
    StructDeclaration *sd = tc->sym;

    // can't emit typeinfo for forward declarations
    if (sd->sizeok != 1)
    {
        sd->error("cannot emit TypeInfo for forward declaration");
        fatal();
    }

    sd->codegen(Type::sir);
    IrStruct* irstruct = sd->ir.irStruct;

    RTTIBuilder b(Type::typeinfostruct);

    // char[] name
    b.push_string(sd->toPrettyChars());

    // void[] init
    // never emit a null array, even for zero initialized typeinfo
    // the size() method uses this array!
    size_t init_size = getTypeStoreSize(tc->irtype->getType());
    b.push_void_array(init_size, irstruct->getInitSymbol());

    // toX functions ground work
    static TypeFunction *tftohash;
    static TypeFunction *tftostring;

    if (!tftohash)
    {
        Scope sc;
        tftohash = new TypeFunction(NULL, Type::thash_t, 0, LINKd);
#if DMDV2
        tftohash ->mod = MODconst;
#endif
        tftohash = (TypeFunction *)tftohash->semantic(0, &sc);

#if DMDV2
        Type *retType = Type::tchar->invariantOf()->arrayOf();
#else
        Type *retType = Type::tchar->arrayOf();
#endif
        tftostring = new TypeFunction(NULL, retType, 0, LINKd);
        tftostring = (TypeFunction *)tftostring->semantic(0, &sc);
    }

    // this one takes a parameter, so we need to build a new one each time
    // to get the right type. can we avoid this?
    TypeFunction *tfcmpptr;
    {
        Scope sc;
        Parameters *arguments = new Parameters;
#if STRUCTTHISREF
        // arg type is ref const T
        Parameter *arg = new Parameter(STCref, tc->constOf(), NULL, NULL);
#else
        // arg type is const T*
        Parameter *arg = new Parameter(STCin, tc->pointerTo(), NULL, NULL);
#endif
        arguments->push(arg);
        tfcmpptr = new TypeFunction(arguments, Type::tint32, 0, LINKd);
#if DMDV2
        tfcmpptr->mod = MODconst;
#endif
        tfcmpptr = (TypeFunction *)tfcmpptr->semantic(0, &sc);
    }

    // well use this module for all overload lookups
    Module *gm = getModule();

    // toHash
    FuncDeclaration* fd = find_method_overload(sd, Id::tohash, tftohash, gm);
    b.push_funcptr(fd);

    // opEquals
#if DMDV2
    fd = sd->xeq;
#else
    fd = find_method_overload(sd, Id::eq, tfcmpptr, gm);
#endif
    b.push_funcptr(fd);

    // opCmp
    fd = find_method_overload(sd, Id::cmp, tfcmpptr, gm);
    b.push_funcptr(fd);

    // toString
    fd = find_method_overload(sd, Id::tostring, tftostring, gm);
    b.push_funcptr(fd);

    // uint m_flags;
    unsigned hasptrs = tc->hasPointers() ? 1 : 0;
    b.push_uint(hasptrs);

#if DMDV2

    ClassDeclaration* tscd = Type::typeinfostruct;

    assert((!global.params.is64bit && tscd->fields.dim == 11) ||
           (global.params.is64bit && tscd->fields.dim == 13));

    // const(MemberInfo[]) function(in char[]) xgetMembers;
    b.push_funcptr(sd->findGetMembers());

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
        TypeTuple *tup = tc->toArgTypes();
        assert(tup->arguments->dim <= 2);
        for (unsigned i = 0; i < 2; i++)
        {
            if (i < tup->arguments->dim)
            {
                Type *targ = ((Parameter *)tup->arguments->data[i])->type;
                targ = targ->merge();
                b.push_typeinfo(targ);
            }
            else
                b.push_null(Type::typeinfo->type);
        }
    }

#endif

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

#if DMDV2
void TypeInfoClassDeclaration::codegen(Ir*i)
{

    IrGlobal* irg = new IrGlobal(this);
    ir.irGlobal = irg;
    assert(tinfo->ty == Tclass);
    TypeClass *tc = (TypeClass *)tinfo;
    tc->sym->codegen(Type::sir); // make sure class is resolved
    irg->value = tc->sym->ir.irStruct->getClassInfoSymbol();
}
#endif

void TypeInfoClassDeclaration::llvmDefine()
{
#if DMDV2
    assert(0);
#endif
    Logger::println("TypeInfoClassDeclaration::llvmDefine() %s", toChars());
    LOG_SCOPE;

    // make sure class is resolved
    assert(tinfo->ty == Tclass);
    TypeClass *tc = (TypeClass *)tinfo;
    tc->sym->codegen(Type::sir);

    RTTIBuilder b(Type::typeinfoclass);

    // TypeInfo base
    b.push_classinfo(tc->sym);

    // finish
    b.finalize(ir.irGlobal);
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
    TypeTuple *tu = (TypeTuple *)tinfo;

    size_t dim = tu->arguments->dim;
    std::vector<LLConstant*> arrInits;
    arrInits.reserve(dim);

    LLType* tiTy = DtoType(Type::typeinfo->type);

    for (size_t i = 0; i < dim; i++)
    {
        Parameter *arg = (Parameter *)tu->arguments->data[i];
        arrInits.push_back(DtoTypeInfoOf(arg->type, true));
    }

    // build array
    LLArrayType* arrTy = LLArrayType::get(tiTy, dim);
    LLConstant* arrC = LLConstantArray::get(arrTy, arrInits);

    RTTIBuilder b(Type::typeinfotypelist);

    // push TypeInfo[]
    b.push_array(arrC, dim, Type::typeinfo->type, NULL);

    // finish
    b.finalize(ir.irGlobal);
}

/* ========================================================================= */

#if DMDV2

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

#endif
