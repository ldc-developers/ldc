//===-- gen/dibuilder.h - Debug information builder -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dibuilder.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/optimizer.h"
#include "ir/irtypeaggr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "enum.h"
#include "module.h"
#include "mtype.h"

////////////////////////////////////////////////////////////////////////////////

#if LDC_LLVM_VER >= 307
typedef llvm::DINode DIFlags;
#else
typedef llvm::DIDescriptor DIFlags;
#endif

////////////////////////////////////////////////////////////////////////////////

// get the module the symbol is in, or - for template instances - the current module
Module *ldc::DIBuilder::getDefinedModule(Dsymbol *s)
{
    // templates are defined in current module
    if (DtoIsTemplateInstance(s))
    {
        return IR->dmodule;
    }
    // array operations as well
    else if (FuncDeclaration* fd = s->isFuncDeclaration())
    {
        if (fd->isArrayOp && (willInline() || !isDruntimeArrayOp(fd)))
            return IR->dmodule;
    }
    // otherwise use the symbol's module
    return s->getModule();
}

////////////////////////////////////////////////////////////////////////////////

ldc::DIBuilder::DIBuilder(IRState *const IR, llvm::Module &M)
    : IR(IR), DBuilder(M)
{
}

llvm::LLVMContext &ldc::DIBuilder::getContext()
{
    return IR->context();
}

ldc::DIScope ldc::DIBuilder::GetCurrentScope()
{
    IrFunction *fn = IR->func();
    if (fn->diLexicalBlocks.empty())
    {
        assert(static_cast<llvm::MDNode *>(fn->diSubprogram) != 0);
        return fn->diSubprogram;
    }
    return fn->diLexicalBlocks.top();
}

void ldc::DIBuilder::Declare(llvm::Value *var, ldc::DILocalVariable divar
#if LDC_LLVM_VER >= 306
    , ldc::DIExpression diexpr
#endif
    )
{
    llvm::Instruction *instr = DBuilder.insertDeclare(var, divar,
#if LDC_LLVM_VER >= 306
        diexpr,
#endif
#if LDC_LLVM_VER >= 307
        IR->ir->getCurrentDebugLocation(),
#endif
        IR->scopebb());
    instr->setDebugLoc(IR->ir->getCurrentDebugLocation());
}

ldc::DIFile ldc::DIBuilder::CreateFile(Loc& loc)
{
    llvm::SmallString<128> path(loc.filename ? loc.filename : "");
    llvm::sys::fs::make_absolute(path);

    return DBuilder.createFile(
        llvm::sys::path::filename(path),
        llvm::sys::path::parent_path(path)
    );
}

ldc::DIType ldc::DIBuilder::CreateBasicType(Type *type)
{
    using namespace llvm::dwarf;

    Type *t = type->toBasetype();
    llvm::Type *T = DtoType(type);

    // find encoding
    unsigned Encoding;
    switch (t->ty)
    {
    case Tbool:
        Encoding = DW_ATE_boolean;
        break;
    case Tchar:
    case Twchar:
    case Tdchar:
        Encoding = type->isunsigned() ? DW_ATE_unsigned_char
                                      : DW_ATE_signed_char;
        break;
    case Tint8:
    case Tint16:
    case Tint32:
    case Tint64:
    case Tint128:
        Encoding = DW_ATE_signed;
        break;
    case Tuns8:
    case Tuns16:
    case Tuns32:
    case Tuns64:
    case Tuns128:
        Encoding = DW_ATE_unsigned;
        break;
    case Tfloat32:
    case Tfloat64:
    case Tfloat80:
        Encoding = DW_ATE_float;
        break;
    case Timaginary32:
    case Timaginary64:
    case Timaginary80:
        Encoding = DW_ATE_imaginary_float;
        break;
    case Tcomplex32:
    case Tcomplex64:
    case Tcomplex80:
        Encoding = DW_ATE_complex_float;
        break;
    default:
        llvm_unreachable("Unsupported basic type for debug info in DIBuilder::CreateBasicType");
    }

    return DBuilder.createBasicType(
        type->toChars(), // name
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        Encoding
    );
}

ldc::DIType ldc::DIBuilder::CreateEnumType(Type *type)
{
    llvm::Type *T = DtoType(type);

    assert(type->ty == Tenum && "only enums allowed for debug info in dwarfEnumType");
    TypeEnum *te = static_cast<TypeEnum *>(type);
#if LDC_LLVM_VER >= 306
    llvm::SmallVector<llvm::Metadata *, 8> subscripts;
#else
    llvm::SmallVector<llvm::Value *, 8> subscripts;
#endif
    for (Dsymbols::iterator I = te->sym->members->begin(),
                            E = te->sym->members->end();
                            I != E; ++I)
    {
        EnumMember *em = (*I)->isEnumMember();
        llvm::StringRef Name(em->toChars());
        uint64_t Val = em->value->toInteger();
#if LDC_LLVM_VER >= 306
        llvm::Metadata *Subscript = DBuilder.createEnumerator(Name, Val);
#else
        llvm::Value *Subscript = DBuilder.createEnumerator(Name, Val);
#endif
        subscripts.push_back(Subscript);
    }

    llvm::StringRef Name = te->toChars();
    unsigned LineNumber = te->sym->loc.linnum;
    ldc::DIFile File(CreateFile(te->sym->loc));

    return DBuilder.createEnumerationType(
        GetCU(),
        Name,
        File,
        LineNumber,
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        DBuilder.getOrCreateArray(subscripts) // subscripts
#if LDC_LLVM_VER >= 302
        , CreateTypeDescription(te->sym->memtype, false)
#endif
    );
}

ldc::DIType ldc::DIBuilder::CreatePointerType(Type *type)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    assert(t->ty == Tpointer && "Only pointers allowed for debug info in DIBuilder::CreatePointerType");

    // find base type
    Type *nt = t->nextOf();
    ldc::DIType basetype(CreateTypeDescription(nt, false));

    return DBuilder.createPointerType(
        basetype,
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        type->toChars() // name
    );
}

ldc::DIType ldc::DIBuilder::CreateVectorType(Type *type)
{
    LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    assert(t->ty == Tvector && "Only vectors allowed for debug info in DIBuilder::CreateVectorType");
    TypeVector *tv = static_cast<TypeVector *>(t);
    Type *te = tv->elementType();
    int64_t Dim = tv->size(Loc()) / te->size(Loc());
#if LDC_LLVM_VER >= 306
    llvm::Metadata *subscripts[] =
#else
    llvm::Value *subscripts[] =
#endif
    {
        DBuilder.getOrCreateSubrange(0, Dim)
    };
    ldc::DIType basetype(CreateTypeDescription(te, false));

    return DBuilder.createVectorType(
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        basetype, // element type
        DBuilder.getOrCreateArray(subscripts) // subscripts
    );
}

ldc::DIType ldc::DIBuilder::CreateMemberType(unsigned linnum, Type *type,
                                             ldc::DIFile file,
                                             const char* c_name,
                                             unsigned offset,
                                             PROTKIND prot)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    // find base type
    ldc::DIType basetype(CreateTypeDescription(t, true));

    unsigned Flags = 0;
    switch (prot) {
        case PROTprivate:
            Flags = DIFlags::FlagPrivate;
            break;
        case PROTprotected:
            Flags = DIFlags::FlagProtected;
            break;
#if LDC_LLVM_VER >= 306
        case PROTpublic:
            Flags = DIFlags::FlagPublic;
            break;
#endif
        default:
            break;
    }

    return DBuilder.createMemberType(
        GetCU(),
        c_name, // name
        file, // file
        linnum, // line number
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        offset*8, // offset (bits)
        Flags, // flags
        basetype // derived from
    );
}

void ldc::DIBuilder::AddBaseFields(ClassDeclaration *sd, ldc::DIFile file,
#if LDC_LLVM_VER >= 306
                                   std::vector<llvm::Metadata*> &elems
#else
                                   std::vector<llvm::Value*> &elems
#endif
                                   )
{
    if (sd->baseClass)
    {
        AddBaseFields(sd->baseClass, file, elems);
    }

    size_t narr = sd->fields.dim;
    elems.reserve(narr);
    for (VarDeclarations::iterator I = sd->fields.begin(),
                                   E = sd->fields.end();
                                   I != E; ++I)
    {
        VarDeclaration* vd = *I;
        elems.push_back(CreateMemberType(vd->loc.linnum, vd->type, file, vd->toChars(), vd->offset, vd->prot().kind));
    }
}

ldc::DIType ldc::DIBuilder::CreateCompositeType(Type *type)
{
    Type* t = type->toBasetype();
    assert((t->ty == Tstruct || t->ty == Tclass) &&
           "Unsupported type for debug info in DIBuilder::CreateCompositeType");
    AggregateDeclaration* sd;
    if (t->ty == Tstruct)
    {
        TypeStruct* ts = static_cast<TypeStruct*>(t);
        sd = ts->sym;
    }
    else
    {
        TypeClass* tc = static_cast<TypeClass*>(t);
        sd = tc->sym;
    }
    assert(sd);

    // Use the actual type associated with the declaration, ignoring any
    // const/wrappers.
    LLType *T = DtoType(sd->type);
    IrTypeAggr *ir = sd->type->ctype->isAggr();
    assert(ir);

    if (static_cast<llvm::MDNode *>(ir->diCompositeType) != 0)
        return ir->diCompositeType;

    // if we don't know the aggregate's size, we don't know enough about it
    // to provide debug info. probably a forward-declared struct?
    if (sd->sizeok == SIZEOKnone)
#if LDC_LLVM_VER >= 304
        return DBuilder.createUnspecifiedType(sd->toChars());
#else
        return llvm::DICompositeType(NULL);
#endif

    // elements
#if LDC_LLVM_VER >= 306
    std::vector<llvm::Metadata *> elems;
#else
    std::vector<llvm::Value *> elems;
#endif

    // defaults
    llvm::StringRef name = sd->toChars();
    unsigned linnum = sd->loc.linnum;
    ldc::DICompileUnit CU(GetCU());
    assert(CU && "Compilation unit missing or corrupted");
    ldc::DIFile file(CreateFile(sd->loc));
#if LDC_LLVM_VER >= 307
    ldc::DIType derivedFrom = nullptr;
#else
    ldc::DIType derivedFrom;
#endif

    // set diCompositeType to handle recursive types properly
    unsigned tag = (t->ty == Tstruct) ? llvm::dwarf::DW_TAG_structure_type
                                        : llvm::dwarf::DW_TAG_class_type;
#if LDC_LLVM_VER >= 307
    ir->diCompositeType = DBuilder.createReplaceableCompositeType(
#elif LDC_LLVM_VER >= 305
    ir->diCompositeType = DBuilder.createReplaceableForwardDecl(
#else
    ir->diCompositeType = DBuilder.createForwardDecl(
#endif
                                                           tag, name,
#if LDC_LLVM_VER >= 302
                                                           CU,
#endif
                                                           file, linnum);

    if (!sd->isInterfaceDeclaration()) // plain interfaces don't have one
    {
        if (t->ty == Tstruct)
        {
            elems.reserve(sd->fields.dim);
            for (VarDeclarations::iterator I = sd->fields.begin(),
                                           E = sd->fields.end();
                                           I != E; ++I)
            {
                VarDeclaration* vd = *I;
                ldc::DIType dt = CreateMemberType(vd->loc.linnum, vd->type, file, vd->toChars(), vd->offset, vd->prot().kind);
                elems.push_back(dt);
            }
        }
        else
        {
            ClassDeclaration *classDecl = sd->isClassDeclaration();
            AddBaseFields(classDecl, file, elems);
            if (classDecl->baseClass)
                derivedFrom = CreateCompositeType(classDecl->baseClass->getType());
        }
    }

#if LDC_LLVM_VER >= 307
    llvm::DINodeArray elemsArray = DBuilder.getOrCreateArray(elems);
#else
    llvm::DIArray elemsArray = DBuilder.getOrCreateArray(elems);
#endif

    ldc::DIType ret;
    if (t->ty == Tclass) {
        ret = DBuilder.createClassType(
           CU, // compile unit where defined
           name, // name
           file, // file where defined
           linnum, // line number where defined
           getTypeBitSize(T), // size in bits
           getABITypeAlign(T)*8, // alignment in bits
           0, // offset in bits,
           DIFlags::FlagFwdDecl, // flags
           derivedFrom, // DerivedFrom
           elemsArray
        );
    } else {
        ret = DBuilder.createStructType(
           CU, // compile unit where defined
           name, // name
           file, // file where defined
           linnum, // line number where defined
           getTypeBitSize(T), // size in bits
           getABITypeAlign(T)*8, // alignment in bits
           DIFlags::FlagFwdDecl, // flags
#if LDC_LLVM_VER >= 303
           derivedFrom, // DerivedFrom
#endif
           elemsArray
        );
    }

#if LDC_LLVM_VER >= 307
    ir->diCompositeType = DBuilder.replaceTemporary(llvm::TempDINode(ir->diCompositeType), static_cast<llvm::DIType*>(ret));
#else
    ir->diCompositeType.replaceAllUsesWith(ret);
#endif
    ir->diCompositeType = ret;

    return ret;
}

ldc::DIType ldc::DIBuilder::CreateArrayType(Type *type)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    assert(t->ty == Tarray && "Only arrays allowed for debug info in DIBuilder::CreateArrayType");

    Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
    ldc::DIFile file(CreateFile(loc));

#if LDC_LLVM_VER >= 306
    llvm::Metadata *elems[] =
#else
    llvm::Value *elems[] =
#endif
    {
        CreateMemberType(0, Type::tsize_t, file, "length", 0, PROTpublic),
        CreateMemberType(0, t->nextOf()->pointerTo(), file, "ptr",
                         global.params.is64bit ? 8 : 4, PROTpublic)
    };

    return DBuilder.createStructType
       (
        GetCU(),
        llvm::StringRef(), // Name TODO: Really no name for arrays? t->toChars()?
        file, // File
        0, // LineNo
        getTypeBitSize(T), // size in bits
        getABITypeAlign(T)*8, // alignment in bits
        0, // What here?
#if LDC_LLVM_VER >= 307
        nullptr, // DerivedFrom
#elif LDC_LLVM_VER >= 303
        llvm::DIType(), // DerivedFrom
#endif
        DBuilder.getOrCreateArray(elems)
    );
}

ldc::DIType ldc::DIBuilder::CreateSArrayType(Type *type)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    // find base type
#if LDC_LLVM_VER >= 306
    llvm::SmallVector<llvm::Metadata *, 8> subscripts;
#else
    llvm::SmallVector<llvm::Value *, 8> subscripts;
#endif
    while (t->ty == Tsarray)
    {
        TypeSArray *tsa = static_cast<TypeSArray *>(t);
        int64_t Count = tsa->dim->toInteger();
#if LDC_LLVM_VER >= 306
        llvm::Metadata *subscript = DBuilder.getOrCreateSubrange(0, Count - 1);
#else
        llvm::Value *subscript = DBuilder.getOrCreateSubrange(0, Count-1);
#endif
        subscripts.push_back(subscript);
        t = t->nextOf();
    }
    ldc::DIType basetype(CreateTypeDescription(t, false));

    return DBuilder.createArrayType(
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        basetype, // element type
        DBuilder.getOrCreateArray(subscripts) // subscripts
    );
}

ldc::DIType ldc::DIBuilder::CreateAArrayType(Type *type)
{
    // FIXME: Implement
#if LDC_LLVM_VER >= 304
    return DBuilder.createUnspecifiedType(type->toChars());
#else
    return llvm::DIType(NULL);
#endif
}

////////////////////////////////////////////////////////////////////////////////

ldc::DISubroutineType ldc::DIBuilder::CreateFunctionType(Type *type)
{
    TypeFunction *t = static_cast<TypeFunction*>(type);
    Type *retType = t->next;

    Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
    ldc::DIFile file(CreateFile(loc));

    // Create "dummy" subroutine type for the return type
#if LDC_LLVM_VER >= 306
    llvm::SmallVector<llvm::Metadata*, 16> Elts;
#else
    llvm::SmallVector<llvm::Value*, 16> Elts;
#endif
    Elts.push_back(CreateTypeDescription(retType, true));
#if LDC_LLVM_VER >= 307
    llvm::DITypeRefArray EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#elif LDC_LLVM_VER >= 306
    llvm::DITypeArray EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#else
    llvm::DIArray EltTypeArray = DBuilder.getOrCreateArray(Elts);
#endif
    return DBuilder.createSubroutineType(file, EltTypeArray);
}

ldc::DISubroutineType ldc::DIBuilder::CreateDelegateType(Type *type)
{
    // FIXME: Implement
    TypeDelegate *t = static_cast<TypeDelegate*>(type);

    Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
    ldc::DIFile file(CreateFile(loc));

    // Create "dummy" subroutine type for the return type
#if LDC_LLVM_VER >= 306
    llvm::SmallVector<llvm::Metadata*, 16> Elts;
#else
    llvm::SmallVector<llvm::Value*, 16> Elts;
#endif
    Elts.push_back(
#if LDC_LLVM_VER >= 304
        DBuilder.createUnspecifiedType(type->toChars())
#else
        llvm::DIType(NULL)
#endif
    );
#if LDC_LLVM_VER >= 307
    llvm::DITypeRefArray EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#elif LDC_LLVM_VER >= 306
    llvm::DITypeArray EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#else
    llvm::DIArray EltTypeArray = DBuilder.getOrCreateArray(Elts);
#endif
    return DBuilder.createSubroutineType(file, EltTypeArray);
}

////////////////////////////////////////////////////////////////////////////////

ldc::DIType ldc::DIBuilder::CreateTypeDescription(Type* type,
                                                   bool derefclass)
{
    Type *t = type->toBasetype();
    if (derefclass && t->ty == Tclass)
    {
        type = type->pointerTo();
        t = type->toBasetype();
    }

    if (t->ty == Tvoid || t->ty == Tnull)
#if LDC_LLVM_VER >= 304
        return DBuilder.createUnspecifiedType(t->toChars());
#else
        return llvm::DIType(NULL);
#endif
    else if (t->isintegral() || t->isfloating())
    {
        if (t->ty == Tvector)
            return CreateVectorType(type);
        if (type->ty == Tenum)
            return CreateEnumType(type);
        return CreateBasicType(type);
    }
    else if (t->ty == Tpointer)
        return CreatePointerType(type);
    else if (t->ty == Tarray)
        return CreateArrayType(type);
    else if (t->ty == Tsarray)
        return CreateSArrayType(type);
    else if (t->ty == Taarray)
        return CreateAArrayType(type);
    else if (t->ty == Tstruct || t->ty == Tclass)
        return CreateCompositeType(type);
    else if (t->ty == Tfunction)
        return CreateFunctionType(type);
    else if (t->ty == Tdelegate)
        return CreateDelegateType(type);

    // Crash if the type is not supported.
    llvm_unreachable("Unsupported type in debug info");
}

////////////////////////////////////////////////////////////////////////////////

void ldc::DIBuilder::EmitCompileUnit(Module *m)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf compile_unit");
    LOG_SCOPE;

    // prepare srcpath
    llvm::SmallString<128> srcpath(m->srcfile->name->toChars());
    llvm::sys::fs::make_absolute(srcpath);

#if LDC_LLVM_VER >= 304
    // Metadata without a correct version will be stripped by UpgradeDebugInfo.
    gIR->module->addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);

    CUNode =
#endif
    DBuilder.createCompileUnit(
        global.params.symdebug == 2 ? llvm::dwarf::DW_LANG_C
                                    : llvm::dwarf::DW_LANG_D,
        llvm::sys::path::filename(srcpath),
        llvm::sys::path::parent_path(srcpath),
        "LDC (http://wiki.dlang.org/LDC)",
        isOptimizationEnabled(), // isOptimized
        llvm::StringRef(), // Flags TODO
        1 // Runtime Version TODO
    );
#if LDC_LLVM_VER < 304
    CUNode = DBuilder.getCU();
#endif
}

ldc::DISubprogram ldc::DIBuilder::EmitSubProgram(FuncDeclaration *fd)
{
    if (!global.params.symdebug)
#if LDC_LLVM_VER >= 307
        return nullptr;
#else
        return llvm::DISubprogram();
#endif

    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    ldc::DICompileUnit CU(GetCU());
    assert(CU && "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");

    ldc::DIFile file(CreateFile(fd->loc));

    // Create subroutine type
    ldc::DISubroutineType DIFnType = CreateFunctionType(static_cast<TypeFunction*>(fd->type));

    // FIXME: duplicates ?
    return DBuilder.createFunction(
        CU, // context
        fd->toPrettyChars(), // name
        mangleExact(fd), // linkage name
        file, // file
        fd->loc.linnum, // line no
        DIFnType, // type
        fd->protection == PROTprivate, // is local to unit
        IR->dmodule == getDefinedModule(fd), // isdefinition
        fd->loc.linnum, // FIXME: scope line
        DIFlags::FlagPrototyped, // Flags
        isOptimizationEnabled(), // isOptimized
        getIrFunc(fd)->func
    );
}

ldc::DISubprogram ldc::DIBuilder::EmitModuleCTor(llvm::Function* Fn,
                                                 llvm::StringRef prettyname)
{
    if (!global.params.symdebug)
#if LDC_LLVM_VER >= 307
        return nullptr;
#else
        return llvm::DISubprogram();
#endif

    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    ldc::DICompileUnit CU(GetCU());
    assert(CU && "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");

    Loc loc(IR->dmodule->srcfile->toChars(), 0, 0);
    ldc::DIFile file(CreateFile(loc));

    // Create "dummy" subroutine type for the return type
#if LDC_LLVM_VER >= 306
    llvm::SmallVector<llvm::Metadata *, 1> Elts;
#else
    llvm::SmallVector<llvm::Value *, 1> Elts;
#endif
    Elts.push_back(CreateTypeDescription(Type::tvoid, true));
#if LDC_LLVM_VER >= 307
    llvm::DITypeRefArray EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#elif LDC_LLVM_VER >= 306
    llvm::DITypeArray EltTypeArray = DBuilder.getOrCreateTypeArray(Elts);
#else
    llvm::DIArray EltTypeArray = DBuilder.getOrCreateArray(Elts);
#endif
    ldc::DISubroutineType DIFnType = DBuilder.createSubroutineType(file, EltTypeArray);

    // FIXME: duplicates ?
    return DBuilder.createFunction(
        CU, // context
        prettyname, // name
        Fn->getName(), // linkage name
        file, // file
        0, // line no
        DIFnType, // return type. TODO: fill it up
        true, // is local to unit
        true, // isdefinition
        0, // FIXME: scope line
        DIFlags::FlagPrototyped | DIFlags::FlagArtificial,
        isOptimizationEnabled(), // isOptimized
        Fn
    );
}

void ldc::DIBuilder::EmitFuncStart(FuncDeclaration *fd)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf funcstart");
    LOG_SCOPE;

    assert(static_cast<llvm::MDNode *>(getIrFunc(fd)->diSubprogram) != 0);
    EmitStopPoint(fd->loc);
}

void ldc::DIBuilder::EmitFuncEnd(FuncDeclaration *fd)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf funcend");
    LOG_SCOPE;

    assert(static_cast<llvm::MDNode *>(getIrFunc(fd)->diSubprogram) != 0);
    EmitStopPoint(fd->endloc);
}

void ldc::DIBuilder::EmitBlockStart(Loc& loc)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf block start");
    LOG_SCOPE;

    ldc::DILexicalBlock block = DBuilder.createLexicalBlock(
            GetCurrentScope(), // scope
            CreateFile(loc), // file
            loc.linnum, // line
            loc.charnum // column
#if LDC_LLVM_VER == 305
            , 0 // DWARF path discriminator value
#endif
            );
    IR->func()->diLexicalBlocks.push(block);
    EmitStopPoint(loc);
}

void ldc::DIBuilder::EmitBlockEnd()
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf block end");
    LOG_SCOPE;

    IrFunction *fn = IR->func();
    assert(!fn->diLexicalBlocks.empty());
    fn->diLexicalBlocks.pop();
}

void ldc::DIBuilder::EmitStopPoint(Loc& loc)
{
    if (!global.params.symdebug || !loc.linnum)
        return;

    Logger::println("D to dwarf stoppoint at line %u, column %u", loc.linnum, loc.charnum);
    LOG_SCOPE;
    IR->ir->SetCurrentDebugLocation(llvm::DebugLoc::get(loc.linnum, loc.charnum, GetCurrentScope()));
}

void ldc::DIBuilder::EmitValue(llvm::Value *val, VarDeclaration *vd)
{
    IrFunction::VariableMap::iterator sub = IR->func()->variableMap.find(vd);
    if (sub == IR->func()->variableMap.end())
        return;

    ldc::DILocalVariable debugVariable = sub->second;
    if (!global.params.symdebug || !debugVariable)
        return;

    llvm::Instruction *instr = DBuilder.insertDbgValueIntrinsic(val, 0, debugVariable,
#if LDC_LLVM_VER >= 306
        DBuilder.createExpression(),
#endif
#if LDC_LLVM_VER >= 307
        IR->ir->getCurrentDebugLocation(),
#endif
        IR->scopebb());
    instr->setDebugLoc(IR->ir->getCurrentDebugLocation());
}

void ldc::DIBuilder::EmitLocalVariable(llvm::Value *ll, VarDeclaration *vd,
                                       Type *type, bool isThisPtr,
#if LDC_LLVM_VER >= 306
                                       llvm::ArrayRef<int64_t> addr
#else
                                       llvm::ArrayRef<llvm::Value *> addr
#endif
                           )
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf local variable");
    LOG_SCOPE;

    IrFunction::VariableMap& variableMap = IR->func()->variableMap;
    IrFunction::VariableMap::iterator sub = variableMap.find(vd);
    if (sub != variableMap.end())
        return; // ensure that the debug variable is created only once

    // get type description
    ldc::DIType TD = CreateTypeDescription(type ? type : vd->type, true);
    if (static_cast<llvm::MDNode *>(TD) == 0)
        return; // unsupported

    // get variable description
    assert(!vd->isDataseg() && "static variable");

    unsigned tag;
    if (vd->isParameter())
        tag = llvm::dwarf::DW_TAG_arg_variable;
    else
        tag = llvm::dwarf::DW_TAG_auto_variable;

    ldc::DILocalVariable debugVariable;
    unsigned Flags = 0;
    if (isThisPtr)
    {
#if LDC_LLVM_VER >= 302
        Flags |= DIFlags::FlagArtificial | DIFlags::FlagObjectPointer;
#else
        Flags |= DIFlags::FlagArtificial;
#endif
    }

#if LDC_LLVM_VER < 306
    if (addr.empty()) {
#endif
        debugVariable = DBuilder.createLocalVariable(
            tag, // tag
            GetCurrentScope(), // scope
            vd->toChars(), // name
            CreateFile(vd->loc), // file
            vd->loc.linnum, // line num
            TD, // type
            true, // preserve
            Flags // flags
        );
#if LDC_LLVM_VER < 306
    }
    else {
        debugVariable = DBuilder.createComplexVariable(
            tag, // tag
            GetCurrentScope(), // scope
            vd->toChars(), // name
            CreateFile(vd->loc), // file
            vd->loc.linnum, // line num
            TD, // type
            addr
        );
    }
#endif
    variableMap[vd] = debugVariable;

    // declare
#if LDC_LLVM_VER >= 306
    Declare(ll, debugVariable, addr.empty() ? DBuilder.createExpression() : DBuilder.createExpression(addr));
#else
    Declare(ll, debugVariable);
#endif
}

ldc::DIGlobalVariable ldc::DIBuilder::EmitGlobalVariable(llvm::GlobalVariable *ll, VarDeclaration *vd)
{
    if (!global.params.symdebug)
#if LDC_LLVM_VER >= 307
        return nullptr;
#else
        return llvm::DIGlobalVariable();
#endif

    Logger::println("D to dwarf global_variable");
    LOG_SCOPE;

    assert(vd->isDataseg() || (vd->storage_class & (STCconst | STCimmutable) && vd->init));

    return DBuilder.createGlobalVariable(
#if LDC_LLVM_VER >= 306
        GetCU(), // context
#endif
        vd->toChars(), // name
#if LDC_LLVM_VER >= 303
        mangle(vd), // linkage name
#endif
        CreateFile(vd->loc), // file
        vd->loc.linnum, // line num
        CreateTypeDescription(vd->type, false), // type
        vd->protection == PROTprivate, // is local to unit
        ll // value
    );
}

void ldc::DIBuilder::EmitModuleEnd()
{
    if (!global.params.symdebug)
        return;

    DBuilder.finalize();
}
