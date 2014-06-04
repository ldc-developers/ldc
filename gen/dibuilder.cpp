//===-- gen/dibuilder.h - Debug information builder -------------*- C++ -*-===//
//
//                         LDC � the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dibuilder.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/utils.h"
#include "ir/irtypeaggr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "enum.h"
#include "module.h"
#include "mtype.h"

////////////////////////////////////////////////////////////////////////////////

// get the module the symbol is in, or - for template instances - the current module
Module *ldc::DIBuilder::getDefinedModule(Dsymbol *s)
{
    // templates are defined in current module
    if (DtoIsTemplateInstance(s, true))
    {
        return IR->dmodule;
    }
    // array operations as well
    else if (FuncDeclaration* fd = s->isFuncDeclaration())
    {
        if (fd->isArrayOp == 1)
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

llvm::DIDescriptor ldc::DIBuilder::GetCurrentScope()
{
    IrFunction *fn = IR->func();
    if (fn->diLexicalBlocks.empty())
    {
        assert(static_cast<llvm::MDNode *>(fn->diSubprogram) != 0);
        return fn->diSubprogram;
    }
    return fn->diLexicalBlocks.top();
}

void ldc::DIBuilder::Declare(llvm::Value *var, llvm::DIVariable divar)
{
    llvm::Instruction *instr = DBuilder.insertDeclare(var, divar, IR->scopebb());
    instr->setDebugLoc(IR->ir->getCurrentDebugLocation());
}

llvm::DIFile ldc::DIBuilder::CreateFile(Loc loc)
{
    llvm::SmallString<128> path(loc.filename ? loc.filename : "");
    llvm::sys::fs::make_absolute(path);

    return DBuilder.createFile(
        llvm::sys::path::filename(path),
        llvm::sys::path::parent_path(path)
    );
}

llvm::DIType ldc::DIBuilder::CreateBasicType(Type *type)
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

llvm::DIType ldc::DIBuilder::CreateEnumType(Type *type)
{
    llvm::Type *T = DtoType(type);

    assert(type->ty == Tenum && "only enums allowed for debug info in dwarfEnumType");
    TypeEnum *te = static_cast<TypeEnum *>(type);
    llvm::SmallVector<llvm::Value *, 8> subscripts;
    for (ArrayIter<Dsymbol> it(te->sym->members); it.more(); it.next())
    {
        EnumMember *em = it->isEnumMember();
        llvm::StringRef Name(em->toChars());
        uint64_t Val = em->value->toInteger();
        llvm::Value *Subscript = DBuilder.createEnumerator(Name, Val);
        subscripts.push_back(Subscript);
    }

    llvm::StringRef Name = te->toChars();
    unsigned LineNumber = te->sym->loc.linnum;
    llvm::DIFile File = CreateFile(te->sym->loc);

    return DBuilder.createEnumerationType(
        llvm::DICompileUnit(GetCU()),
        Name,
        File,
        LineNumber,
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        DBuilder.getOrCreateArray(subscripts) // subscripts
#if LDC_LLVM_VER >= 302
        , CreateTypeDescription(te->sym->memtype, NULL)
#endif
    );
}

llvm::DIType ldc::DIBuilder::CreatePointerType(Type *type)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    assert(t->ty == Tpointer && "Only pointers allowed for debug info in DIBuilder::CreatePointerType");

    // find base type
    Type *nt = t->nextOf();
    llvm::DIType basetype = CreateTypeDescription(nt, NULL);

    return DBuilder.createPointerType(
        basetype,
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        type->toChars() // name
    );
}

llvm::DIType ldc::DIBuilder::CreateVectorType(Type *type)
{
    LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    assert(t->ty == Tvector && "Only vectors allowed for debug info in DIBuilder::CreateVectorType");
    TypeVector *tv = static_cast<TypeVector *>(t);
    Type *te = tv->elementType();
    int64_t Dim = tv->size(Loc()) / te->size(Loc());
    llvm::Value *subscripts[] =
    {
        DBuilder.getOrCreateSubrange(0, Dim)
    };
    llvm::DIType basetype = CreateTypeDescription(te, NULL);

    return DBuilder.createVectorType(
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        basetype, // element type
        DBuilder.getOrCreateArray(subscripts) // subscripts
    );
}

llvm::DIType ldc::DIBuilder::CreateMemberType(unsigned linnum, Type *type,
                                                llvm::DIFile file,
                                                const char* c_name,
                                                unsigned offset)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    // find base type
    llvm::DIType basetype = CreateTypeDescription(t, NULL, true);

    return DBuilder.createMemberType(
        llvm::DICompileUnit(GetCU()),
        c_name, // name
        file, // file
        linnum, // line number
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        offset*8, // offset (bits)
//FIXME: need flags?
        0, // flags
        basetype // derived from
    );
}

void ldc::DIBuilder::AddBaseFields(ClassDeclaration *sd, llvm::DIFile file,
                                     std::vector<llvm::Value*> &elems)
{
    if (sd->baseClass)
    {
        AddBaseFields(sd->baseClass, file, elems);
    }

    ArrayIter<VarDeclaration> it(sd->fields);
    size_t narr = sd->fields.dim;
    elems.reserve(narr);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();
        elems.push_back(CreateMemberType(vd->loc.linnum, vd->type, file, vd->toChars(), vd->offset));
    }
}

llvm::DIType ldc::DIBuilder::CreateCompositeType(Type *type)
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
    // const/� wrappers.
    LLType *T = DtoType(sd->type);
    IrTypeAggr *ir = sd->type->irtype->isAggr();
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
    std::vector<llvm::Value *> elems;

    // defaults
    llvm::StringRef name = sd->toChars();
    unsigned linnum = sd->loc.linnum;
    llvm::DICompileUnit CU(GetCU());
    assert(CU && CU.Verify() && "Compilation unit missing or corrupted");
    llvm::DIFile file = CreateFile(sd->loc);
    llvm::DIType derivedFrom;

    // set diCompositeType to handle recursive types properly
    unsigned tag = (t->ty == Tstruct) ? llvm::dwarf::DW_TAG_structure_type
                                        : llvm::dwarf::DW_TAG_class_type;
    ir->diCompositeType = DBuilder.createForwardDecl(tag, name,
#if LDC_LLVM_VER >= 302
                                                           CU,
#endif
                                                           file, linnum);

    if (!sd->isInterfaceDeclaration()) // plain interfaces don't have one
    {
        if (t->ty == Tstruct)
        {
            ArrayIter<VarDeclaration> it(sd->fields);
            size_t narr = sd->fields.dim;
            elems.reserve(narr);
            for (; !it.done(); it.next())
            {
                VarDeclaration* vd = it.get();
                llvm::DIType dt = CreateMemberType(vd->loc.linnum, vd->type, file, vd->toChars(), vd->offset);
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

    llvm::DIArray elemsArray = DBuilder.getOrCreateArray(elems);

    llvm::DIType ret;
    if (t->ty == Tclass) {
        ret = DBuilder.createClassType(
           CU, // compile unit where defined
           name, // name
           file, // file where defined
           linnum, // line number where defined
           getTypeBitSize(T), // size in bits
           getABITypeAlign(T)*8, // alignment in bits
           0, // offset in bits,
           llvm::DIType::FlagFwdDecl, // flags
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
           llvm::DIType::FlagFwdDecl, // flags
#if LDC_LLVM_VER >= 303
           derivedFrom, // DerivedFrom
#endif
           elemsArray
        );
    }

    ir->diCompositeType.replaceAllUsesWith(ret);
    ir->diCompositeType = ret;

    return ret;
}

llvm::DIType ldc::DIBuilder::CreateArrayType(Type *type)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    assert(t->ty == Tarray && "Only arrays allowed for debug info in DIBuilder::CreateArrayType");

    llvm::DIFile file = CreateFile(Loc(IR->dmodule, 0));

    llvm::Value *elems[] = {
        CreateMemberType(0, Type::tsize_t, file, "length", 0),
        CreateMemberType(0, t->nextOf()->pointerTo(), file, "ptr",
                         global.params.is64bit ? 8 : 4)
    };

    return DBuilder.createStructType
       (
        llvm::DICompileUnit(GetCU()),
        llvm::StringRef(), // Name TODO: Really no name for arrays? t->toChars()?
        file, // File
        0, // LineNo
        getTypeBitSize(T), // size in bits
        getABITypeAlign(T)*8, // alignment in bits
        0, // What here?
#if LDC_LLVM_VER >= 303
        llvm::DIType(), // DerivedFrom
#endif
        DBuilder.getOrCreateArray(elems)
    );
}

llvm::DIType ldc::DIBuilder::CreateSArrayType(Type *type)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    // find base type
    llvm::SmallVector<llvm::Value *, 8> subscripts;
    while (t->ty == Tsarray)
    {
        TypeSArray *tsa = static_cast<TypeSArray *>(t);
        int64_t Count = tsa->dim->toInteger();
        llvm::Value *subscript = DBuilder.getOrCreateSubrange(0, Count-1);
        subscripts.push_back(subscript);
        t = t->nextOf();
    }
    llvm::DIType basetype = CreateTypeDescription(t, NULL);

    return DBuilder.createArrayType(
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        basetype, // element type
        DBuilder.getOrCreateArray(subscripts) // subscripts
    );
}

llvm::DIType ldc::DIBuilder::CreateAArrayType(Type *type)
{
    // FIXME: Implement
#if LDC_LLVM_VER >= 304
    return DBuilder.createUnspecifiedType(type->toChars());
#else
    return llvm::DIType(NULL);
#endif
}

////////////////////////////////////////////////////////////////////////////////

ldc::DIFunctionType ldc::DIBuilder::CreateFunctionType(Type *type)
{
    TypeFunction *t = static_cast<TypeFunction*>(type);
    Type *retType = t->next;

    llvm::DIFile file = CreateFile(Loc(IR->dmodule, 0));

    // Create "dummy" subroutine type for the return type
    llvm::SmallVector<llvm::Value*, 16> Elts;
    Elts.push_back(CreateTypeDescription(retType, NULL, true));
    llvm::DIArray EltTypeArray = DBuilder.getOrCreateArray(Elts);
    return DBuilder.createSubroutineType(file, EltTypeArray);
}

ldc::DIFunctionType ldc::DIBuilder::CreateDelegateType(Type *type)
{
    // FIXME: Implement
    TypeDelegate *t = static_cast<TypeDelegate*>(type);

    llvm::DIFile file = CreateFile(Loc(IR->dmodule, 0));

    // Create "dummy" subroutine type for the return type
    llvm::SmallVector<llvm::Value*, 16> Elts;
    Elts.push_back(
#if LDC_LLVM_VER >= 304
        DBuilder.createUnspecifiedType(type->toChars())
#else
        llvm::DIType(NULL)
#endif
    );
    llvm::DIArray EltTypeArray = DBuilder.getOrCreateArray(Elts);
    return DBuilder.createSubroutineType(file, EltTypeArray);
}

////////////////////////////////////////////////////////////////////////////////

llvm::DIType ldc::DIBuilder::CreateTypeDescription(Type* type,
                                                   const char* c_name,
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
    CUNode =
#endif
    DBuilder.createCompileUnit(
        global.params.symdebug == 2 ? llvm::dwarf::DW_LANG_C
                                    : llvm::dwarf::DW_LANG_D,
        llvm::sys::path::filename(srcpath),
        llvm::sys::path::parent_path(srcpath),
        "LDC (http://wiki.dlang.org/LDC)",
        false, // isOptimized TODO
        llvm::StringRef(), // Flags TODO
        1 // Runtime Version TODO
    );
#if LDC_LLVM_VER < 304
    CUNode = DBuilder.getCU();
#endif
}

llvm::DISubprogram ldc::DIBuilder::EmitSubProgram(FuncDeclaration *fd)
{
    if (!global.params.symdebug)
        return llvm::DISubprogram();

    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    llvm::DICompileUnit CU(GetCU());
    assert(CU && CU.Verify() && "Compilation unit missing or corrupted in DIBuilder::EmitSubProgram");

    llvm::DIFile file = CreateFile(fd->loc);

    // Create subroutine type
    ldc::DIFunctionType DIFnType = CreateFunctionType(static_cast<TypeFunction*>(fd->type));

    // FIXME: duplicates ?
    return DBuilder.createFunction(
        CU, // context
        fd->toPrettyChars(), // name
        fd->mangle(), // linkage name
        file, // file
        fd->loc.linnum, // line no
        DIFnType, // type
        fd->protection == PROTprivate, // is local to unit
        IR->dmodule == getDefinedModule(fd), // isdefinition
        fd->loc.linnum, // FIXME: scope line
        0, // Flags
        false, // isOptimized
        fd->ir.irFunc->func
    );
}

llvm::DISubprogram ldc::DIBuilder::EmitSubProgramInternal(llvm::StringRef prettyname,
                                                            llvm::StringRef mangledname)
{
    if (!global.params.symdebug)
        return llvm::DISubprogram();

    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    llvm::DIFile file(CreateFile(Loc(IR->dmodule, 0)));

    // Create "dummy" subroutine type for the return type
    llvm::SmallVector<llvm::Value *, 1> Elts;
#if LDC_LLVM_VER >= 304
    Elts.push_back(DBuilder.createUnspecifiedType(prettyname));
#else
    Elts.push_back(llvm::DIType(NULL));
#endif
    llvm::DIArray EltTypeArray = DBuilder.getOrCreateArray(Elts);
    ldc::DIFunctionType DIFnType = DBuilder.createSubroutineType(file, EltTypeArray);

    // FIXME: duplicates ?
    return DBuilder.createFunction(
        llvm::DIDescriptor(file), // context
        prettyname, // name
        mangledname, // linkage name
        file, // file
        0, // line no
        DIFnType, // return type. TODO: fill it up
        true, // is local to unit
        true, // isdefinition
        0 // FIXME: scope line
    );
}

void ldc::DIBuilder::EmitFuncStart(FuncDeclaration *fd)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf funcstart");
    LOG_SCOPE;

    assert(static_cast<llvm::MDNode *>(fd->ir.irFunc->diSubprogram) != 0);
    EmitStopPoint(fd->loc.linnum);
}

void ldc::DIBuilder::EmitFuncEnd(FuncDeclaration *fd)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf funcend");
    LOG_SCOPE;

    assert(static_cast<llvm::MDNode *>(fd->ir.irFunc->diSubprogram) != 0);
}

void ldc::DIBuilder::EmitBlockStart(Loc loc)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf block start");
    LOG_SCOPE;

#if LDC_LLVM_VER >= 305
    llvm::DILexicalBlock block = DBuilder.createLexicalBlock(
            GetCurrentScope(), // scope
            CreateFile(loc), // file
            loc.linnum, // line
            0, // column
            0 // DWARF path discriminator value
            );
#else
    llvm::DILexicalBlock block = DBuilder.createLexicalBlock(
            GetCurrentScope(), // scope
            CreateFile(loc), // file
            loc.linnum, // line
            0 // column
            );
#endif
    IR->func()->diLexicalBlocks.push(block);
    EmitStopPoint(loc.linnum);
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

void ldc::DIBuilder::EmitStopPoint(unsigned ln)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf stoppoint at line %u", ln);
    LOG_SCOPE;
    llvm::DebugLoc loc = llvm::DebugLoc::get(ln, 0, GetCurrentScope());
    IR->ir->SetCurrentDebugLocation(loc);
}

void ldc::DIBuilder::EmitValue(llvm::Value *val, VarDeclaration *vd)
{
    if (!global.params.symdebug || !vd->debugVariable)
        return;

    llvm::Instruction *instr = DBuilder.insertDbgValueIntrinsic(val, 0, vd->debugVariable, IR->scopebb());
    instr->setDebugLoc(IR->ir->getCurrentDebugLocation());
}

void ldc::DIBuilder::EmitLocalVariable(llvm::Value *ll, VarDeclaration *vd,
                           llvm::ArrayRef<llvm::Value *> addr)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf local variable");
    LOG_SCOPE;

    if (IR->func()->diSubprogram == vd->debugFunc) // ensure that the debug variable is created only once
        return;

    // get type description
    llvm::DIType TD = CreateTypeDescription(vd->type, NULL, true);
    if (static_cast<llvm::MDNode *>(TD) == 0)
        return; // unsupported

    // get variable description
    assert(!vd->isDataseg() && "static variable");

    unsigned tag;
    if (vd->isParameter())
        tag = llvm::dwarf::DW_TAG_arg_variable;
    else
        tag = llvm::dwarf::DW_TAG_auto_variable;

    if (addr.empty()) {
        vd->debugVariable = DBuilder.createLocalVariable(
            tag, // tag
            GetCurrentScope(), // scope
            vd->toChars(), // name
            CreateFile(vd->loc), // file
            vd->loc.linnum, // line num
            TD, // type
            true // preserve
        );
    } else {
        vd->debugVariable = DBuilder.createComplexVariable(
            tag, // tag
            GetCurrentScope(), // scope
            vd->toChars(), // name
            CreateFile(vd->loc), // file
            vd->loc.linnum, // line num
            TD, // type
            addr
        );
    }
    vd->debugFunc = IR->func()->diSubprogram;

    // declare
    Declare(ll, vd->debugVariable);
}

llvm::DIGlobalVariable ldc::DIBuilder::EmitGlobalVariable(llvm::GlobalVariable *ll, VarDeclaration *vd)
{
    if (!global.params.symdebug)
        return llvm::DIGlobalVariable();

    Logger::println("D to dwarf global_variable");
    LOG_SCOPE;

    assert(vd->isDataseg() || (vd->storage_class & (STCconst | STCimmutable) && vd->init));

    return DBuilder.createGlobalVariable(
        vd->toChars(), // name
#if LDC_LLVM_VER >= 303
        vd->mangle(), // linkage name
#endif
        CreateFile(vd->loc), // file
        vd->loc.linnum, // line num
        CreateTypeDescription(vd->type, NULL), // type
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
