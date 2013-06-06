//===-- todebug.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/todebug.h"
#include "arraytypes.h"
#include "declaration.h"
#include "dsymbol.h"
#include "enum.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/utils.h"
#include "ir/irmodule.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm::dwarf;

//////////////////////////////////////////////////////////////////////////////////////////////////

// get the module the symbol is in, or - for template instances - the current module
static Module* getDefinedModule(Dsymbol* s)
{
    // templates are defined in current module
    if (DtoIsTemplateInstance(s, true))
    {
        return gIR->dmodule;
    }
    // array operations as well
    else if (FuncDeclaration* fd = s->isFuncDeclaration())
    {
        if (fd->isArrayOp == 1)
            return gIR->dmodule;
    }
    // otherwise use the symbol's module
    return s->getModule();
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIDescriptor getCurrentScope()
{
    IrFunction *fn = gIR->func();
    if (fn->diLexicalBlocks.empty()) {
        assert(static_cast<llvm::MDNode*>(fn->diSubprogram) != 0);
        return fn->diSubprogram;
    }
    return fn->diLexicalBlocks.top();
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfTypeDescription_impl(Type* type, const char* c_name);
static llvm::DIType dwarfTypeDescription(Type* type, const char* c_name);

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIFile DtoDwarfFile(Loc loc)
{
    llvm::SmallString<128> path(loc.filename ? loc.filename : "");
    llvm::sys::fs::make_absolute(path);

    return gIR->dibuilder.createFile(
        llvm::sys::path::filename(path),
        llvm::sys::path::parent_path(path)
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfBasicType(Type* type)
{
    Type* t = type->toBasetype();
    LLType* T = DtoType(type);

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
        llvm_unreachable("Unsupported basic type for debug info");
    }

    return gIR->dibuilder.createBasicType(
        type->toChars(), // name
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        Encoding
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfEnumType(Type *type)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    // FIXME

    return llvm::DIType(NULL);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfPointerType(Type* type)
{
    LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    assert(t->ty == Tpointer && "only pointers allowed for debug info in dwarfPointerType");

    // find base type
    llvm::DIType basetype;
    Type* nt = t->nextOf();
    basetype = dwarfTypeDescription_impl(nt, NULL);
    if (nt->ty == Tvoid)
        basetype = llvm::DIType(NULL);

    return gIR->dibuilder.createPointerType(
        basetype,
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        type->toChars() // name
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfVectorType(Type* type)
{
    LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    assert(t->ty == Tvector && "only vectors allowed for debug info in dwarfVectorType");
    TypeVector *tv = static_cast<TypeVector *>(t);
    Type *te = tv->elementType();
    int64_t Dim = tv->size(Loc(0)) / te->size(Loc(0));
    llvm::Value *subscripts[] =
    {
        gIR->dibuilder.getOrCreateSubrange(0, Dim)
    };
    llvm::DIType basetype = dwarfTypeDescription_impl(te, NULL);

    return gIR->dibuilder.createVectorType(
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        basetype, // element type
        gIR->dibuilder.getOrCreateArray(subscripts) // subscripts
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfMemberType(unsigned linnum, Type* type, llvm::DIFile file, const char* c_name, unsigned offset)
{
    LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    // find base type
    llvm::DIType basetype;
    basetype = dwarfTypeDescription(t, NULL);
    if (t->ty == Tvoid)
        basetype = llvm::DIType(NULL);

    return gIR->dibuilder.createMemberType(
        llvm::DIDescriptor(file),
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

//////////////////////////////////////////////////////////////////////////////////////////////////

static void add_base_fields(
    ClassDeclaration* sd,
    llvm::DIFile file,
    std::vector<llvm::Value*>& elems)
{
    if (sd->baseClass)
    {
        add_base_fields(sd->baseClass, file, elems);
    }

    ArrayIter<VarDeclaration> it(sd->fields);
    size_t narr = sd->fields.dim;
    elems.reserve(narr);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();
        elems.push_back(dwarfMemberType(vd->loc.linnum, vd->type, file, vd->toChars(), vd->offset));
    }
}


static llvm::DIType dwarfCompositeType(Type* type)
{
    LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    assert((t->ty == Tstruct || t->ty == Tclass) &&
           "unsupported type for dwarfCompositeType");
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

    // make sure it's resolved
    sd->codegen(Type::sir);

    // if we don't know the aggregate's size, we don't know enough about it
    // to provide debug info. probably a forward-declared struct?
    if (sd->sizeok == 0)
        return llvm::DICompositeType(NULL);

    IrStruct* ir = sd->ir.irStruct;
    assert(ir);
    if (static_cast<llvm::MDNode*>(ir->diCompositeType) != 0)
        return ir->diCompositeType;

    // elements
    std::vector<llvm::Value*> elems;

    // defaults
    llvm::StringRef name = sd->toChars();
    unsigned linnum = sd->loc.linnum;
    llvm::DIFile file = DtoDwarfFile(sd->loc);
    llvm::DIType derivedFrom;

    // set diCompositeType to handle recursive types properly
    if (!ir->diCompositeType) {
        unsigned tag = (t->ty == Tstruct) ? llvm::dwarf::DW_TAG_structure_type
                                          : llvm::dwarf::DW_TAG_class_type;
        ir->diCompositeType = gIR->dibuilder.createForwardDecl(tag, name,
#if LDC_LLVM_VER >= 302
                                                               llvm::DIDescriptor(file),
#endif
                                                               file, linnum);
    }

    if (!ir->aggrdecl->isInterfaceDeclaration()) // plain interfaces don't have one
    {
        if (t->ty == Tstruct)
        {
            ArrayIter<VarDeclaration> it(sd->fields);
            size_t narr = sd->fields.dim;
            elems.reserve(narr);
            for (; !it.done(); it.next())
            {
                VarDeclaration* vd = it.get();
                llvm::DIType dt = dwarfMemberType(vd->loc.linnum, vd->type, file, vd->toChars(), vd->offset);
                elems.push_back(dt);
            }
        }
        else
        {
            ClassDeclaration *classDecl = ir->aggrdecl->isClassDeclaration();
            add_base_fields(classDecl, file, elems);
            if (classDecl->baseClass)
                derivedFrom = dwarfCompositeType(classDecl->baseClass->getType());
        }
    }

    llvm::DIArray elemsArray = gIR->dibuilder.getOrCreateArray(elems);

    llvm::DIType ret;
    if (t->ty == Tclass) {
        ret = gIR->dibuilder.createClassType(
           llvm::DIDescriptor(file),
           name, // name
           file, // compile unit where defined
           linnum, // line number where defined
           getTypeBitSize(T), // size in bits
           getABITypeAlign(T)*8, // alignment in bits
           0, // offset in bits,
           llvm::DIType::FlagFwdDecl, // flags
           derivedFrom, // DerivedFrom
           elemsArray
        );
    } else {
        ret = gIR->dibuilder.createStructType(
           llvm::DIDescriptor(file),
           name, // name
           file, // compile unit where defined
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

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIGlobalVariable dwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd)
{
    assert(vd->isDataseg() || (vd->storage_class & (STCconst | STCimmutable) && vd->init));

    return gIR->dibuilder.createGlobalVariable(
        vd->toChars(), // name
#if LDC_LLVM_VER >= 303
        vd->mangle(), // linkage name
#endif
        DtoDwarfFile(vd->loc), // file
        vd->loc.linnum, // line num
        dwarfTypeDescription_impl(vd->type, NULL), // type
        vd->protection == PROTprivate, // is local to unit
        ll // value
    );
}


//////////////////////////////////////////////////////////////////////////////////////////////////

static void dwarfDeclare(LLValue* var, llvm::DIVariable divar)
{
    llvm::Instruction *instr = gIR->dibuilder.insertDeclare(var, divar, gIR->scopebb());
    instr->setDebugLoc(gIR->ir->getCurrentDebugLocation());
}

//////////////////////////////////////////////////////////////////////////////////////////////////


static llvm::DIType dwarfArrayType(Type* type) {
    LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    llvm::DIFile file = DtoDwarfFile(Loc(gIR->dmodule, 0));

    llvm::Value* elems[] = {
        dwarfMemberType(0, Type::tsize_t, file, "length", 0),
        dwarfMemberType(0, t->nextOf()->pointerTo(), file, "ptr",
                        global.params.is64bit ? 8 : 4)
    };

    return gIR->dibuilder.createStructType
       (
        llvm::DIDescriptor(file),
        llvm::StringRef(), // Name TODO: Really no name for arrays?
        file, // File
        0, // LineNo
        getTypeBitSize(T), // size in bits
        getABITypeAlign(T)*8, // alignment in bits
        0, // What here?
#if LDC_LLVM_VER >= 303
        llvm::DIType(), // DerivedFrom
#endif
        gIR->dibuilder.getOrCreateArray(elems)
    );

}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfSArrayType(Type *type)
{
    llvm::Type *T = DtoType(type);
    Type *t = type->toBasetype();

    // find base type
    llvm::SmallVector<llvm::Value *, 8> subscripts;
    while (t->ty == Tsarray)
    {
        TypeSArray *tsa = static_cast<TypeSArray *>(t);
        int64_t Count = tsa->dim->toInteger();
        llvm::Value *subscript = gIR->dibuilder.getOrCreateSubrange(0, Count-1);
        subscripts.push_back(subscript);
        t = t->nextOf();
    }
    llvm::DIType basetype = dwarfTypeDescription_impl(t, NULL);

    return gIR->dibuilder.createArrayType(
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        basetype, // element type
        gIR->dibuilder.getOrCreateArray(subscripts) // subscripts
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfTypeDescription_impl(Type* type, const char* c_name)
{
    Type* t = type->toBasetype();
    if (t->ty == Tvoid)
        return llvm::DIType(NULL);
    else if (t->ty == Tvector)
        return dwarfVectorType(type);
    else if (t->ty == Tenum)
        return dwarfEnumType(type);
    else if (t->isintegral() || t->isfloating())
        return dwarfBasicType(type);
    else if (t->ty == Tpointer)
        return dwarfPointerType(type);
    else if (t->ty == Tarray)
        return dwarfArrayType(type);
    else if (t->ty == Tsarray)
        return dwarfSArrayType(type);
    else if (t->ty == Tstruct || t->ty == Tclass)
        return dwarfCompositeType(type);

    return llvm::DIType(NULL);
}

static llvm::DIType dwarfTypeDescription(Type* type, const char* c_name)
{
    Type* t = type->toBasetype();
    if (t->ty == Tclass)
        return dwarfTypeDescription_impl(type->pointerTo(), c_name);
    else
        return dwarfTypeDescription_impl(type, c_name);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfLocalVariable(LLValue* ll, VarDeclaration* vd, llvm::ArrayRef<LLValue*> addr)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf local variable");
    LOG_SCOPE;

    if (gIR->func()->diSubprogram == vd->debugFunc) // ensure that the debug variable is created only once
        return;

    // get type description
    llvm::DIType TD = dwarfTypeDescription(vd->type, NULL);
    if (static_cast<llvm::MDNode*>(TD) == 0)
        return; // unsupported

    // get variable description
    assert(!vd->isDataseg() && "static variable");

    unsigned tag;
    if (vd->isParameter())
        tag = DW_TAG_arg_variable;
    else
        tag = DW_TAG_auto_variable;

    if (addr.empty()) {
        vd->debugVariable = gIR->dibuilder.createLocalVariable(
            tag, // tag
            getCurrentScope(), // scope
            vd->toChars(), // name
            DtoDwarfFile(vd->loc), // file
            vd->loc.linnum, // line num
            TD, // type
            true // preserve
        );
    } else {
        vd->debugVariable = gIR->dibuilder.createComplexVariable(
            tag, // tag
            getCurrentScope(), // scope
            vd->toChars(), // name
            DtoDwarfFile(vd->loc), // file
            vd->loc.linnum, // line num
            TD, // type
            addr
        );
    }
    vd->debugFunc = gIR->func()->diSubprogram;

    // declare
    dwarfDeclare(ll, vd->debugVariable);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfCompileUnit(Module* m)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf compile_unit");
    LOG_SCOPE;

    // prepare srcpath
    const char *srcname = m->srcfile->name->toChars();
    std::string srcpath(FileName::path(srcname));
    if (!FileName::absolute(srcpath.c_str())) {
        llvm::sys::Path tmp = llvm::sys::Path::GetCurrentDirectory();
        tmp.appendComponent(srcpath);
        srcpath = tmp.str();
        if (!srcpath.empty() && *srcpath.rbegin() != '/' && *srcpath.rbegin() != '\\')
            srcpath = srcpath + '/';
    } else {
        srcname = FileName::name(srcname);
    }

    gIR->dibuilder.createCompileUnit(
        global.params.symdebug == 2 ? DW_LANG_C : DW_LANG_D,
        srcname,
        srcpath,
        "LDC (http://wiki.dlang.org/LDC)",
        false, // isOptimized TODO
        llvm::StringRef(), // Flags TODO
        1 // Runtime Version TODO
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DISubprogram DtoDwarfSubProgram(FuncDeclaration* fd)
{
    if (!global.params.symdebug)
        return llvm::DISubprogram();

    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    llvm::DICompileUnit CU(gIR->dibuilder.getCU());
    assert(CU && CU.Verify() && "Compilation unit missing or corrupted");

    llvm::DIFile file = DtoDwarfFile(fd->loc);
    Type *retType = static_cast<TypeFunction*>(fd->type)->next;

    // Create "dummy" subroutine type for the return type
    llvm::SmallVector<llvm::Value*, 16> Elts;
    Elts.push_back(dwarfTypeDescription(retType, NULL));
    llvm::DIArray EltTypeArray = gIR->dibuilder.getOrCreateArray(Elts);
#if LDC_LLVM_VER >= 304
    llvm::DICompositeType
#else
    llvm::DIType 
#endif
             DIFnType = gIR->dibuilder.createSubroutineType(file, EltTypeArray);

    // FIXME: duplicates ?
    return gIR->dibuilder.createFunction(
        CU, // context
        fd->toPrettyChars(), // name
        fd->mangle(), // linkage name
        file, // file
        fd->loc.linnum, // line no
        DIFnType, // type
        fd->protection == PROTprivate, // is local to unit
        gIR->dmodule == getDefinedModule(fd), // isdefinition
        fd->loc.linnum, // FIXME: scope line
        0, // Flags
        false, // isOptimized
        fd->ir.irFunc->func
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DISubprogram DtoDwarfSubProgramInternal(const char* prettyname, const char* mangledname)
{
    if (!global.params.symdebug)
        return llvm::DISubprogram();

    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    llvm::DIFile file(DtoDwarfFile(Loc(gIR->dmodule, 0)));

    // Create "dummy" subroutine type for the return type
    llvm::SmallVector<llvm::Value*, 1> Elts;
    Elts.push_back(llvm::DIType(NULL));
    llvm::DIArray EltTypeArray = gIR->dibuilder.getOrCreateArray(Elts);
#if LDC_LLVM_VER >= 304
    llvm::DICompositeType 
#else
    llvm::DIType
#endif
             DIFnType = gIR->dibuilder.createSubroutineType(file, EltTypeArray);

    // FIXME: duplicates ?
    return gIR->dibuilder.createFunction(
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

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DIGlobalVariable DtoDwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd)
{
    if (!global.params.symdebug)
        return llvm::DIGlobalVariable();

    Logger::println("D to dwarf global_variable");
    LOG_SCOPE;

    // FIXME: duplicates ?
    return dwarfGlobalVariable(ll, vd);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncStart(FuncDeclaration* fd)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf funcstart");
    LOG_SCOPE;

    assert(static_cast<llvm::MDNode*>(fd->ir.irFunc->diSubprogram) != 0);
    DtoDwarfStopPoint(fd->loc.linnum);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncEnd(FuncDeclaration* fd)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf funcend");
    LOG_SCOPE;

    assert(static_cast<llvm::MDNode*>(fd->ir.irFunc->diSubprogram) != 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfBlockStart(Loc loc)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf block start");
    LOG_SCOPE;

    llvm::DILexicalBlock block = gIR->dibuilder.createLexicalBlock(
            getCurrentScope(), // scope
            DtoDwarfFile(loc), // file
            loc.linnum, // line
            0 // column
            );
    gIR->func()->diLexicalBlocks.push(block);
    DtoDwarfStopPoint(loc.linnum);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfBlockEnd()
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf block end");
    LOG_SCOPE;

    IrFunction *fn = gIR->func();
    assert(!fn->diLexicalBlocks.empty());
    fn->diLexicalBlocks.pop();
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfStopPoint(unsigned ln)
{
    if (!global.params.symdebug)
        return;

    Logger::println("D to dwarf stoppoint at line %u", ln);
    LOG_SCOPE;
    llvm::DebugLoc loc = llvm::DebugLoc::get(ln, 0, getCurrentScope());
    gIR->ir->SetCurrentDebugLocation(loc);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfValue(LLValue *val, VarDeclaration* vd)
{
    if (!global.params.symdebug || !vd->debugVariable)
        return;

    llvm::Instruction *instr = gIR->dibuilder.insertDbgValueIntrinsic(val, 0, vd->debugVariable, gIR->scopebb());
    instr->setDebugLoc(gIR->ir->getCurrentDebugLocation());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfModuleEnd()
{
    if (!global.params.symdebug)
        return;

    gIR->dibuilder.finalize();
}
