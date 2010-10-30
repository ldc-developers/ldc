#include "gen/llvm.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/System/Path.h"

#include "declaration.h"
#include "module.h"
#include "mars.h"

#include "gen/todebug.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/logger.h"
#include "gen/llvmhelpers.h"
#include "gen/linkage.h"
#include "gen/utils.h"

#include "ir/irmodule.h"

using namespace llvm::dwarf;

#ifndef DISABLE_DEBUG_INFO

//////////////////////////////////////////////////////////////////////////////////////////////////

// get the module the symbol is in, or - for template instances - the current module
static Module* getDefinedModule(Dsymbol* s)
{
    // templates are defined in current module
    if (DtoIsTemplateInstance(s))
    {
        return gIR->dmodule;
    }
    // array operations as well
    else if (FuncDeclaration* fd = s->isFuncDeclaration())
    {
        if (fd->isArrayOp)
            return gIR->dmodule;
    }
    // otherwise use the symbol's module
    return s->getModule();
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfTypeDescription_impl(Type* type, llvm::DICompileUnit cu, const char* c_name);
static llvm::DIType dwarfTypeDescription(Type* type, llvm::DICompileUnit cu, const char* c_name);

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DIFile DtoDwarfFile(Loc loc, llvm::DICompileUnit compileUnit)
{
    llvm::StringRef path;
    llvm::StringRef name;
    llvm::StringRef dir;
    if (loc.filename) {
        path = llvm::StringRef(loc.filename);
        name = llvm::StringRef(basename(loc.filename));
        dir = path.substr(0, path.size() - name.size());
    }
    return gIR->difactory.CreateFile(name, dir, compileUnit);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIBasicType dwarfBasicType(Type* type, llvm::DICompileUnit compileUnit)
{
    Type* t = type->toBasetype();
    const LLType* T = DtoType(type);

    // find encoding
    unsigned id;
    if (t->isintegral())
    {
        if (type->isunsigned())
            id = DW_ATE_unsigned;
        else
            id = DW_ATE_signed;
    }
    else if (t->isfloating())
    {
        id = DW_ATE_float;
    }
    else
    {
        assert(0 && "unsupported basictype for debug info");
    }

    return gIR->difactory.CreateBasicType(
        compileUnit, // context
        type->toChars(), // name
        DtoDwarfFile(Loc(gIR->dmodule, 0), DtoDwarfCompileUnit(gIR->dmodule)), // file
        0, // line number
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        0, // offset (bits)
//FIXME: need flags?
        0, // flags
        id // encoding
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIDerivedType dwarfDerivedType(Type* type, llvm::DICompileUnit compileUnit)
{
    const LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    assert(t->ty == Tpointer && "unsupported derivedtype for debug info, only pointers allowed");

    // find base type
    llvm::DIType basetype;
    Type* nt = t->nextOf();
    basetype = dwarfTypeDescription_impl(nt, compileUnit, NULL);
    if (nt->ty == Tvoid)
        basetype = llvm::DIType(NULL);

    return gIR->difactory.CreateDerivedType(
        DW_TAG_pointer_type, // tag
        compileUnit, // context
        "", // name
        DtoDwarfFile(Loc(gIR->dmodule, 0), DtoDwarfCompileUnit(gIR->dmodule)), // file
        0, // line number
        getTypeBitSize(T), // size (bits)
        getABITypeAlign(T)*8, // align (bits)
        0, // offset (bits)
//FIXME: need flags?
        0, // flags
        basetype // derived from
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIDerivedType dwarfMemberType(unsigned linnum, Type* type, llvm::DICompileUnit compileUnit, llvm::DIFile file, const char* c_name, unsigned offset)
{
    const LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    // find base type
    llvm::DIType basetype;
    basetype = dwarfTypeDescription(t, compileUnit, NULL);
    if (t->ty == Tvoid)
        basetype = llvm::DIType(NULL);

    return gIR->difactory.CreateDerivedType(
        DW_TAG_member, // tag
        compileUnit, // context
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
    llvm::DICompileUnit compileUnit,
    llvm::DIFile file,
    std::vector<llvm::DIDescriptor>& elems)
{
    if (sd->baseClass)
    {
        add_base_fields(sd->baseClass, compileUnit, file, elems);
    }

    ArrayIter<VarDeclaration> it(sd->fields);
    size_t narr = sd->fields.dim;
    elems.reserve(narr);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();
        elems.push_back(dwarfMemberType(vd->loc.linnum, vd->type, compileUnit, file, vd->toChars(), vd->offset));
    }
}



//FIXME: This does not use llvm's DIFactory as it can't
//   handle recursive types properly.
static llvm::DICompositeType dwarfCompositeType(Type* type, llvm::DICompileUnit compileUnit)
{
    const LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    // defaults
    llvm::StringRef name;
    unsigned linnum = 0;
    llvm::DIFile file;

    // prepare tag and members
    unsigned tag;

    // elements
    std::vector<llvm::DIDescriptor> elems;

    // pointer to ir->diCompositeType
    llvm::DICompositeType *diCompositeType = 0;

    llvm::DICompositeType derivedFrom;

    // dynamic array
    if (t->ty == Tarray)
    {
        tag = DW_TAG_structure_type;
        elems.push_back(dwarfMemberType(0, Type::tsize_t, compileUnit, file, "length", 0));
        elems.push_back(dwarfMemberType(0, t->nextOf()->pointerTo(), compileUnit, file, "ptr", global.params.is64bit?8:4));
        file = DtoDwarfFile(Loc(gIR->dmodule, 0), DtoDwarfCompileUnit(gIR->dmodule));
    }

    // struct/class
    else if (t->ty == Tstruct || t->ty == Tclass)
    {
        AggregateDeclaration* sd;
        if (t->ty == Tstruct)
        {
            TypeStruct* ts = (TypeStruct*)t;
            sd = ts->sym;
            tag = DW_TAG_structure_type;
        }
        else
        {
            TypeClass* tc = (TypeClass*)t;
            sd = tc->sym;
            tag = DW_TAG_class_type;
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
        if ((llvm::MDNode*)ir->diCompositeType != 0)
            return ir->diCompositeType;

        diCompositeType = &ir->diCompositeType;
        name = sd->toChars();
        linnum = sd->loc.linnum;
        file = DtoDwarfFile(sd->loc, DtoDwarfCompileUnit(getDefinedModule(sd)));
        // set diCompositeType to handle recursive types properly
        ir->diCompositeType = gIR->difactory.CreateCompositeTypeEx(
                                   tag, // tag
                                   compileUnit, // context
                                   name, // name
                                   file, // compile unit where defined
                                   linnum, // line number where defined
                                   LLConstantInt::get(LLType::getInt64Ty(gIR->context()), getTypeBitSize(T), false), // size in bits
                                   LLConstantInt::get(LLType::getInt64Ty(gIR->context()), getABITypeAlign(T)*8, false), // alignment in bits
                                   LLConstantInt::get(LLType::getInt64Ty(gIR->context()), 0, false), // offset in bits,
                                   0, // flags
                                   derivedFrom, // DerivedFrom
                                   llvm::DIArray(0)
                                   );

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
                    llvm::DIDerivedType dt = dwarfMemberType(vd->loc.linnum, vd->type, compileUnit, file, vd->toChars(), vd->offset);
                    elems.push_back(dt);
                }
            }
            else
            {
                ClassDeclaration *classDecl = ir->aggrdecl->isClassDeclaration();
                add_base_fields(classDecl, compileUnit, file, elems);
                if (classDecl->baseClass)
                    derivedFrom = dwarfCompositeType(classDecl->baseClass->getType(), compileUnit);
            }
        }
    }

    // unsupported composite type
    else
    {
        assert(0 && "unsupported compositetype for debug info");
    }

    llvm::DIArray elemsArray =
        gIR->difactory.GetOrCreateArray(elems.data(), elems.size());

    llvm::DICompositeType ret = gIR->difactory.CreateCompositeTypeEx(
                                    tag, // tag
                                    compileUnit, // context
                                    name, // name
                                    file, // compile unit where defined
                                    linnum, // line number where defined
                                    LLConstantInt::get(LLType::getInt64Ty(gIR->context()), getTypeBitSize(T), false), // size in bits
                                    LLConstantInt::get(LLType::getInt64Ty(gIR->context()), getABITypeAlign(T)*8, false), // alignment in bits
                                    LLConstantInt::get(LLType::getInt64Ty(gIR->context()), 0, false), // offset in bits,
                                    0, // flags
                                    derivedFrom, // DerivedFrom
                                    elemsArray);
    if (diCompositeType)
        *diCompositeType = ret;
    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIGlobalVariable dwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd)
{
#if DMDV2
    assert(vd->isDataseg() || (vd->storage_class & (STCconst | STCimmutable) && vd->init));
#else
    assert(vd->isDataseg());
#endif
    llvm::DICompileUnit compileUnit = DtoDwarfCompileUnit(gIR->dmodule);

    return gIR->difactory.CreateGlobalVariable(
        compileUnit, // context
        vd->mangle(), // name
        vd->toPrettyChars(), // displayname
        vd->toChars(), // linkage name
        DtoDwarfFile(vd->loc, DtoDwarfCompileUnit(getDefinedModule(vd))), // file
        vd->loc.linnum, // line num
        dwarfTypeDescription_impl(vd->type, compileUnit, NULL), // type
        vd->protection == PROTprivate, // is local to unit
        getDefinedModule(vd) == gIR->dmodule, // is definition
        ll // value
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIVariable dwarfVariable(VarDeclaration* vd, llvm::DIType type)
{
    assert(!vd->isDataseg() && "static variable");

    unsigned tag;
    if (vd->isParameter())
        tag = DW_TAG_arg_variable;
    else
        tag = DW_TAG_auto_variable;

    return gIR->difactory.CreateVariable(
        tag, // tag
        gIR->func()->diSubprogram, // context
        vd->toChars(), // name
        DtoDwarfFile(vd->loc, DtoDwarfCompileUnit(getDefinedModule(vd))), // file
        vd->loc.linnum, // line num
        type // type
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static void dwarfDeclare(LLValue* var, llvm::DIVariable divar)
{
    gIR->difactory.InsertDeclare(var, divar, gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIType dwarfTypeDescription_impl(Type* type, llvm::DICompileUnit cu, const char* c_name)
{
    Type* t = type->toBasetype();
    if (t->ty == Tvoid)
        return llvm::DIType(NULL);
    else if (t->isintegral() || t->isfloating())
        return dwarfBasicType(type, cu);
    else if (t->ty == Tpointer)
        return dwarfDerivedType(type, cu);
    else if (t->ty == Tarray || t->ty == Tstruct || t->ty == Tclass)
        return dwarfCompositeType(type, cu);

    return llvm::DIType(NULL);
}

static llvm::DIType dwarfTypeDescription(Type* type, llvm::DICompileUnit cu, const char* c_name)
{
    Type* t = type->toBasetype();
    if (t->ty == Tclass)
        return dwarfTypeDescription_impl(type->pointerTo(), cu, c_name);
    else
        return dwarfTypeDescription_impl(type, cu, c_name);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfLocalVariable(LLValue* ll, VarDeclaration* vd)
{
    Logger::println("D to dwarf local variable");
    LOG_SCOPE;

    // get compile units
    llvm::DICompileUnit thisCU = DtoDwarfCompileUnit(gIR->dmodule);
    llvm::DICompileUnit varCU = DtoDwarfCompileUnit(getDefinedModule(vd));

    // get type description
    llvm::DIType TD = dwarfTypeDescription(vd->type, thisCU, NULL);
    if ((llvm::MDNode*)TD == 0)
        return; // unsupported

    // get variable description
    llvm::DIVariable VD = dwarfVariable(vd, TD);

    // declare
    dwarfDeclare(ll, VD);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DICompileUnit DtoDwarfCompileUnit(Module* m)
{
    Logger::println("D to dwarf compile_unit");
    LOG_SCOPE;

    // we might be generating for an import
    IrModule* irmod = getIrModule(m);

    if ((llvm::MDNode*)irmod->diCompileUnit != 0)
    {
        //assert (irmod->diCompileUnit.getGV()->getParent() == gIR->module
        //   && "debug info compile unit belongs to incorrect llvm module!");
        return irmod->diCompileUnit;
    }

    // prepare srcpath
    std::string srcpath(FileName::path(m->srcfile->name->toChars()));
    if (!FileName::absolute(srcpath.c_str())) {
        llvm::sys::Path tmp = llvm::sys::Path::GetCurrentDirectory();
        tmp.appendComponent(srcpath);
        srcpath = tmp.str();
        if (!srcpath.empty() && *srcpath.rbegin() != '/' && *srcpath.rbegin() != '\\')
            srcpath = srcpath + '/';
    }

    // make compile unit
    irmod->diCompileUnit = gIR->difactory.CreateCompileUnit(
        global.params.symdebug == 2 ? DW_LANG_C : DW_LANG_D,
        m->srcfile->name->toChars(),
        srcpath,
        "LDC (http://www.dsource.org/projects/ldc)",
//FIXME: What do these two mean?
        false, // isMain,
        false // isOptimized
    );

    return irmod->diCompileUnit;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DISubprogram DtoDwarfSubProgram(FuncDeclaration* fd)
{
    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    llvm::DICompileUnit context = DtoDwarfCompileUnit(gIR->dmodule);
    llvm::DIFile file = DtoDwarfFile(fd->loc, DtoDwarfCompileUnit(getDefinedModule(fd)));

    // FIXME: duplicates ?
    return gIR->difactory.CreateSubprogram(
        context, // context
        fd->toPrettyChars(), // name
        fd->toPrettyChars(), // display name
        fd->mangle(), // linkage name
        file, // file
        fd->loc.linnum, // line no
//FIXME: what's this type for?
        llvm::DIType(NULL), // type
        fd->protection == PROTprivate, // is local to unit
        gIR->dmodule == getDefinedModule(fd) // isdefinition
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DISubprogram DtoDwarfSubProgramInternal(const char* prettyname, const char* mangledname)
{
    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    llvm::DICompileUnit context = DtoDwarfCompileUnit(gIR->dmodule);

    // FIXME: duplicates ?
    return gIR->difactory.CreateSubprogram(
        context, // context
        prettyname, // name
        prettyname, // display name
        mangledname, // linkage name
        DtoDwarfFile(Loc(gIR->dmodule, 0), context), // compile unit
        0, // line no
//FIXME: what's this type for?
        llvm::DIType(NULL), // type
        true, // is local to unit
        true // isdefinition
    );
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DIGlobalVariable DtoDwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd)
{
    Logger::println("D to dwarf global_variable");
    LOG_SCOPE;

    // FIXME: duplicates ?
    return dwarfGlobalVariable(ll, vd);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncStart(FuncDeclaration* fd)
{
    Logger::println("D to dwarf funcstart");
    LOG_SCOPE;

    assert((llvm::MDNode*)fd->ir.irFunc->diSubprogram != 0);
    //TODO:
    //gIR->difactory.InsertSubprogramStart(fd->ir.irFunc->diSubprogram, gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncEnd(FuncDeclaration* fd)
{
    Logger::println("D to dwarf funcend");
    LOG_SCOPE;

    assert((llvm::MDNode*)fd->ir.irFunc->diSubprogram != 0);
    //TODO:
    //gIR->difactory.InsertRegionEnd(fd->ir.irFunc->diSubprogram, gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfStopPoint(unsigned ln)
{
    Logger::println("D to dwarf stoppoint at line %u", ln);
    LOG_SCOPE;
    // TODO: possibly it is wrong.
    llvm::DebugLoc loc = llvm::DebugLoc::get(ln, 0, DtoDwarfCompileUnit(gIR->dmodule));
    gIR->ir->SetCurrentDebugLocation(loc);
}

#endif
