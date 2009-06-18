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

#define DBG_NULL    ( LLConstant::getNullValue(DBG_TYPE) )
#define DBG_TYPE    ( getPtrToType(llvm::StructType::get(NULL,NULL)) )
#define DBG_CAST(X) ( llvm::ConstantExpr::getBitCast(X, DBG_TYPE) )

#define DBG_TAG(X)  ( llvm::ConstantExpr::getAdd( DtoConstUint( X ), DtoConstUint( llvm::LLVMDebugVersion ) ) )

//////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Emits a global variable, LLVM Dwarf style, only declares.
 * @param type Type of variable.
 * @param name Name.
 * @return The global variable.
 */
static LLGlobalVariable* emitDwarfGlobalDecl(const LLStructType* type, const char* name, bool linkonce=false)
{
    LLGlobalValue::LinkageTypes linkage = linkonce
        ? DEBUGINFO_LINKONCE_LINKAGE_TYPE
        : LLGlobalValue::InternalLinkage;
    LLGlobalVariable* gv = new LLGlobalVariable(type, true, linkage, NULL, name, gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::DIAnchor getDwarfAnchor(dwarf_constants c)
{
    switch (c)
    {
    case DW_TAG_compile_unit:
        return gIR->difactory.GetOrCreateCompileUnitAnchor();
    case DW_TAG_variable:
        return gIR->difactory.GetOrCreateGlobalVariableAnchor();
    case DW_TAG_subprogram:
        return gIR->difactory.GetOrCreateSubprogramAnchor();
    default:
        assert(0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static const llvm::StructType* getDwarfCompileUnitType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.compile_unit.type"));
}

static const llvm::StructType* getDwarfSubProgramType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.subprogram.type"));
}

static const llvm::StructType* getDwarfVariableType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.variable.type"));
}

static const llvm::StructType* getDwarfDerivedTypeType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.derivedtype.type"));
}

static const llvm::StructType* getDwarfBasicTypeType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.basictype.type"));
}

static const llvm::StructType* getDwarfCompositeTypeType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.compositetype.type"));
}

static const llvm::StructType* getDwarfGlobalVariableType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.global_variable.type"));
}

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
        llvm::DICompileUnit(NULL), // compile unit
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
        llvm::DICompileUnit(NULL), // compile unit
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

static llvm::DIDerivedType dwarfMemberType(unsigned linnum, Type* type, llvm::DICompileUnit compileUnit, llvm::DICompileUnit definedCU, const char* c_name, unsigned offset)
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
        definedCU, // compile unit
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
    llvm::DICompileUnit definedCU,
    std::vector<LLConstant*>& elems)
{
    if (sd->baseClass)
    {
        add_base_fields(sd->baseClass, compileUnit, definedCU, elems);
    }

    ArrayIter<VarDeclaration> it(sd->fields);
    size_t narr = sd->fields.dim;
    elems.reserve(narr);
    for (; !it.done(); it.next())
    {
        VarDeclaration* vd = it.get();
        LLGlobalVariable* ptr = dwarfMemberType(vd->loc.linnum, vd->type, compileUnit, definedCU, vd->toChars(), vd->offset).getGV();
        elems.push_back(DBG_CAST(ptr));
    }
}

//FIXME: This does not use llvm's DIFactory as it can't 
//   handle recursive types properly.
static llvm::DICompositeType dwarfCompositeType(Type* type, llvm::DICompileUnit compileUnit)
{
    const LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    // defaults
    LLConstant* name = getNullPtr(getVoidPtrType());
    LLGlobalVariable* members = NULL;
    unsigned linnum = 0;
    llvm::DICompileUnit definedCU;

    // prepare tag and members
    unsigned tag;

    // declare final global variable
    LLGlobalVariable* gv = NULL;

    // dynamic array
    if (t->ty == Tarray)
    {
        tag = DW_TAG_structure_type;

        LLGlobalVariable* len = dwarfMemberType(0, Type::tsize_t, compileUnit, llvm::DICompileUnit(NULL), "length", 0).getGV();
        assert(len);
        LLGlobalVariable* ptr = dwarfMemberType(0, t->nextOf()->pointerTo(), compileUnit, llvm::DICompileUnit(NULL), "ptr", global.params.is64bit?8:4).getGV();
        assert(ptr);

        const LLArrayType* at = LLArrayType::get(DBG_TYPE, 2);

        std::vector<LLConstant*> elems(2);
        elems[0] = DBG_CAST(len);
        elems[1] = DBG_CAST(ptr);

        LLConstant* ca = LLConstantArray::get(at, elems);
        members = new LLGlobalVariable(ca->getType(), true, LLGlobalValue::InternalLinkage, ca, ".array", gIR->module);
        members->setSection("llvm.metadata");

        name = DtoConstStringPtr(t->toChars(), "llvm.metadata");
    }

    // struct/class
    else if (t->ty == Tstruct || t->ty == Tclass)
    {
        AggregateDeclaration* sd;
        if (t->ty == Tstruct)
        {
            TypeStruct* ts = (TypeStruct*)t;
            sd = ts->sym;
        }
        else
        {
            TypeClass* tc = (TypeClass*)t;
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
        if (!ir->diCompositeType.isNull())
            return ir->diCompositeType;

        // set to handle recursive types properly
        gv = emitDwarfGlobalDecl(getDwarfCompositeTypeType(), "llvm.dbg.compositetype");
        // set bogus initializer to satisfy asserts in DICompositeType constructor
        gv->setInitializer(LLConstant::getNullValue(getDwarfCompositeTypeType()));
        ir->diCompositeType = llvm::DICompositeType(gv);

        tag = DW_TAG_structure_type;

        name = DtoConstStringPtr(sd->toChars(), "llvm.metadata");
        linnum = sd->loc.linnum;
        definedCU = DtoDwarfCompileUnit(getDefinedModule(sd));

        std::vector<LLConstant*> elems;
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
                    LLGlobalVariable* ptr = dwarfMemberType(vd->loc.linnum, vd->type, compileUnit, definedCU, vd->toChars(), vd->offset).getGV();
                    elems.push_back(DBG_CAST(ptr));
                }
            }
            else
            {
                add_base_fields(ir->aggrdecl->isClassDeclaration(), compileUnit, definedCU, elems);
            }
        }

        const LLArrayType* at = LLArrayType::get(DBG_TYPE, elems.size());
        LLConstant* ca = LLConstantArray::get(at, elems);
        members = new LLGlobalVariable(ca->getType(), true, LLGlobalValue::InternalLinkage, ca, ".array", gIR->module);
        members->setSection("llvm.metadata");
    }

    // unsupported composite type
    else
    {
        assert(0 && "unsupported compositetype for debug info");
    }

    std::vector<LLConstant*> vals(11);

    // tag
    vals[0] = DBG_TAG(tag);

    // context
    vals[1] = DBG_CAST(compileUnit.getGV());

    // name
    vals[2] = name;

    // compile unit where defined
    if (definedCU.getGV())
        vals[3] = DBG_CAST(definedCU.getGV());
    else
        vals[3] = DBG_NULL;

    // line number where defined
    vals[4] = DtoConstInt(linnum);

    // size in bits
    vals[5] = LLConstantInt::get(LLType::Int64Ty, getTypeBitSize(T), false);

    // alignment in bits
    vals[6] = LLConstantInt::get(LLType::Int64Ty, getABITypeAlign(T)*8, false);

    // offset in bits
    vals[7] = LLConstantInt::get(LLType::Int64Ty, 0, false);

    // FIXME: dont know what this is
    vals[8] = DtoConstUint(0);

    // FIXME: ditto
    vals[9] = DBG_NULL;

    // members array
    if (members)
        vals[10] = DBG_CAST(members);
    else
        vals[10] = DBG_NULL;

    // set initializer
    if (!gv)
        gv = emitDwarfGlobalDecl(getDwarfCompositeTypeType(), "llvm.dbg.compositetype");
    LLConstant* initia = LLConstantStruct::get(getDwarfCompositeTypeType(), vals);
    gv->setInitializer(initia);

    return llvm::DICompositeType(gv);
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
        DtoDwarfCompileUnit(getDefinedModule(vd)), // compile unit
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
        DtoDwarfCompileUnit(getDefinedModule(vd)), // compile unit
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
    if (TD.isNull())
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

    if (!irmod->diCompileUnit.isNull())
    {
        assert (irmod->diCompileUnit.getGV()->getParent() == gIR->module
            && "debug info compile unit belongs to incorrect llvm module!");
        return irmod->diCompileUnit;
    }

    // prepare srcpath
    std::string srcpath(FileName::path(m->srcfile->name->toChars()));
    if (!FileName::absolute(srcpath.c_str())) {
        llvm::sys::Path tmp = llvm::sys::Path::GetCurrentDirectory();
        tmp.appendComponent(srcpath);
        srcpath = tmp.toString();
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
    
    // if the linkage stays internal, we can't llvm-link the generated modules together:
    // llvm's DwarfWriter uses path and filename to determine the symbol name and we'd
    // end up with duplicate symbols
    irmod->diCompileUnit.getGV()->setLinkage(DEBUGINFO_LINKONCE_LINKAGE_TYPE);
    irmod->diCompileUnit.getGV()->setName(std::string("llvm.dbg.compile_unit_") + srcpath + m->srcfile->name->toChars());

    return irmod->diCompileUnit;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::DISubprogram DtoDwarfSubProgram(FuncDeclaration* fd)
{
    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    llvm::DICompileUnit context = DtoDwarfCompileUnit(gIR->dmodule);
    llvm::DICompileUnit definition = DtoDwarfCompileUnit(getDefinedModule(fd));

    // FIXME: duplicates ?
    return gIR->difactory.CreateSubprogram(
        context, // context
        fd->toPrettyChars(), // name
        fd->toPrettyChars(), // display name
        fd->mangle(), // linkage name
        definition, // compile unit
        fd->loc.linnum, // line no
//FIXME: what's this type for?
        llvm::DIType(NULL), // type
        fd->protection == PROTprivate, // is local to unit
        context.getGV() == definition.getGV() // isdefinition
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
        context, // compile unit
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

    assert(!fd->ir.irFunc->diSubprogram.isNull());
    gIR->difactory.InsertSubprogramStart(fd->ir.irFunc->diSubprogram, gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncEnd(FuncDeclaration* fd)
{
    Logger::println("D to dwarf funcend");
    LOG_SCOPE;

    assert(!fd->ir.irFunc->diSubprogram.isNull());
    gIR->difactory.InsertRegionEnd(fd->ir.irFunc->diSubprogram, gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfStopPoint(unsigned ln)
{
    Logger::println("D to dwarf stoppoint at line %u", ln);
    LOG_SCOPE;

    gIR->difactory.InsertStopPoint(
        DtoDwarfCompileUnit(getDefinedModule(gIR->func()->decl)), // compile unit
        ln, // line no
        0, // col no
        gIR->scopebb()
    );
}
