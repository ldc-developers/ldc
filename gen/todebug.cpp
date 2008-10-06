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

#include "ir/irmodule.h"

using namespace llvm::dwarf;

#define DBG_NULL    ( LLConstant::getNullValue(DBG_TYPE) )
#define DBG_TYPE    ( getPtrToType(llvm::StructType::get(NULL,NULL)) )
#define DBG_CAST(X) ( llvm::ConstantExpr::getBitCast(X, DBG_TYPE) )

#define DBG_TAG(X)  ( llvm::ConstantExpr::getAdd( DtoConstUint( X ), DtoConstUint( llvm::LLVMDebugVersion ) ) )

//////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Emits a global variable, LLVM Dwarf style.
 * @param type Type of variable.
 * @param values Initializers.
 * @param name Name.
 * @return The global variable.
 */
static LLGlobalVariable* emitDwarfGlobal(const LLStructType* type, const std::vector<LLConstant*> values, const char* name, bool linkonce=false)
{
    LLConstant* c = llvm::ConstantStruct::get(type, values);
    LLGlobalValue::LinkageTypes linkage = linkonce ? LLGlobalValue::LinkOnceLinkage : LLGlobalValue::InternalLinkage;
    LLGlobalVariable* gv = new LLGlobalVariable(type, true, linkage, c, name, gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

/**
 * Emits a global variable, LLVM Dwarf style, only declares.
 * @param type Type of variable.
 * @param name Name.
 * @return The global variable.
 */
static LLGlobalVariable* emitDwarfGlobalDecl(const LLStructType* type, const char* name, bool linkonce=false)
{
    LLGlobalValue::LinkageTypes linkage = linkonce ? LLGlobalValue::LinkOnceLinkage : LLGlobalValue::InternalLinkage;
    LLGlobalVariable* gv = new LLGlobalVariable(type, true, linkage, NULL, name, gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Emits the Dwarf anchors that are used repeatedly by LLVM debug info.
 */
static void emitDwarfAnchors()
{
    const llvm::StructType* anchorTy = isaStruct(gIR->module->getTypeByName("llvm.dbg.anchor.type"));
    std::vector<LLConstant*> vals(2);

    vals[0] = DtoConstUint(llvm::LLVMDebugVersion);
    vals[1] = DtoConstUint(DW_TAG_compile_unit);
    gIR->dwarfCUs = emitDwarfGlobal(anchorTy, vals, "llvm.dbg.compile_units", true);

    vals[0] = DtoConstUint(llvm::LLVMDebugVersion);
    vals[1] = DtoConstUint(DW_TAG_variable);
    gIR->dwarfGVs = emitDwarfGlobal(anchorTy, vals, "llvm.dbg.global_variables", true);

    vals[0] = DtoConstUint(llvm::LLVMDebugVersion);
    vals[1] = DtoConstUint(DW_TAG_subprogram);
    gIR->dwarfSPs = emitDwarfGlobal(anchorTy, vals, "llvm.dbg.subprograms", true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLConstant* getDwarfAnchor(dwarf_constants c)
{
    if (!gIR->dwarfCUs)
        emitDwarfAnchors();
    switch (c)
    {
    case DW_TAG_compile_unit:
        return gIR->dwarfCUs;
    case DW_TAG_variable:
        return gIR->dwarfGVs;
    case DW_TAG_subprogram:
        return gIR->dwarfSPs;
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

static LLGlobalVariable* dwarfCompileUnit(Module* m)
{
    std::vector<LLConstant*> vals(6);
    vals[0] = DBG_TAG(DW_TAG_compile_unit);
    vals[1] = DBG_CAST(getDwarfAnchor(DW_TAG_compile_unit));

    vals[2] = DtoConstUint(DW_LANG_C);// _D)); // doesn't seem to work
    vals[3] = DtoConstStringPtr(m->srcfile->name->toChars(), "llvm.metadata");
    std::string srcpath(FileName::path(m->srcfile->name->toChars()));
    if (srcpath.empty())
		srcpath = llvm::sys::Path::GetCurrentDirectory().toString();
    vals[4] = DtoConstStringPtr(srcpath.c_str(), "llvm.metadata");
    vals[5] = DtoConstStringPtr("LDC (http://www.dsource.org/projects/llvmdc)", "llvm.metadata");

    LLGlobalVariable* gv = emitDwarfGlobal(getDwarfCompileUnitType(), vals, "llvm.dbg.compile_unit");
    m->ir.irModule->dwarfCompileUnit = gv;
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfSubProgram(FuncDeclaration* fd, llvm::GlobalVariable* compileUnit)
{
    std::vector<LLConstant*> vals(11);
    vals[0] = DBG_TAG(DW_TAG_subprogram);
    vals[1] = DBG_CAST(getDwarfAnchor(DW_TAG_subprogram));

    vals[2] = DBG_CAST(compileUnit);
    vals[3] = DtoConstStringPtr(fd->toPrettyChars(), "llvm.metadata");
    vals[4] = vals[3];
    vals[5] = DtoConstStringPtr(fd->mangle(), "llvm.metadata");
    vals[6] = DBG_CAST( DtoDwarfCompileUnit(fd->getModule()) );
    vals[7] = DtoConstUint(fd->loc.linnum);
    vals[8] = DBG_NULL;
    vals[9] = DtoConstBool(fd->protection == PROTprivate);
    vals[10] = DtoConstBool(fd->getModule() == gIR->dmodule);

    Logger::println("emitting subprogram global");

    return emitDwarfGlobal(getDwarfSubProgramType(), vals, "llvm.dbg.subprogram");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfTypeDescription_impl(Type* type, LLGlobalVariable* cu, const char* c_name);
static LLGlobalVariable* dwarfTypeDescription(Type* type, LLGlobalVariable* cu, const char* c_name);

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfBasicType(Type* type, llvm::GlobalVariable* compileUnit)
{
    Type* t = type->toBasetype();

    const LLType* T = DtoType(type);

    std::vector<LLConstant*> vals(10);

    // tag
    vals[0] = DBG_TAG(DW_TAG_base_type);

    // context
    vals[1] = DBG_CAST(compileUnit);

    // name
    vals[2] = DtoConstStringPtr(type->toChars(), "llvm.metadata");

    // compile unit where defined
    vals[3] = DBG_NULL;

    // line number where defined
    vals[4] = DtoConstInt(0);

    // size in bits
    vals[5] = LLConstantInt::get(LLType::Int64Ty, getTypeBitSize(T), false);

    // alignment in bits
    vals[6] = LLConstantInt::get(LLType::Int64Ty, getABITypeAlign(T)*8, false);

    // offset in bits
    vals[7] = LLConstantInt::get(LLType::Int64Ty, 0, false);

    // FIXME: dont know what this is
    vals[8] = DtoConstUint(0);

    // dwarf type
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
    vals[9] = DtoConstUint(id);

    return emitDwarfGlobal(getDwarfBasicTypeType(), vals, "llvm.dbg.basictype");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfDerivedType(Type* type, llvm::GlobalVariable* compileUnit)
{
    const LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    // defaults
    LLConstant* name = getNullPtr(getVoidPtrType());

    // find tag
    unsigned tag;
    if (t->ty == Tpointer)
    {
        tag = DW_TAG_pointer_type;
    }
    else
    {
        assert(0 && "unsupported derivedtype for debug info");
    }

    std::vector<LLConstant*> vals(10);

    // tag
    vals[0] = DBG_TAG(tag);

    // context
    vals[1] = DBG_CAST(compileUnit);

    // name
    vals[2] = name;

    // compile unit where defined
    vals[3] = DBG_NULL;

    // line number where defined
    vals[4] = DtoConstInt(0);

    // size in bits
    vals[5] = LLConstantInt::get(LLType::Int64Ty, getTypeBitSize(T), false);

    // alignment in bits
    vals[6] = LLConstantInt::get(LLType::Int64Ty, getABITypeAlign(T)*8, false);

    // offset in bits
    vals[7] = LLConstantInt::get(LLType::Int64Ty, 0, false);

    // FIXME: dont know what this is
    vals[8] = DtoConstUint(0);

    // base type
    Type* nt = t->nextOf();
    LLGlobalVariable* nTD = dwarfTypeDescription_impl(nt, compileUnit, NULL);
    if (nt->ty == Tvoid || !nTD)
        vals[9] = DBG_NULL;
    else
        vals[9] = DBG_CAST(nTD);

    return emitDwarfGlobal(getDwarfDerivedTypeType(), vals, "llvm.dbg.derivedtype");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfMemberType(unsigned linnum, Type* type, LLGlobalVariable* compileUnit, LLGlobalVariable* definedCU, const char* c_name, unsigned offset)
{
    const LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    // defaults
    LLConstant* name;
    if (c_name)
        name = DtoConstStringPtr(c_name, "llvm.metadata");
    else
        name = getNullPtr(getVoidPtrType());

    std::vector<LLConstant*> vals(10);

    // tag
    vals[0] = DBG_TAG(DW_TAG_member);

    // context
    vals[1] = DBG_CAST(compileUnit);

    // name
    vals[2] = name;

    // compile unit where defined
    if (definedCU)
        vals[3] = DBG_CAST(definedCU);
    else
        vals[3] = DBG_NULL;

    // line number where defined
    vals[4] = DtoConstInt(linnum);

    // size in bits
    vals[5] = LLConstantInt::get(LLType::Int64Ty, getTypeBitSize(T), false);

    // alignment in bits
    vals[6] = LLConstantInt::get(LLType::Int64Ty, getABITypeAlign(T)*8, false);

    // offset in bits
    vals[7] = LLConstantInt::get(LLType::Int64Ty, offset*8, false);

    // FIXME: dont know what this is
    vals[8] = DtoConstUint(0);

    // base type
    LLGlobalVariable* nTD = dwarfTypeDescription(t, compileUnit, NULL);
    if (t->ty == Tvoid || !nTD)
        vals[9] = DBG_NULL;
    else
        vals[9] = DBG_CAST(nTD);

    return emitDwarfGlobal(getDwarfDerivedTypeType(), vals, "llvm.dbg.derivedtype");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfCompositeType(Type* type, llvm::GlobalVariable* compileUnit)
{
    const LLType* T = DtoType(type);
    Type* t = type->toBasetype();

    // defaults
    LLConstant* name = getNullPtr(getVoidPtrType());
    LLGlobalVariable* members = NULL;
    unsigned linnum = 0;
    LLGlobalVariable* definedCU = NULL;

    // prepare tag and members
    unsigned tag;

    // declare final global variable
    LLGlobalVariable* gv = NULL;

    // dynamic array
    if (t->ty == Tarray)
    {
        tag = DW_TAG_structure_type;

        LLGlobalVariable* len = dwarfMemberType(0, Type::tsize_t, compileUnit, NULL, "length", 0);
        assert(len);
        LLGlobalVariable* ptr = dwarfMemberType(0, t->nextOf()->pointerTo(), compileUnit, NULL, "ptr", global.params.is64bit?8:4);
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

        IrStruct* ir = sd->ir.irStruct;
        assert(ir);
        if (ir->dwarfComposite)
            return ir->dwarfComposite;

        // set to handle recursive types properly
        gv = emitDwarfGlobalDecl(getDwarfCompositeTypeType(), "llvm.dbg.compositetype");
        ir->dwarfComposite = gv;

        tag = DW_TAG_structure_type;

        name = DtoConstStringPtr(sd->toChars(), "llvm.metadata");
        linnum = sd->loc.linnum;
        definedCU = DtoDwarfCompileUnit(sd->getModule());

        std::vector<LLConstant*> elems;
        elems.reserve(ir->offsets.size());
        for (IrStruct::OffsetMap::iterator i=ir->offsets.begin(); i!=ir->offsets.end(); ++i)
        {
            unsigned offset = i->first;
            IrStruct::Offset& o = i->second;

            LLGlobalVariable* ptr = dwarfMemberType(o.var->loc.linnum, o.var->type, compileUnit, definedCU, o.var->toChars(), offset);
            elems.push_back(DBG_CAST(ptr));
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
    vals[1] = DBG_CAST(compileUnit);

    // name
    vals[2] = name;

    // compile unit where defined
    if (definedCU)
        vals[3] = DBG_CAST(definedCU);
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

    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd)
{
    assert(vd->isDataseg());
    LLGlobalVariable* compileUnit = DtoDwarfCompileUnit(gIR->dmodule);

    std::vector<LLConstant*> vals(12);
    vals[0] = DBG_TAG(DW_TAG_variable);
    vals[1] = DBG_CAST(getDwarfAnchor(DW_TAG_variable));

    vals[2] = DBG_CAST(compileUnit);

    vals[3] = DtoConstStringPtr(vd->mangle(), "llvm.metadata");
    vals[4] = DtoConstStringPtr(vd->toPrettyChars(), "llvm.metadata");
    vals[5] = DtoConstStringPtr(vd->toChars(), "llvm.metadata");

    vals[6] = DBG_CAST(DtoDwarfCompileUnit(vd->getModule()));
    vals[7] = DtoConstUint(vd->loc.linnum);

    LLGlobalVariable* TY = dwarfTypeDescription_impl(vd->type, compileUnit, NULL);
    vals[8] = TY ? DBG_CAST(TY) : DBG_NULL;
    vals[9] = DtoConstBool(vd->protection == PROTprivate);
    vals[10] = DtoConstBool(vd->getModule() == gIR->dmodule);

    vals[11] = DBG_CAST(ll);

    return emitDwarfGlobal(getDwarfGlobalVariableType(), vals, "llvm.dbg.global_variable");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfVariable(VarDeclaration* vd, LLGlobalVariable* typeDescr)
{
    assert(!vd->isDataseg() && "static variable");

    unsigned tag;
    if (vd->isParameter())
        tag = DW_TAG_arg_variable;
    else
        tag = DW_TAG_auto_variable;

    std::vector<LLConstant*> vals(6);
    // tag
    vals[0] = DBG_TAG(tag);
    // context
    vals[1] = DBG_CAST(gIR->func()->dwarfSubProg);
    // name
    vals[2] = DtoConstStringPtr(vd->toChars(), "llvm.metadata");
    // compile unit where defined
    vals[3] = DBG_CAST(DtoDwarfCompileUnit(vd->getModule()));
    // line number where defined
    vals[4] = DtoConstUint(vd->loc.linnum);
    // type descriptor
    vals[5] = DBG_CAST(typeDescr);

    return emitDwarfGlobal(getDwarfVariableType(), vals, "llvm.dbg.variable");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static void dwarfDeclare(LLValue* var, LLGlobalVariable* varDescr)
{
    LLSmallVector<LLValue*,2> args(2);
    args[0] = DtoBitCast(var, DBG_TYPE);
    args[1] = DBG_CAST(varDescr);
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.declare"), args.begin(), args.end());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfTypeDescription_impl(Type* type, LLGlobalVariable* cu, const char* c_name)
{
    Type* t = type->toBasetype();
    if (t->ty == Tvoid)
        return NULL;
    else if (t->isintegral() || t->isfloating())
        return dwarfBasicType(type, cu);
    else if (t->ty == Tpointer)
        return dwarfDerivedType(type, cu);
    else if (t->ty == Tarray || t->ty == Tstruct || t->ty == Tclass)
        return dwarfCompositeType(type, cu);

    return NULL;
}

static LLGlobalVariable* dwarfTypeDescription(Type* type, LLGlobalVariable* cu, const char* c_name)
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
    LLGlobalVariable* thisCU = DtoDwarfCompileUnit(gIR->dmodule);
    LLGlobalVariable* varCU = thisCU;
    if (vd->getModule() != gIR->dmodule)
        varCU = DtoDwarfCompileUnit(vd->getModule());

    // get type description
    Type* t = vd->type->toBasetype();
    LLGlobalVariable* TD = dwarfTypeDescription(vd->type, thisCU, NULL);
    if (TD == NULL)
        return; // unsupported

    // get variable description
    LLGlobalVariable* VD;
    VD = dwarfVariable(vd, TD);

    // declare
    dwarfDeclare(ll, VD);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

LLGlobalVariable* DtoDwarfCompileUnit(Module* m)
{
    Logger::println("D to dwarf compile_unit");
    LOG_SCOPE;

    // we might be generating for an import
    if (!m->ir.irModule)
        m->ir.irModule = new IrModule(m);
    else if (m->ir.irModule->dwarfCompileUnit)
    {
        if (m->ir.irModule->dwarfCompileUnit->getParent() == gIR->module)
            return m->ir.irModule->dwarfCompileUnit;
    }

    LLGlobalVariable* gv = dwarfCompileUnit(m);
    m->ir.irModule->dwarfCompileUnit = gv;
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

LLGlobalVariable* DtoDwarfSubProgram(FuncDeclaration* fd)
{
    Logger::println("D to dwarf subprogram");
    LOG_SCOPE;

    // FIXME: duplicates ?
    return dwarfSubProgram(fd, DtoDwarfCompileUnit(gIR->dmodule));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

LLGlobalVariable* DtoDwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd)
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

    assert(fd->ir.irFunc->dwarfSubProg);
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.func.start"), DBG_CAST(fd->ir.irFunc->dwarfSubProg));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncEnd(FuncDeclaration* fd)
{
    Logger::println("D to dwarf funcend");
    LOG_SCOPE;

    assert(fd->ir.irFunc->dwarfSubProg);
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.region.end"), DBG_CAST(fd->ir.irFunc->dwarfSubProg));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfStopPoint(unsigned ln)
{
    Logger::println("D to dwarf stoppoint at line %u", ln);
    LOG_SCOPE;

    LLSmallVector<LLValue*,3> args(3);
    args[0] = DtoConstUint(ln);
    args[1] = DtoConstUint(0);
    FuncDeclaration* fd = gIR->func()->decl;
    args[2] = DBG_CAST(DtoDwarfCompileUnit(fd->getModule()));
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.stoppoint"), args.begin(), args.end());
}
