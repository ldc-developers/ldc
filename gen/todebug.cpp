#include "gen/llvm.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

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
    LLGlobalVariable* gv = new LLGlobalVariable(c->getType(), true, linkage, c, name, gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::GlobalVariable* dbg_compile_units = 0;
static llvm::GlobalVariable* dbg_global_variables = 0;
static llvm::GlobalVariable* dbg_subprograms = 0;

/**
 * Emits the Dwarf anchors that are used repeatedly by LLVM debug info.
 */
static void emitDwarfAnchors()
{
    const llvm::StructType* anchorTy = isaStruct(gIR->module->getTypeByName("llvm.dbg.anchor.type"));
    std::vector<LLConstant*> vals(2);

    if (!gIR->module->getNamedGlobal("llvm.dbg.compile_units")) {
        vals[0] = DtoConstUint(llvm::LLVMDebugVersion);
        vals[1] = DtoConstUint(DW_TAG_compile_unit);
        dbg_compile_units = emitDwarfGlobal(anchorTy, vals, "llvm.dbg.compile_units", true);
    }
    if (!gIR->module->getNamedGlobal("llvm.dbg.global_variables")) {
        vals[0] = DtoConstUint(llvm::LLVMDebugVersion);
        vals[1] = DtoConstUint(DW_TAG_variable);
        dbg_global_variables = emitDwarfGlobal(anchorTy, vals, "llvm.dbg.global_variables", true);
    }
    if (!gIR->module->getNamedGlobal("llvm.dbg.subprograms")) {
        vals[0] = DtoConstUint(llvm::LLVMDebugVersion);
        vals[1] = DtoConstUint(DW_TAG_subprogram);
        dbg_subprograms = emitDwarfGlobal(anchorTy, vals, "llvm.dbg.subprograms", true);
    }
}

static LLConstant* getDwarfAnchor(dwarf_constants c)
{
    emitDwarfAnchors();
    switch (c)
    {
    case DW_TAG_compile_unit:
        return dbg_compile_units;
    case DW_TAG_variable:
        return dbg_global_variables;
    case DW_TAG_subprogram:
        return dbg_subprograms;
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
    std::vector<LLConstant*> vals;
    vals.push_back(DBG_TAG(DW_TAG_compile_unit));
    vals.push_back(DBG_CAST(getDwarfAnchor(DW_TAG_compile_unit)));

    vals.push_back(DtoConstUint(DW_LANG_C));// _D)); // doesn't seem to work
    vals.push_back(DtoConstStringPtr(m->srcfile->name->toChars(), "llvm.metadata"));
    std::string srcpath(FileName::path(m->srcfile->name->toChars()));
    if (srcpath.empty()) {
        const char* str = get_current_dir_name();
        assert(str != NULL);
        srcpath = str;
    }
    vals.push_back(DtoConstStringPtr(srcpath.c_str(), "llvm.metadata"));
    vals.push_back(DtoConstStringPtr("LLVMDC (http://www.dsource.org/projects/llvmdc)", "llvm.metadata"));

    LLGlobalVariable* gv = emitDwarfGlobal(getDwarfCompileUnitType(), vals, "llvm.dbg.compile_unit");
    m->ir.irModule->dwarfCompileUnit = gv;
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfSubProgram(FuncDeclaration* fd, llvm::GlobalVariable* compileUnit)
{
    std::vector<LLConstant*> vals;
    vals.push_back(DBG_TAG(DW_TAG_subprogram));
    vals.push_back(DBG_CAST(getDwarfAnchor(DW_TAG_subprogram)));

    vals.push_back(DBG_CAST(compileUnit));
    vals.push_back(DtoConstStringPtr(fd->toPrettyChars(), "llvm.metadata"));
    vals.push_back(vals.back());
    vals.push_back(DtoConstStringPtr(fd->mangle(), "llvm.metadata"));
    vals.push_back(DBG_CAST( DtoDwarfCompileUnit(fd->getModule()) ));
    vals.push_back(DtoConstUint(fd->loc.linnum));
    vals.push_back(DBG_NULL);
    vals.push_back(DtoConstBool(fd->protection == PROTprivate));
    vals.push_back(DtoConstBool(fd->getModule() == gIR->dmodule));

    return emitDwarfGlobal(getDwarfSubProgramType(), vals, "llvm.dbg.subprogram");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfTypeDescription(Loc loc, Type* type, LLGlobalVariable* cu, const char* c_name);

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfBasicType(Type* type, llvm::GlobalVariable* compileUnit)
{
    Type* t = type->toBasetype();

    const LLType* T = DtoType(type);

    std::vector<LLConstant*> vals;

    // tag
    vals.push_back(DBG_TAG(DW_TAG_base_type));

    // context
    vals.push_back(DBG_CAST(compileUnit));

    // name
    vals.push_back(DtoConstStringPtr(type->toChars(), "llvm.metadata"));

    // compile unit where defined
    vals.push_back(DBG_NULL);

    // line number where defined
    vals.push_back(DtoConstInt(0));

    // size in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, getTypeBitSize(T), false));

    // alignment in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, getABITypeAlign(T)*8, false));

    // offset in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, 0, false));

    // FIXME: dont know what this is
    vals.push_back(DtoConstUint(0));

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
    vals.push_back(DtoConstUint(id));

    return emitDwarfGlobal(getDwarfBasicTypeType(), vals, "llvm.dbg.basictype");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfDerivedType(Loc loc, Type* type, llvm::GlobalVariable* compileUnit)
{
    const LLType* T = DtoType(type);
    Type* t = DtoDType(type);

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

    std::vector<LLConstant*> vals;

    // tag
    vals.push_back(DBG_TAG(tag));

    // context
    vals.push_back(DBG_CAST(compileUnit));

    // name
    vals.push_back(name);

    // compile unit where defined
    vals.push_back(DBG_NULL);

    // line number where defined
    vals.push_back(DtoConstInt(0));

    // size in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, getTypeBitSize(T), false));

    // alignment in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, getABITypeAlign(T)*8, false));

    // offset in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, 0, false));

    // FIXME: dont know what this is
    vals.push_back(DtoConstUint(0));

    // base type
    Type* nt = t->nextOf();
    LLGlobalVariable* nTD = dwarfTypeDescription(loc, nt, compileUnit, NULL);
    if (nt->ty == Tvoid || !nTD)
        vals.push_back(DBG_NULL);
    else
        vals.push_back(DBG_CAST(nTD));

    return emitDwarfGlobal(getDwarfDerivedTypeType(), vals, "llvm.dbg.derivedtype");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfMemberType(Loc loc, Type* type, llvm::GlobalVariable* compileUnit, const char* c_name, unsigned offset)
{
    const LLType* T = DtoType(type);
    Type* t = DtoDType(type);

    // defaults
    LLConstant* name;
    if (c_name)
        name = DtoConstStringPtr(c_name, "llvm.metadata");
    else
        name = getNullPtr(getVoidPtrType());

    std::vector<LLConstant*> vals;

    // tag
    vals.push_back(DBG_TAG(DW_TAG_member));

    // context
    vals.push_back(DBG_CAST(compileUnit));

    // name
    vals.push_back(name);

    // compile unit where defined
    vals.push_back(DBG_NULL);

    // line number where defined
    vals.push_back(DtoConstInt(0));

    // size in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, getTypeBitSize(T), false));

    // alignment in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, getABITypeAlign(T)*8, false));

    // offset in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, offset*8, false));

    // FIXME: dont know what this is
    vals.push_back(DtoConstUint(0));

    // base type
    LLGlobalVariable* nTD = dwarfTypeDescription(loc, t, compileUnit, NULL);
    if (t->ty == Tvoid || !nTD)
        vals.push_back(DBG_NULL);
    else
        vals.push_back(DBG_CAST(nTD));

    return emitDwarfGlobal(getDwarfDerivedTypeType(), vals, "llvm.dbg.derivedtype");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfCompositeType(Loc loc, Type* type, llvm::GlobalVariable* compileUnit)
{
    const LLType* T = DtoType(type);
    Type* t = DtoDType(type);

    // defaults
    LLConstant* name = getNullPtr(getVoidPtrType());
    LLGlobalVariable* members = NULL;

    // prepare tag and members
    unsigned tag;
    if (t->ty == Tarray)
    {
        tag = DW_TAG_structure_type;

        LLGlobalVariable* len = dwarfMemberType(loc, Type::tsize_t, compileUnit, "length", 0);
        assert(len);
        LLGlobalVariable* ptr = dwarfMemberType(loc, t->nextOf()->pointerTo(), compileUnit, "ptr", global.params.is64bit?8:4);
        assert(ptr);

        const LLArrayType* at = LLArrayType::get(DBG_TYPE, 2);

        std::vector<LLConstant*> elems;
        elems.push_back(DBG_CAST(len));
        elems.push_back(DBG_CAST(ptr));

//         elems[0]->dump();
//         elems[1]->dump();
//         at->dump();

        LLConstant* ca = LLConstantArray::get(at, elems);
        members = new LLGlobalVariable(ca->getType(), true, LLGlobalValue::InternalLinkage, ca, ".array", gIR->module);
        members->setSection("llvm.metadata");
    }
    else
    {
        assert(0 && "unsupported compositetype for debug info");
    }

    std::vector<LLConstant*> vals;

    // tag
    vals.push_back(DBG_TAG(tag));

    // context
    vals.push_back(DBG_CAST(compileUnit));

    // name
    vals.push_back(name);

    // compile unit where defined
    vals.push_back(DBG_NULL);

    // line number where defined
    vals.push_back(DtoConstInt(0));

    // size in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, getTypeBitSize(T), false));

    // alignment in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, getABITypeAlign(T)*8, false));

    // offset in bits
    vals.push_back(LLConstantInt::get(LLType::Int64Ty, 0, false));

    // FIXME: dont know what this is
    vals.push_back(DtoConstUint(0));

    // FIXME: ditto
    vals.push_back(DBG_NULL);

    // members array
    if (members)
        vals.push_back(DBG_CAST(members));
    else
        vals.push_back(DBG_NULL);

    return emitDwarfGlobal(getDwarfCompositeTypeType(), vals, "llvm.dbg.compositetype");
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd)
{
    assert(vd->isDataseg());
    LLGlobalVariable* compileUnit = DtoDwarfCompileUnit(gIR->dmodule);

    std::vector<LLConstant*> vals;
    vals.push_back(DBG_TAG(DW_TAG_variable));
    vals.push_back(DBG_CAST(getDwarfAnchor(DW_TAG_variable)));

    vals.push_back(DBG_CAST(compileUnit));

    vals.push_back(DtoConstStringPtr(vd->mangle(), "llvm.metadata"));
    vals.push_back(DtoConstStringPtr(vd->toPrettyChars(), "llvm.metadata"));
    vals.push_back(DtoConstStringPtr(vd->toChars(), "llvm.metadata"));

    vals.push_back(DBG_CAST(DtoDwarfCompileUnit(vd->getModule())));
    vals.push_back(DtoConstUint(vd->loc.linnum));

    LLGlobalVariable* TY = dwarfTypeDescription(vd->loc, vd->type, compileUnit, NULL);
    vals.push_back(TY ? DBG_CAST(TY) : DBG_NULL);
    vals.push_back(DtoConstBool(vd->protection == PROTprivate));
    vals.push_back(DtoConstBool(vd->getModule() == gIR->dmodule));

    vals.push_back(DBG_CAST(ll));

    return emitDwarfGlobal(getDwarfGlobalVariableType(), vals, "llvm.dbg.global_variable");
}

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
    LLSmallVector<LLValue*,2> args;
    args.push_back(DtoBitCast(var, DBG_TYPE));
    args.push_back(DBG_CAST(varDescr));
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.declare"), args.begin(), args.end());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfTypeDescription(Loc loc, Type* type, LLGlobalVariable* cu, const char* c_name)
{
    Type* t = type->toBasetype();
    if (t->ty == Tvoid)
        return NULL;
    else if (t->isintegral() || t->isfloating())
        return dwarfBasicType(type, cu);
    else if (t->ty == Tpointer)
        return dwarfDerivedType(loc, type, cu);
    else if (t->ty == Tarray)
        return dwarfCompositeType(loc, type, cu);

    if (global.params.warnings)
        warning("%s: unsupported type for debug info: %s", loc.toChars(), type->toChars());
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfLocalVariable(LLValue* ll, VarDeclaration* vd)
{
    // get compile units
    LLGlobalVariable* thisCU = DtoDwarfCompileUnit(gIR->dmodule);
    LLGlobalVariable* varCU = thisCU;
    if (vd->getModule() != gIR->dmodule)
        varCU = DtoDwarfCompileUnit(vd->getModule());

    // get type description
    Type* t = vd->type->toBasetype();
    LLGlobalVariable* TD = dwarfTypeDescription(vd->loc, vd->type, thisCU, NULL);
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
    // FIXME: duplicates ?
    return dwarfSubProgram(fd, DtoDwarfCompileUnit(gIR->dmodule));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

LLGlobalVariable* DtoDwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd)
{
    // FIXME: duplicates ?
    return dwarfGlobalVariable(ll, vd);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncStart(FuncDeclaration* fd)
{
    assert(fd->ir.irFunc->dwarfSubProg);
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.func.start"), DBG_CAST(fd->ir.irFunc->dwarfSubProg));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncEnd(FuncDeclaration* fd)
{
    assert(fd->ir.irFunc->dwarfSubProg);
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.region.end"), DBG_CAST(fd->ir.irFunc->dwarfSubProg));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfStopPoint(unsigned ln)
{
    LLSmallVector<LLValue*,3> args;
    args.push_back(DtoConstUint(ln));
    args.push_back(DtoConstUint(0));
    FuncDeclaration* fd = gIR->func()->decl;
    args.push_back(DBG_CAST(DtoDwarfCompileUnit(fd->getModule())));
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.stoppoint"), args.begin(), args.end());
}
