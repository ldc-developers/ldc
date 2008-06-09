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

//////////////////////////////////////////////////////////////////////////////////////////////////

static llvm::GlobalVariable* dbg_compile_units = 0;
static llvm::GlobalVariable* dbg_global_variables = 0;
static llvm::GlobalVariable* dbg_subprograms = 0;

const llvm::StructType* GetDwarfAnchorType()
{
    /*
    %llvm.dbg.anchor.type = type {
        uint,   ;; Tag = 0 + LLVMDebugVersion
        uint    ;; Tag of descriptors grouped by the anchor
    }
    */

    const llvm::StructType* t = isaStruct(gIR->module->getTypeByName("llvm.dbg.anchor.type"));

    /*
    %llvm.dbg.compile_units       = linkonce constant %llvm.dbg.anchor.type  { uint 0, uint 17 } ;; DW_TAG_compile_unit
    %llvm.dbg.global_variables    = linkonce constant %llvm.dbg.anchor.type  { uint 0, uint 52 } ;; DW_TAG_variable
    %llvm.dbg.subprograms         = linkonce constant %llvm.dbg.anchor.type  { uint 0, uint 46 } ;; DW_TAG_subprogram
    */
    if (!gIR->module->getNamedGlobal("llvm.dbg.compile_units")) {
        std::vector<LLConstant*> vals;
        vals.push_back(DtoConstUint(llvm::LLVMDebugVersion));
        vals.push_back(DtoConstUint(DW_TAG_compile_unit));
        LLConstant* i = llvm::ConstantStruct::get(t, vals);
        dbg_compile_units = new llvm::GlobalVariable(t,true,llvm::GlobalValue::LinkOnceLinkage,i,"llvm.dbg.compile_units",gIR->module);
        dbg_compile_units->setSection("llvm.metadata");
    }
    if (!gIR->module->getNamedGlobal("llvm.dbg.global_variables")) {
        std::vector<LLConstant*> vals;
        vals.push_back(DtoConstUint(llvm::LLVMDebugVersion));
        vals.push_back(DtoConstUint(DW_TAG_variable));
        LLConstant* i = llvm::ConstantStruct::get(t, vals);
        dbg_global_variables = new llvm::GlobalVariable(t,true,llvm::GlobalValue::LinkOnceLinkage,i,"llvm.dbg.global_variables",gIR->module);
        dbg_global_variables->setSection("llvm.metadata");
    }
    if (!gIR->module->getNamedGlobal("llvm.dbg.subprograms")) {
        std::vector<LLConstant*> vals;
        vals.push_back(DtoConstUint(llvm::LLVMDebugVersion));
        vals.push_back(DtoConstUint(DW_TAG_subprogram));
        LLConstant* i = llvm::ConstantStruct::get(t, vals);
        dbg_subprograms = new llvm::GlobalVariable(t,true,llvm::GlobalValue::LinkOnceLinkage,i,"llvm.dbg.subprograms",gIR->module);
        dbg_subprograms->setSection("llvm.metadata");
    }

    return t;
}

LLConstant* GetDwarfAnchor(llvm::dwarf::dwarf_constants c)
{
    GetDwarfAnchorType();
    switch (c)
    {
    case DW_TAG_compile_unit:
        return dbg_compile_units;
    case DW_TAG_variable:
        return dbg_global_variables;
    case DW_TAG_subprogram:
        return dbg_subprograms;
    }
    assert(0);
    return 0;
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

//////////////////////////////////////////////////////////////////////////////////////////////////

LLGlobalVariable* DtoDwarfCompileUnit(Module* m)
{
    if (!m->ir.irModule)
        m->ir.irModule = new IrModule(m);
    else if (m->ir.irModule->dwarfCompileUnit)
    {
        if (m->ir.irModule->dwarfCompileUnit->getParent() == gIR->module)
            return m->ir.irModule->dwarfCompileUnit;
    }

    // create a valid compile unit constant for the current module

    LLConstant* c = NULL;

    std::vector<LLConstant*> vals;
    vals.push_back(llvm::ConstantExpr::getAdd(
        DtoConstUint(DW_TAG_compile_unit),
        DtoConstUint(llvm::LLVMDebugVersion)));
    vals.push_back(DBG_CAST(GetDwarfAnchor(DW_TAG_compile_unit)));

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

    c = llvm::ConstantStruct::get(getDwarfCompileUnitType(), vals);

    llvm::GlobalVariable* gv = new llvm::GlobalVariable(c->getType(), true, llvm::GlobalValue::InternalLinkage, c, "llvm.dbg.compile_unit", gIR->module);
    gv->setSection("llvm.metadata");

    m->ir.irModule->dwarfCompileUnit = gv;
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

LLGlobalVariable* DtoDwarfSubProgram(FuncDeclaration* fd, llvm::GlobalVariable* compileUnit)
{
    std::vector<LLConstant*> vals;
    vals.push_back(llvm::ConstantExpr::getAdd(
        DtoConstUint(DW_TAG_subprogram),
        DtoConstUint(llvm::LLVMDebugVersion)));
    vals.push_back(DBG_CAST(GetDwarfAnchor(DW_TAG_subprogram)));

    vals.push_back(DBG_CAST(compileUnit));
    vals.push_back(DtoConstStringPtr(fd->toPrettyChars(), "llvm.metadata"));
    vals.push_back(vals.back());
    vals.push_back(DtoConstStringPtr(fd->mangle(), "llvm.metadata"));
    vals.push_back(DBG_CAST(compileUnit));
    vals.push_back(DtoConstUint(fd->loc.linnum));
    vals.push_back(DBG_NULL);
    vals.push_back(DtoConstBool(fd->protection == PROTprivate));
    vals.push_back(DtoConstBool(fd->getModule() == gIR->dmodule));

    LLConstant* c = llvm::ConstantStruct::get(getDwarfSubProgramType(), vals);
    llvm::GlobalVariable* gv = new llvm::GlobalVariable(c->getType(), true, llvm::GlobalValue::InternalLinkage, c, "llvm.dbg.subprogram", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncStart(FuncDeclaration* fd)
{
    assert(fd->ir.irFunc->dwarfSubProg);
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.func.start"), DBG_CAST(fd->ir.irFunc->dwarfSubProg));
}

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

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfTypeDescription(Loc loc, Type* type, LLGlobalVariable* cu, const char* c_name);

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfBasicType(Type* type, llvm::GlobalVariable* compileUnit)
{
    Type* t = type->toBasetype();

    const LLType* T = DtoType(type);

    std::vector<LLConstant*> vals;

    // tag
    vals.push_back(llvm::ConstantExpr::getAdd(
        DtoConstUint(DW_TAG_base_type),
        DtoConstUint(llvm::LLVMDebugVersion)));

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
            id = llvm::dwarf::DW_ATE_unsigned;
        else
            id = llvm::dwarf::DW_ATE_signed;
    }
    else if (t->isfloating())
    {
        id = llvm::dwarf::DW_ATE_float;
    }
    else
    {
        assert(0 && "unsupported basictype for debug info");
    }
    vals.push_back(DtoConstUint(id));

    LLConstant* c = llvm::ConstantStruct::get(getDwarfBasicTypeType(), vals);
    LLGlobalVariable* gv = new LLGlobalVariable(c->getType(), true, LLGlobalValue::InternalLinkage, c, "llvm.dbg.basictype", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
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
        tag = llvm::dwarf::DW_TAG_pointer_type;
    }
    else
    {
        assert(0 && "unsupported derivedtype for debug info");
    }

    std::vector<LLConstant*> vals;

    // tag
    vals.push_back(llvm::ConstantExpr::getAdd(
        DtoConstUint(tag),
        DtoConstUint(llvm::LLVMDebugVersion)));

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

    LLConstant* c = llvm::ConstantStruct::get(getDwarfDerivedTypeType(), vals);
    LLGlobalVariable* gv = new LLGlobalVariable(c->getType(), true, LLGlobalValue::InternalLinkage, c, "llvm.dbg.derivedtype", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
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
    vals.push_back(llvm::ConstantExpr::getAdd(
        DtoConstUint(llvm::dwarf::DW_TAG_member),
        DtoConstUint(llvm::LLVMDebugVersion)));

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

    LLConstant* c = llvm::ConstantStruct::get(getDwarfDerivedTypeType(), vals);
    LLGlobalVariable* gv = new LLGlobalVariable(c->getType(), true, LLGlobalValue::InternalLinkage, c, "llvm.dbg.derivedtype", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfCompositeType(Loc loc, Type* type, llvm::GlobalVariable* compileUnit)
{
    const LLType* T = DtoType(type);
    Type* t = DtoDType(type);

    // defaults
    LLConstant* name = getNullPtr(getVoidPtrType());
    LLGlobalVariable* members = NULL;

    // find tag
    unsigned tag;
    if (t->ty == Tarray)
    {
        tag = llvm::dwarf::DW_TAG_structure_type;

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
    vals.push_back(llvm::ConstantExpr::getAdd(
        DtoConstUint(tag),
        DtoConstUint(llvm::LLVMDebugVersion)));

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

    LLConstant* c = llvm::ConstantStruct::get(getDwarfCompositeTypeType(), vals);
    LLGlobalVariable* gv = new LLGlobalVariable(c->getType(), true, LLGlobalValue::InternalLinkage, c, "llvm.dbg.compositetype", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

static LLGlobalVariable* dwarfVariable(VarDeclaration* vd, LLGlobalVariable* typeDescr)
{
    unsigned tag;
    if (vd->isParameter())
        tag = DW_TAG_arg_variable;
    else if (vd->isCodeseg())
        assert(0 && "a static variable");
    else
        tag = DW_TAG_auto_variable;

    std::vector<LLConstant*> vals;
    // tag
    vals.push_back(llvm::ConstantExpr::getAdd(
        DtoConstUint(tag),
        DtoConstUint(llvm::LLVMDebugVersion)));
    // context
    vals.push_back(DBG_CAST(gIR->func()->dwarfSubProg));
    // name
    vals.push_back(DtoConstStringPtr(vd->toChars(), "llvm.metadata"));
    // compile unit where defined
    vals.push_back(DBG_CAST(DtoDwarfCompileUnit(vd->getModule())));
    // line number where defined
    vals.push_back(DtoConstUint(vd->loc.linnum));
    // type descriptor
    vals.push_back(DBG_CAST(typeDescr));

    LLConstant* c = llvm::ConstantStruct::get(getDwarfVariableType(), vals);
    LLGlobalVariable* gv = new LLGlobalVariable(c->getType(), true, LLGlobalValue::InternalLinkage, c, "llvm.dbg.variable", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
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














