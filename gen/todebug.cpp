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

static const llvm::PointerType* ptrTy(const LLType* t)
{
    return llvm::PointerType::get(t, 0);
}

static const llvm::PointerType* dbgArrTy()
{
    return ptrTy(llvm::StructType::get(NULL,NULL));
}

static LLConstant* dbgToArrTy(LLConstant* c)
{
    return llvm::ConstantExpr::getBitCast(c, dbgArrTy());
}

#define Ty(X) LLType::X

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

const llvm::StructType* GetDwarfCompileUnitType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.compile_unit.type"));
}

const llvm::StructType* GetDwarfSubProgramType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.subprogram.type"));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable* DtoDwarfCompileUnit(Module* m)
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
    vals.push_back(dbgToArrTy(GetDwarfAnchor(DW_TAG_compile_unit)));

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

    c = llvm::ConstantStruct::get(GetDwarfCompileUnitType(), vals);

    llvm::GlobalVariable* gv = new llvm::GlobalVariable(GetDwarfCompileUnitType(), true, llvm::GlobalValue::InternalLinkage, c, "llvm.dbg.compile_unit", gIR->module);
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
    vals.push_back(dbgToArrTy(GetDwarfAnchor(DW_TAG_subprogram)));

    vals.push_back(dbgToArrTy(compileUnit));
    vals.push_back(DtoConstStringPtr(fd->toPrettyChars(), "llvm.metadata"));
    vals.push_back(vals.back());
    vals.push_back(DtoConstStringPtr(fd->mangle(), "llvm.metadata"));
    vals.push_back(dbgToArrTy(compileUnit));
    vals.push_back(DtoConstUint(fd->loc.linnum));
    vals.push_back(llvm::ConstantPointerNull::get(dbgArrTy()));
    vals.push_back(DtoConstBool(fd->protection == PROTprivate));
    vals.push_back(DtoConstBool(fd->getModule() == gIR->dmodule));

    LLConstant* c = llvm::ConstantStruct::get(GetDwarfSubProgramType(), vals);
    llvm::GlobalVariable* gv = new llvm::GlobalVariable(c->getType(), true, llvm::GlobalValue::InternalLinkage, c, "llvm.dbg.subprogram", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfFuncStart(FuncDeclaration* fd)
{
    assert(fd->ir.irFunc->dwarfSubProg);
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.func.start"), dbgToArrTy(fd->ir.irFunc->dwarfSubProg));
}

void DtoDwarfFuncEnd(FuncDeclaration* fd)
{
    assert(fd->ir.irFunc->dwarfSubProg);
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.region.end"), dbgToArrTy(fd->ir.irFunc->dwarfSubProg));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfStopPoint(unsigned ln)
{
    LLSmallVector<LLValue*,3> args;
    args.push_back(DtoConstUint(ln));
    args.push_back(DtoConstUint(0));
    FuncDeclaration* fd = gIR->func()->decl;
    args.push_back(dbgToArrTy(DtoDwarfCompileUnit(fd->getModule())));
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.stoppoint"), args.begin(), args.end());
}

//////////////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* GetDwarfBasicTypeType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.basictype.type"));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

LLGlobalVariable* DtoDwarfBasicType(Type* type, llvm::GlobalVariable* compileUnit)
{
    const LLType* T = DtoType(type);

    std::vector<LLConstant*> vals;
    // tag
    vals.push_back(llvm::ConstantExpr::getAdd(
        DtoConstUint(DW_TAG_base_type),
        DtoConstUint(llvm::LLVMDebugVersion)));
    // context
    vals.push_back(dbgToArrTy(compileUnit));
    // name
    vals.push_back(DtoConstStringPtr(type->toChars(), "llvm.metadata"));
    // compile unit where defined
    vals.push_back(getNullPtr(dbgArrTy()));
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
    if (type->isintegral())
    {
        if (type->isunsigned())
            id = llvm::dwarf::DW_ATE_unsigned;
        else
            id = llvm::dwarf::DW_ATE_signed;
    }
    else if (type->isfloating())
    {
        id = llvm::dwarf::DW_ATE_float;
    }
    else
    {
        assert(0 && "unsupported basictype for debug info");
    }
    vals.push_back(DtoConstUint(id));

    LLConstant* c = llvm::ConstantStruct::get(GetDwarfBasicTypeType(), vals);
    LLGlobalVariable* gv = new LLGlobalVariable(c->getType(), true, LLGlobalValue::InternalLinkage, c, "llvm.dbg.basictype", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

const llvm::StructType* GetDwarfVariableType() {
    return isaStruct(gIR->module->getTypeByName("llvm.dbg.variable.type"));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

LLGlobalVariable* DtoDwarfVariable(VarDeclaration* vd, LLGlobalVariable* typeDescr)
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
    vals.push_back(dbgToArrTy(gIR->func()->dwarfSubProg));
    // name
    vals.push_back(DtoConstStringPtr(vd->toChars(), "llvm.metadata"));
    // compile unit where defined
    vals.push_back(dbgToArrTy(DtoDwarfCompileUnit(vd->getModule())));
    // line number where defined
    vals.push_back(DtoConstUint(vd->loc.linnum));
    // type descriptor
    vals.push_back(dbgToArrTy(typeDescr));

    LLConstant* c = llvm::ConstantStruct::get(GetDwarfVariableType(), vals);
    LLGlobalVariable* gv = new LLGlobalVariable(c->getType(), true, LLGlobalValue::InternalLinkage, c, "llvm.dbg.variable", gIR->module);
    gv->setSection("llvm.metadata");
    return gv;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void DtoDwarfDeclare(LLValue* var, LLGlobalVariable* varDescr)
{
    LLSmallVector<LLValue*,2> args;
    args.push_back(DtoBitCast(var, dbgArrTy()));
    args.push_back(dbgToArrTy(varDescr));
    gIR->ir->CreateCall(gIR->module->getFunction("llvm.dbg.declare"), args.begin(), args.end());
}
