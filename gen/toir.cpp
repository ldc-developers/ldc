// Backend stubs

/* DMDFE backend stubs
 * This file contains the implementations of the backend routines.
 * For dmdfe these do nothing but print a message saying the module
 * has been parsed. Substitute your own behaviors for these routimes.
 */

#include <stdio.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <iostream>

#include "gen/llvm.h"

#include "attrib.h"
#include "total.h"
#include "init.h"
#include "mtype.h"
#include "hdrgen.h"
#include "port.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/runtime.h"
#include "gen/arrays.h"
#include "gen/structs.h"
#include "gen/classes.h"
#include "gen/typeinf.h"
#include "gen/complex.h"
#include "gen/dvalue.h"
#include "gen/aa.h"
#include "gen/functions.h"

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DeclarationExp::toElem(IRState* p)
{
    Logger::print("DeclarationExp::toElem: %s | T=%s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // variable declaration
    if (VarDeclaration* vd = declaration->isVarDeclaration())
    {
        Logger::println("VarDeclaration");

        // static
        if (vd->isDataseg())
        {
            vd->toObjFile(); // TODO
        }
        else
        {
            if (global.params.llvmAnnotate)
                DtoAnnotation(toChars());

            Logger::println("vdtype = %s", vd->type->toChars());

            // referenced by nested delegate?
            if (vd->nestedref) {
                Logger::println("has nestedref set");
                assert(vd->ir.irLocal);
                vd->ir.irLocal->value = p->func()->decl->ir.irFunc->nestedVar;
                assert(vd->ir.irLocal->value);
                assert(vd->ir.irLocal->nestedIndex >= 0);
            }
            // normal stack variable
            else {
                // allocate storage on the stack
                const llvm::Type* lltype = DtoType(vd->type);
                llvm::AllocaInst* allocainst = new llvm::AllocaInst(lltype, vd->toChars(), p->topallocapoint());
                //allocainst->setAlignment(vd->type->alignsize()); // TODO
                assert(!vd->ir.irLocal);
                vd->ir.irLocal = new IrLocal(vd);
                vd->ir.irLocal->value = allocainst;
            }

            Logger::cout() << "llvm value for decl: " << *vd->ir.irLocal->value << '\n';
            DValue* ie = DtoInitializer(vd->init);
        }

        return new DVarValue(vd, vd->ir.getIrValue(), true);
    }
    // struct declaration
    else if (StructDeclaration* s = declaration->isStructDeclaration())
    {
        Logger::println("StructDeclaration");
        DtoForceConstInitDsymbol(s);
    }
    // function declaration
    else if (FuncDeclaration* f = declaration->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        DtoForceDeclareDsymbol(f);
    }
    // alias declaration
    else if (AliasDeclaration* a = declaration->isAliasDeclaration())
    {
        Logger::println("AliasDeclaration - no work");
        // do nothing
    }
    // enum
    else if (EnumDeclaration* e = declaration->isEnumDeclaration())
    {
        Logger::println("EnumDeclaration - no work");
        // do nothing
    }
    // class
    else if (ClassDeclaration* e = declaration->isClassDeclaration())
    {
        Logger::println("ClassDeclaration");
        DtoForceConstInitDsymbol(e);
    }
    // typedef
    else if (TypedefDeclaration* tdef = declaration->isTypedefDeclaration())
    {
        Logger::println("TypedefDeclaration");
        tdef->type->getTypeInfo(NULL);
    }
    // attribute declaration
    else if (AttribDeclaration* a = declaration->isAttribDeclaration())
    {
        Logger::println("AttribDeclaration");
        for (int i=0; i < a->decl->dim; ++i)
        {
            DtoForceDeclareDsymbol((Dsymbol*)a->decl->data[i]);
        }
    }
    // unsupported declaration
    else
    {
        error("Unimplemented DeclarationExp type. kind: %s", declaration->kind());
        assert(0);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* VarExp::toElem(IRState* p)
{
    Logger::print("VarExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(var);
    if (VarDeclaration* vd = var->isVarDeclaration())
    {
        Logger::println("VarDeclaration %s", vd->toChars());

        // _arguments
        if (vd->ident == Id::_arguments)
        {
            Logger::println("Id::_arguments");
            /*if (!vd->ir.getIrValue())
                vd->ir.getIrValue() = p->func()->decl->irFunc->_arguments;
            assert(vd->ir.getIrValue());
            return new DVarValue(vd, vd->ir.getIrValue(), true);*/
            llvm::Value* v = p->func()->decl->ir.irFunc->_arguments;
            assert(v);
            return new DVarValue(vd, v, true);
        }
        // _argptr
        else if (vd->ident == Id::_argptr)
        {
            Logger::println("Id::_argptr");
            /*if (!vd->ir.getIrValue())
                vd->ir.getIrValue() = p->func()->decl->irFunc->_argptr;
            assert(vd->ir.getIrValue());
            return new DVarValue(vd, vd->ir.getIrValue(), true);*/
            llvm::Value* v = p->func()->decl->ir.irFunc->_argptr;
            assert(v);
            return new DVarValue(vd, v, true);
        }
        // _dollar
        else if (vd->ident == Id::dollar)
        {
            Logger::println("Id::dollar");
            assert(!p->arrays.empty());
            llvm::Value* tmp = DtoArrayLen(p->arrays.back());
            return new DVarValue(vd, tmp, false);
        }
        // typeinfo
        else if (TypeInfoDeclaration* tid = vd->isTypeInfoDeclaration())
        {
            Logger::println("TypeInfoDeclaration");
            DtoForceDeclareDsymbol(tid);
            assert(tid->ir.getIrValue());
            const llvm::Type* vartype = DtoType(type);
            llvm::Value* m;
            if (tid->ir.getIrValue()->getType() != getPtrToType(vartype))
                m = p->ir->CreateBitCast(tid->ir.getIrValue(), vartype, "tmp");
            else
                m = tid->ir.getIrValue();
            return new DVarValue(vd, m, true);
        }
        // classinfo
        else if (ClassInfoDeclaration* cid = vd->isClassInfoDeclaration())
        {
            Logger::println("ClassInfoDeclaration: %s", cid->cd->toChars());
            DtoDeclareClassInfo(cid->cd);
            assert(cid->cd->ir.irStruct->classInfo);
            return new DVarValue(vd, cid->cd->ir.irStruct->classInfo, true);
        }
        // nested variable
        else if (vd->nestedref) {
            Logger::println("nested variable");
            return new DVarValue(vd, DtoNestedVariable(vd), true);
        }
        // function parameter
        else if (vd->isParameter()) {
            Logger::println("function param");
            if (vd->isRef() || vd->isOut() || DtoIsPassedByRef(vd->type) || llvm::isa<llvm::AllocaInst>(vd->ir.getIrValue())) {
                return new DVarValue(vd, vd->ir.getIrValue(), true);
            }
            else if (llvm::isa<llvm::Argument>(vd->ir.getIrValue())) {
                return new DImValue(type, vd->ir.getIrValue());
            }
            else assert(0);
        }
        else {
            // take care of forward references of global variables
            if (vd->isDataseg() || (vd->storage_class & STCextern)) {
                vd->toObjFile();
                DtoConstInitGlobal(vd);
            }
            if (!vd->ir.getIrValue() || DtoType(vd->type)->isAbstract()) {
                Logger::println("global variable not resolved :/ %s", vd->toChars());
                Logger::cout() << *DtoType(vd->type) << '\n';
                assert(0);
            }
            return new DVarValue(vd, vd->ir.getIrValue(), true);
        }
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        if (fdecl->llvmInternal != LLVMva_arg) {// && fdecl->llvmValue == 0)
            DtoForceDeclareDsymbol(fdecl);
        }
        return new DFuncValue(fdecl, fdecl->ir.irFunc->func);
    }
    else if (SymbolDeclaration* sdecl = var->isSymbolDeclaration())
    {
        // this seems to be the static initialiser for structs
        Type* sdecltype = DtoDType(sdecl->type);
        Logger::print("Sym: type=%s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = (TypeStruct*)sdecltype;
        assert(ts->sym);
        DtoForceConstInitDsymbol(ts->sym);
        assert(ts->sym->ir.irStruct->init);
        return new DVarValue(type, ts->sym->ir.irStruct->init, true);
    }
    else
    {
        assert(0 && "Unimplemented VarExp type");
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* VarExp::toConstElem(IRState* p)
{
    Logger::print("VarExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    if (SymbolDeclaration* sdecl = var->isSymbolDeclaration())
    {
        // this seems to be the static initialiser for structs
        Type* sdecltype = DtoDType(sdecl->type);
        Logger::print("Sym: type=%s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = (TypeStruct*)sdecltype;
        DtoForceConstInitDsymbol(ts->sym);
        assert(ts->sym->ir.irStruct->constInit);
        return ts->sym->ir.irStruct->constInit;
    }
    else if (TypeInfoDeclaration* ti = var->isTypeInfoDeclaration())
    {
        DtoForceDeclareDsymbol(ti);
        assert(ti->ir.getIrValue());
        const llvm::Type* vartype = DtoType(type);
        llvm::Constant* m = isaConstant(ti->ir.getIrValue());
        assert(m);
        if (ti->ir.getIrValue()->getType() != getPtrToType(vartype))
            m = llvm::ConstantExpr::getBitCast(m, vartype);
        return m;
    }
    assert(0 && "Unsupported const VarExp kind");
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* IntegerExp::toElem(IRState* p)
{
    Logger::print("IntegerExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    llvm::Constant* c = toConstElem(p);
    return new DConstValue(type, c);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* IntegerExp::toConstElem(IRState* p)
{
    Logger::print("IntegerExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    const llvm::Type* t = DtoType(type);
    if (isaPointer(t)) {
        Logger::println("pointer");
        llvm::Constant* i = llvm::ConstantInt::get(DtoSize_t(),(uint64_t)value,false);
        return llvm::ConstantExpr::getIntToPtr(i, t);
    }
    assert(llvm::isa<llvm::IntegerType>(t));
    llvm::Constant* c = llvm::ConstantInt::get(t,(uint64_t)value,!type->isunsigned());
    assert(c);
    Logger::cout() << "value = " << *c << '\n';
    return c;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* RealExp::toElem(IRState* p)
{
    Logger::print("RealExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    llvm::Constant* c = toConstElem(p);
    return new DConstValue(type, c);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* RealExp::toConstElem(IRState* p)
{
    Logger::print("RealExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    Type* t = DtoDType(type);
    return DtoConstFP(t, value);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* NullExp::toElem(IRState* p)
{
    Logger::print("NullExp::toElem(type=%s): %s\n", type->toChars(),toChars());
    LOG_SCOPE;
    llvm::Constant* c = toConstElem(p);
    return new DNullValue(type, c);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* NullExp::toConstElem(IRState* p)
{
    Logger::print("NullExp::toConstElem(type=%s): %s\n", type->toChars(),toChars());
    LOG_SCOPE;
    const llvm::Type* t = DtoType(type);
    if (type->ty == Tarray) {
        assert(isaStruct(t));
        return llvm::ConstantAggregateZero::get(t);
    }
    else {
        return llvm::Constant::getNullValue(t);
    }
    assert(0);
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ComplexExp::toElem(IRState* p)
{
    Logger::print("ComplexExp::toElem(): %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    llvm::Constant* c = toConstElem(p);

    if (c->isNullValue()) {
        Type* t = DtoDType(type);
        if (t->ty == Tcomplex32)
            c = DtoConstFP(Type::tfloat32, 0);
        else
            c = DtoConstFP(Type::tfloat64, 0);
        return new DComplexValue(type, c, c);
    }

    return new DComplexValue(type, c->getOperand(0), c->getOperand(1));
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* ComplexExp::toConstElem(IRState* p)
{
    Logger::print("ComplexExp::toConstElem(): %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    return DtoConstComplex(type, value.re, value.im);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* StringExp::toElem(IRState* p)
{
    Logger::print("StringExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* dtype = DtoDType(type);
    Type* cty = DtoDType(dtype->next);

    const llvm::Type* ct = DtoType(cty);
    if (ct == llvm::Type::VoidTy)
        ct = llvm::Type::Int8Ty;
    //printf("ct = %s\n", type->next->toChars());
    const llvm::ArrayType* at = llvm::ArrayType::get(ct,len+1);

    llvm::Constant* _init;
    if (cty->size() == 1) {
        uint8_t* str = (uint8_t*)string;
        std::string cont((char*)str, len);
        _init = llvm::ConstantArray::get(cont,true);
    }
    else if (cty->size() == 2) {
        uint16_t* str = (uint16_t*)string;
        std::vector<llvm::Constant*> vals;
        for(size_t i=0; i<len; ++i) {
            vals.push_back(llvm::ConstantInt::get(ct, str[i], false));;
        }
        vals.push_back(llvm::ConstantInt::get(ct, 0, false));
        _init = llvm::ConstantArray::get(at,vals);
    }
    else if (cty->size() == 4) {
        uint32_t* str = (uint32_t*)string;
        std::vector<llvm::Constant*> vals;
        for(size_t i=0; i<len; ++i) {
            vals.push_back(llvm::ConstantInt::get(ct, str[i], false));;
        }
        vals.push_back(llvm::ConstantInt::get(ct, 0, false));
        _init = llvm::ConstantArray::get(at,vals);
    }
    else
    assert(0);

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;//WeakLinkage;
    Logger::cout() << "type: " << *at << "\ninit: " << *_init << '\n';
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(at,true,_linkage,_init,"stringliteral",gIR->module);

    llvm::ConstantInt* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Constant* idxs[2] = { zero, zero };
    llvm::Constant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);

    if (dtype->ty == Tarray) {
        llvm::Constant* clen = llvm::ConstantInt::get(DtoSize_t(),len,false);
        if (!p->topexp() || p->topexp()->e2 != this) {
            llvm::Value* tmpmem = new llvm::AllocaInst(DtoType(dtype),"tempstring",p->topallocapoint());
            DtoSetArray(tmpmem, clen, arrptr);
            return new DVarValue(type, tmpmem, true);
        }
        else if (p->topexp()->e2 == this) {
            DValue* arr = p->topexp()->v;
            assert(arr);
            if (arr->isSlice()) {
                return new DSliceValue(type, clen, arrptr);
            }
            else {
                DtoSetArray(arr->getRVal(), clen, arrptr);
                return new DImValue(type, arr->getLVal(), true);
            }
        }
        assert(0);
    }
    else if (dtype->ty == Tsarray) {
        const llvm::Type* dstType = getPtrToType(llvm::ArrayType::get(ct, len));
        llvm::Value* emem = (gvar->getType() == dstType) ? gvar : DtoBitCast(gvar, dstType);
        return new DVarValue(type, emem, true);
    }
    else if (dtype->ty == Tpointer) {
        return new DImValue(type, arrptr);
    }

    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* StringExp::toConstElem(IRState* p)
{
    Logger::print("StringExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* t = DtoDType(type);
    Type* cty = DtoDType(t->next);

    bool nullterm = (t->ty != Tsarray);
    size_t endlen = nullterm ? len+1 : len;

    const llvm::Type* ct = DtoType(cty);
    const llvm::ArrayType* at = llvm::ArrayType::get(ct,endlen);

    llvm::Constant* _init;
    if (cty->size() == 1) {
        uint8_t* str = (uint8_t*)string;
        std::string cont((char*)str, len);
        _init = llvm::ConstantArray::get(cont, nullterm);
    }
    else if (cty->size() == 2) {
        uint16_t* str = (uint16_t*)string;
        std::vector<llvm::Constant*> vals;
        for(size_t i=0; i<len; ++i) {
            vals.push_back(llvm::ConstantInt::get(ct, str[i], false));;
        }
        if (nullterm)
            vals.push_back(llvm::ConstantInt::get(ct, 0, false));
        _init = llvm::ConstantArray::get(at,vals);
    }
    else if (cty->size() == 4) {
        uint32_t* str = (uint32_t*)string;
        std::vector<llvm::Constant*> vals;
        for(size_t i=0; i<len; ++i) {
            vals.push_back(llvm::ConstantInt::get(ct, str[i], false));;
        }
        if (nullterm)
            vals.push_back(llvm::ConstantInt::get(ct, 0, false));
        _init = llvm::ConstantArray::get(at,vals);
    }
    else
    assert(0);

    if (t->ty == Tsarray)
    {
        return _init;
    }

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;//WeakLinkage;
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(_init->getType(),true,_linkage,_init,"stringliteral",gIR->module);

    llvm::ConstantInt* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Constant* idxs[2] = { zero, zero };
    llvm::Constant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);

    if (t->ty == Tpointer) {
        return arrptr;
    }
    else if (t->ty == Tarray) {
        llvm::Constant* clen = llvm::ConstantInt::get(DtoSize_t(),len,false);
        return DtoConstSlice(clen, arrptr);
    }

    assert(0);
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AssignExp::toElem(IRState* p)
{
    Logger::print("AssignExp::toElem: %s | %s = %s\n", toChars(), e1->type->toChars(), e2->type ? e2->type->toChars() : 0);
    LOG_SCOPE;

    p->exps.push_back(IRExp(e1,e2,NULL));

    DValue* l = e1->toElem(p);
    p->topexp()->v = l;
    DValue* r = e2->toElem(p);

    p->exps.pop_back();

    Logger::println("performing assignment");

    DImValue* im = r->isIm();
    if (!im || !im->inPlace()) {
        Logger::println("assignment not inplace");
        if (DArrayLenValue* al = l->isArrayLen())
        {
            DSliceValue* slice = DtoResizeDynArray(l->getType(), l, r);
            DtoAssign(l, slice);
        }
        else
        {
            DtoAssign(l, r);
        }
    }

    if (l->isSlice() || l->isComplex())
        return l;

    llvm::Value* v;
    if (l->isVar() && l->isVar()->lval)
        v = l->getLVal();
    else
        v = l->getRVal();

    return new DVarValue(type, v, true);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AddExp::toElem(IRState* p)
{
    Logger::print("AddExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = DtoDType(type);
    Type* e1type = DtoDType(e1->type);
    Type* e1next = e1type->next ? DtoDType(e1type->next) : NULL;
    Type* e2type = DtoDType(e2->type);

    if (e1type != e2type) {
        if (llvmFieldIndex) {
            assert(e1type->ty == Tpointer && e1next && e1next->ty == Tstruct);
            Logger::println("add to AddrExp of struct");
            assert(r->isConst());
            llvm::ConstantInt* cofs = llvm::cast<llvm::ConstantInt>(r->isConst()->c);

            TypeStruct* ts = (TypeStruct*)e1next;
            std::vector<unsigned> offsets;
            llvm::Value* v = DtoIndexStruct(l->getRVal(), ts->sym, t->next, cofs->getZExtValue(), offsets);
            return new DFieldValue(type, v, true);
        }
        else if (e1type->ty == Tpointer) {
            Logger::println("add to pointer");
            if (r->isConst()) {
                llvm::ConstantInt* cofs = llvm::cast<llvm::ConstantInt>(r->isConst()->c);
                if (cofs->isZero()) {
                    Logger::println("is zero");
                    return new DImValue(type, l->getRVal());
                }
            }
            llvm::Value* v = llvm::GetElementPtrInst::Create(l->getRVal(), r->getRVal(), "tmp", p->scopebb());
            return new DImValue(type, v);
        }
        else if (t->iscomplex()) {
            return DtoComplexAdd(type, l, r);
        }
        assert(0);
    }
    else if (t->iscomplex()) {
        return DtoComplexAdd(type, l, r);
    }
    else {
        return DtoBinAdd(l,r);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AddAssignExp::toElem(IRState* p)
{
    Logger::print("AddAssignExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    p->exps.push_back(IRExp(e1,e2,NULL));
    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);
    p->exps.pop_back();

    Type* t = DtoDType(type);

    DValue* res;
    if (DtoDType(e1->type)->ty == Tpointer) {
        llvm::Value* gep = llvm::GetElementPtrInst::Create(l->getRVal(),r->getRVal(),"tmp",p->scopebb());
        res = new DImValue(type, gep);
    }
    else if (t->iscomplex()) {
        res = DtoComplexAdd(e1->type, l, r);
    }
    else {
        res = DtoBinAdd(l,r);
    }
    DtoAssign(l, res);

    // used as lvalue :/
    if (p->topexp() && p->topexp()->e1 == this)
    {
        assert(!l->isLRValue());
        return l;
    }
    else
    {
        return res;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* MinExp::toElem(IRState* p)
{
    Logger::print("MinExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = DtoDType(type);
    Type* t1 = DtoDType(e1->type);
    Type* t2 = DtoDType(e2->type);

    if (t1->ty == Tpointer && t2->ty == Tpointer) {
        llvm::Value* lv = l->getRVal();
        llvm::Value* rv = r->getRVal();
        Logger::cout() << "lv: " << *lv << " rv: " << *rv << '\n';
        lv = p->ir->CreatePtrToInt(lv, DtoSize_t(), "tmp");
        rv = p->ir->CreatePtrToInt(rv, DtoSize_t(), "tmp");
        llvm::Value* diff = p->ir->CreateSub(lv,rv,"tmp");
        if (diff->getType() != DtoType(type))
            diff = p->ir->CreateIntToPtr(diff, DtoType(type), "tmp");
        return new DImValue(type, diff);
    }
    else if (t1->ty == Tpointer) {
        llvm::Value* idx = p->ir->CreateNeg(r->getRVal(), "tmp");
        llvm::Value* v = llvm::GetElementPtrInst::Create(l->getRVal(), idx, "tmp", p->scopebb());
        return new DImValue(type, v);
    }
    else if (t->iscomplex()) {
        return DtoComplexSub(type, l, r);
    }
    else {
        return DtoBinSub(l,r);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* MinAssignExp::toElem(IRState* p)
{
    Logger::print("MinAssignExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = DtoDType(type);

    DValue* res;
    if (DtoDType(e1->type)->ty == Tpointer) {
        Logger::println("ptr");
        llvm::Value* tmp = r->getRVal();
        llvm::Value* zero = llvm::ConstantInt::get(tmp->getType(),0,false);
        tmp = llvm::BinaryOperator::createSub(zero,tmp,"tmp",p->scopebb());
        tmp = llvm::GetElementPtrInst::Create(l->getRVal(),tmp,"tmp",p->scopebb());
        res = new DImValue(type, tmp);
    }
    else if (t->iscomplex()) {
        Logger::println("complex");
        res = DtoComplexSub(type, l, r);
    }
    else {
        Logger::println("basic");
        res = DtoBinSub(l,r);
    }
    DtoAssign(l, res);

    return l;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* MulExp::toElem(IRState* p)
{
    Logger::print("MulExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    if (type->iscomplex()) {
        return DtoComplexMul(type, l, r);
    }

    return DtoBinMul(l,r);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* MulAssignExp::toElem(IRState* p)
{
    Logger::print("MulAssignExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    DValue* res;
    if (type->iscomplex()) {
        res = DtoComplexMul(type, l, r);
    }
    else {
        res = DtoBinMul(l,r);
    }
    DtoAssign(l, res);

    return l;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DivExp::toElem(IRState* p)
{
    Logger::print("DivExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    if (type->iscomplex()) {
        return DtoComplexDiv(type, l, r);
    }

    return DtoBinDiv(l, r);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DivAssignExp::toElem(IRState* p)
{
    Logger::print("DivAssignExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    DValue* res;
    if (type->iscomplex()) {
        res = DtoComplexDiv(type, l, r);
    }
    else {
        res = DtoBinDiv(l,r);
    }
    DtoAssign(l, res);

    return l;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ModExp::toElem(IRState* p)
{
    Logger::print("ModExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    return DtoBinRem(l, r);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ModAssignExp::toElem(IRState* p)
{
    Logger::print("ModAssignExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    DValue* res = DtoBinRem(l, r);
    DtoAssign(l, res);

    return l;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CallExp::toElem(IRState* p)
{
    Logger::print("CallExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* fn = e1->toElem(p);

    TypeFunction* tf = 0;
    Type* e1type = DtoDType(e1->type);

    bool delegateCall = false;
    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty,0,false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty,1,false);
    LINK dlink = LINKd;

    // hidden struct return parameter handling
    bool retinptr = false;

    // regular functions
    if (e1type->ty == Tfunction) {
        tf = (TypeFunction*)e1type;
        if (tf->llvmRetInPtr) {
            retinptr = true;
        }
        dlink = tf->linkage;
    }

    // delegates
    else if (e1type->ty == Tdelegate) {
        Logger::println("delegateTy = %s\n", e1type->toChars());
        assert(e1type->next->ty == Tfunction);
        tf = (TypeFunction*)e1type->next;
        if (tf->llvmRetInPtr) {
            retinptr = true;
        }
        dlink = tf->linkage;
        delegateCall = true;
    }

    // invalid
    else {
        assert(tf);
    }

    // magic stuff
    bool va_magic = false;
    bool va_intrinsic = false;
    DFuncValue* dfv = fn->isFunc();
    if (dfv && dfv->func) {
        FuncDeclaration* fndecl = dfv->func;
        if (fndecl->llvmInternal == LLVMva_intrinsic) {
            va_magic = true;
            va_intrinsic = true;
        }
        else if (fndecl->llvmInternal == LLVMva_start) {
            va_magic = true;
        }
        else if (fndecl->llvmInternal == LLVMva_arg) {
            //Argument* fnarg = Argument::getNth(tf->parameters, 0);
            Expression* exp = (Expression*)arguments->data[0];
            DValue* expelem = exp->toElem(p);
            Type* t = DtoDType(type);
            const llvm::Type* llt = DtoType(type);
            if (DtoIsPassedByRef(t))
                llt = getPtrToType(llt);
            // TODO
            if (strcmp(global.params.llvmArch, "x86") != 0) {
                warning("%s: va_arg for C variadic functions is broken for anything but x86", loc.toChars());
            }
            return new DImValue(type, p->ir->CreateVAArg(expelem->getLVal(),llt,"tmp"));
        }
        else if (fndecl->llvmInternal == LLVMalloca) {
            //Argument* fnarg = Argument::getNth(tf->parameters, 0);
            Expression* exp = (Expression*)arguments->data[0];
            DValue* expv = exp->toElem(p);
            if (expv->getType()->toBasetype()->ty != Tint32)
                expv = DtoCast(expv, Type::tint32);
            llvm::Value* alloc = new llvm::AllocaInst(llvm::Type::Int8Ty, expv->getRVal(), "alloca", p->scopebb());
            return new DImValue(type, alloc);
        }
    }

    // args
    size_t n = arguments->dim;
    DFuncValue* dfn = fn->isFunc();
    if (dfn && dfn->func && dfn->func->llvmInternal == LLVMva_start)
        n = 1;
    if (delegateCall || (dfn && dfn->vthis)) n++;
    if (retinptr) n++;
    if (tf->linkage == LINKd && tf->varargs == 1) n+=2;
    if (dfn && dfn->func && dfn->func->isNested()) n++;

    llvm::Value* funcval = fn->getRVal();
    assert(funcval != 0);
    std::vector<llvm::Value*> llargs(n, 0);

    const llvm::FunctionType* llfnty = 0;

    // normal function call
    if (llvm::isa<llvm::FunctionType>(funcval->getType())) {
        llfnty = llvm::cast<llvm::FunctionType>(funcval->getType());
    }
    // pointer to something
    else if (isaPointer(funcval->getType())) {
        // pointer to function pointer - I think this not really supposed to happen, but does :/
        // seems like sometimes we get a func* other times a func**
        if (isaPointer(funcval->getType()->getContainedType(0))) {
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
        }
        // function pointer
        if (llvm::isa<llvm::FunctionType>(funcval->getType()->getContainedType(0))) {
            //Logger::cout() << "function pointer type:\n" << *funcval << '\n';
            llfnty = llvm::cast<llvm::FunctionType>(funcval->getType()->getContainedType(0));
        }
        // struct pointer - delegate
        else if (isaStruct(funcval->getType()->getContainedType(0))) {
            funcval = DtoGEP(funcval,zero,one,"tmp",p->scopebb());
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
            const llvm::Type* ty = funcval->getType()->getContainedType(0);
            llfnty = llvm::cast<llvm::FunctionType>(ty);
        }
        // unknown
        else {
            Logger::cout() << "what kind of pointer are we calling? : " << *funcval->getType() << '\n';
        }
    }
    else {
        Logger::cout() << "what are we calling? : " << *funcval << '\n';
    }
    assert(llfnty);
    //Logger::cout() << "Function LLVM type: " << *llfnty << '\n';

    // argument handling
    llvm::FunctionType::param_iterator argiter = llfnty->param_begin();
    int j = 0;

    IRExp* topexp = p->topexp();

    bool isInPlace = false;

    // hidden struct return arguments
    if (retinptr) {
        if (topexp && topexp->e2 == this) {
            assert(topexp->v);
            llvm::Value* tlv = topexp->v->getLVal();
            assert(isaStruct(tlv->getType()->getContainedType(0)));
            llargs[j] = tlv;
            isInPlace = true;
            /*if (DtoIsPassedByRef(tf->next)) {
                isInPlace = true;
            }
            else
            assert(0);*/
        }
        else {
            llargs[j] = new llvm::AllocaInst(argiter->get()->getContainedType(0),"rettmp",p->topallocapoint());
        }

        if (dfn && dfn->func && dfn->func->runTimeHack) {
            const llvm::Type* rettype = getPtrToType(DtoType(type));
            if (llargs[j]->getType() != llfnty->getParamType(j)) {
                Logger::println("llvmRunTimeHack==true - force casting return value param");
                Logger::cout() << "casting: " << *llargs[j] << " to type: " << *llfnty->getParamType(j) << '\n';
                llargs[j] = DtoBitCast(llargs[j], llfnty->getParamType(j));
            }
        }

        ++j;
        ++argiter;
    }

    // this arguments
    if (dfn && dfn->vthis) {
        Logger::cout() << "This Call" << '\n';// func val:" << *funcval << '\n';
        if (dfn->vthis->getType() != argiter->get()) {
            //Logger::cout() << "value: " << *dfn->vthis << " totype: " << *argiter->get() << '\n';
            llargs[j] = DtoBitCast(dfn->vthis, argiter->get());
        }
        else {
            llargs[j] = dfn->vthis;
        }
        ++j;
        ++argiter;
    }
    // delegate context arguments
    else if (delegateCall) {
        Logger::println("Delegate Call");
        llvm::Value* contextptr = DtoGEP(fn->getRVal(),zero,zero,"tmp",p->scopebb());
        llargs[j] = new llvm::LoadInst(contextptr,"tmp",p->scopebb());
        ++j;
        ++argiter;
    }
    // nested call
    else if (dfn && dfn->func && dfn->func->isNested()) {
        Logger::println("Nested Call");
        llvm::Value* contextptr = DtoNestedContext(dfn->func->toParent2()->isFuncDeclaration());
        if (!contextptr)
            contextptr = llvm::ConstantPointerNull::get(getPtrToType(llvm::Type::Int8Ty));
        llargs[j] = DtoBitCast(contextptr, getPtrToType(llvm::Type::Int8Ty));
        ++j;
        ++argiter;
    }

    // va arg function special argument passing
    if (va_magic)
    {
        size_t n = va_intrinsic ? arguments->dim : 1;
        for (int i=0; i<n; i++,j++)
        {
            Argument* fnarg = Argument::getNth(tf->parameters, i);
            Expression* exp = (Expression*)arguments->data[i];
            DValue* expelem = exp->toElem(p);
            llargs[j] = DtoBitCast(expelem->getLVal(), getPtrToType(llvm::Type::Int8Ty));
        }
    }
    // regular arguments
    else
    {
        // d variadic function?
        if (tf->linkage == LINKd && tf->varargs == 1)
        {
            Logger::println("doing d-style variadic arguments");

            size_t nimplicit = j;

            std::vector<const llvm::Type*> vtypes;
            std::vector<llvm::Value*> vtypeinfos;

            // number of non variadic args
            int begin = tf->parameters->dim;
            Logger::println("num non vararg params = %d", begin);

            // build struct with argument types
            for (int i=begin; i<arguments->dim; i++)
            {
                Argument* argu = Argument::getNth(tf->parameters, i);
                Expression* argexp = (Expression*)arguments->data[i];
                vtypes.push_back(DtoType(argexp->type));
            }
            const llvm::StructType* vtype = llvm::StructType::get(vtypes);
            Logger::cout() << "d-variadic argument struct type:\n" << *vtype << '\n';
            llvm::Value* mem = new llvm::AllocaInst(vtype,"_argptr_storage",p->topallocapoint());

            // store arguments in the struct
            for (int i=begin,k=0; i<arguments->dim; i++,k++)
            {
                Expression* argexp = (Expression*)arguments->data[i];
                if (global.params.llvmAnnotate)
                    DtoAnnotation(argexp->toChars());
                DtoVariadicArgument(argexp, DtoGEPi(mem,0,k,"tmp"));
            }

            // build type info array
            assert(Type::typeinfo->ir.irStruct->constInit);
            const llvm::Type* typeinfotype = DtoType(Type::typeinfo->type);
            const llvm::ArrayType* typeinfoarraytype = llvm::ArrayType::get(typeinfotype,vtype->getNumElements());

            llvm::Value* typeinfomem = new llvm::AllocaInst(typeinfoarraytype,"_arguments_storage",p->topallocapoint());
            Logger::cout() << "_arguments storage: " << *typeinfomem << '\n';
            for (int i=begin,k=0; i<arguments->dim; i++,k++)
            {
                Expression* argexp = (Expression*)arguments->data[i];
                TypeInfoDeclaration* tidecl = argexp->type->getTypeInfoDeclaration();
                DtoForceDeclareDsymbol(tidecl);
                assert(tidecl->ir.getIrValue());
                vtypeinfos.push_back(tidecl->ir.getIrValue());
                llvm::Value* v = p->ir->CreateBitCast(vtypeinfos[k], typeinfotype, "tmp");
                p->ir->CreateStore(v, DtoGEPi(typeinfomem,0,k,"tmp"));
            }

            // put data in d-array
            llvm::Value* typeinfoarrayparam = new llvm::AllocaInst(llfnty->getParamType(j)->getContainedType(0),"_arguments_array",p->topallocapoint());
            p->ir->CreateStore(DtoConstSize_t(vtype->getNumElements()), DtoGEPi(typeinfoarrayparam,0,0,"tmp"));
            llvm::Value* casttypeinfomem = p->ir->CreateBitCast(typeinfomem, getPtrToType(typeinfotype), "tmp");
            p->ir->CreateStore(casttypeinfomem, DtoGEPi(typeinfoarrayparam,0,1,"tmp"));

            // specify arguments
            llargs[j] = typeinfoarrayparam;;
            j++;
            llargs[j] = p->ir->CreateBitCast(mem, getPtrToType(llvm::Type::Int8Ty), "tmp");
            j++;

            // pass non variadic args
            for (int i=0; i<begin; i++)
            {
                Argument* fnarg = Argument::getNth(tf->parameters, i);
                DValue* argval = DtoArgument(fnarg, (Expression*)arguments->data[i]);
                llargs[j] = argval->getRVal();
                j++;
            }

            // make sure arg vector has the right size
            llargs.resize(nimplicit+begin+2);
        }
        // normal function
        else {
            Logger::println("doing normal arguments");
            for (int i=0; i<arguments->dim; i++,j++) {
                Argument* fnarg = Argument::getNth(tf->parameters, i);
                if (global.params.llvmAnnotate)
                    DtoAnnotation(((Expression*)arguments->data[i])->toChars());
                DValue* argval = DtoArgument(fnarg, (Expression*)arguments->data[i]);
                llargs[j] = argval->getRVal();
                if (fnarg && llargs[j]->getType() != llfnty->getParamType(j)) {
                    llargs[j] = DtoBitCast(llargs[j], llfnty->getParamType(j));
                }

                // this hack is necessary :/
                if (dfn && dfn->func && dfn->func->runTimeHack) {
                    if (llfnty->getParamType(j) != NULL) {
                        if (llargs[j]->getType() != llfnty->getParamType(j)) {
                            Logger::println("llvmRunTimeHack==true - force casting argument");
                            Logger::cout() << "casting: " << *llargs[j] << " to type: " << *llfnty->getParamType(j) << '\n';
                            llargs[j] = DtoBitCast(llargs[j], llfnty->getParamType(j));
                        }
                    }
                }
            }
        }
    }

    #if 0
    Logger::println("%d params passed", n);
    for (int i=0; i<llargs.size(); ++i) {
        assert(llargs[i]);
        Logger::cout() << "arg["<<i<<"] = " << *llargs[i] << '\n';
    }
    #endif

    // void returns cannot not be named
    const char* varname = "";
    if (llfnty->getReturnType() != llvm::Type::VoidTy)
        varname = "tmp";

    //Logger::cout() << "Calling: " << *funcval << '\n';

    // call the function
    llvm::CallInst* call = llvm::CallInst::Create(funcval, llargs.begin(), llargs.end(), varname, p->scopebb());
    llvm::Value* retllval = (retinptr) ? llargs[0] : call;

    if (retinptr && dfn && dfn->func && dfn->func->runTimeHack) {
        const llvm::Type* rettype = getPtrToType(DtoType(type));
        if (retllval->getType() != rettype) {
            Logger::println("llvmRunTimeHack==true - force casting return value");
            Logger::cout() << "from: " << *retllval->getType() << " to: " << *rettype << '\n';
            retllval = DtoBitCast(retllval, rettype);
        }
    }

    // set calling convention
    if (dfn && dfn->func) {
        int li = dfn->func->llvmInternal;
        if (li != LLVMintrinsic && li != LLVMva_start && li != LLVMva_intrinsic) {
            call->setCallingConv(DtoCallingConv(dlink));
        }
    }
    /*else if (delegateCall) {
        call->setCallingConv(DtoCallingConv(dlink));
    }*/
    else if (dfn && dfn->cc != (unsigned)-1) {
        call->setCallingConv(dfn->cc);
    }
    else {
        call->setCallingConv(DtoCallingConv(dlink));
    }

    return new DImValue(type, retllval, isInPlace);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CastExp::toElem(IRState* p)
{
    Logger::print("CastExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);
    DValue* v = DtoCast(u, to);

    if (v->isSlice()) {
        assert(!gIR->topexp() || gIR->topexp()->e1 != this);
        return v;
    }

    else if (DLRValue* lr = u->isLRValue())
        return new DLRValue(lr->getLType(), lr->getLVal(), to, v->getRVal());

    else if (u->isVar() && u->isVar()->lval)
        return new DLRValue(e1->type, u->getLVal(), to, v->getRVal());

    else if (gIR->topexp() && gIR->topexp()->e1 == this)
        return new DLRValue(e1->type, u->getLVal(), to, v->getRVal());

    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* SymOffExp::toElem(IRState* p)
{
    Logger::print("SymOffExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(0 && "SymOffExp::toElem should no longer be called :/");

    if (VarDeclaration* vd = var->isVarDeclaration())
    {
        Logger::println("VarDeclaration");

        // handle forward reference
        if (!vd->ir.declared && vd->isDataseg()) {
            vd->toObjFile(); // TODO
        }

        assert(vd->ir.getIrValue());
        Type* t = DtoDType(type);
        Type* tnext = DtoDType(t->next);
        Type* vdtype = DtoDType(vd->type);

        llvm::Value* llvalue = vd->nestedref ? DtoNestedVariable(vd) : vd->ir.getIrValue();
        llvm::Value* varmem = 0;

        if (vdtype->ty == Tstruct && !(t->ty == Tpointer && t->next == vdtype)) {
            Logger::println("struct");
            TypeStruct* vdt = (TypeStruct*)vdtype;
            assert(vdt->sym);

            const llvm::Type* llt = DtoType(t);
            if (offset == 0) {
                varmem = p->ir->CreateBitCast(llvalue, llt, "tmp");
            }
            else {
                std::vector<unsigned> dst;
                varmem = DtoIndexStruct(llvalue,vdt->sym, tnext, offset, dst);
            }
        }
        else if (vdtype->ty == Tsarray) {
            Logger::println("sarray");

            assert(llvalue);
            //e->arg = llvalue; // TODO

            const llvm::Type* llt = DtoType(t);
            llvm::Value* off = 0;
            if (offset != 0) {
                Logger::println("offset = %d\n", offset);
            }
            if (offset == 0) {
                varmem = llvalue;
            }
            else {
                const llvm::Type* elemtype = llvalue->getType()->getContainedType(0)->getContainedType(0);
                size_t elemsz = getABITypeSize(elemtype);
                varmem = DtoGEPi(llvalue, 0, offset / elemsz, "tmp");
            }
        }
        else if (offset == 0) {
            Logger::println("normal symoff");

            assert(llvalue);
            varmem = llvalue;

            const llvm::Type* llt = DtoType(t);
            if (llvalue->getType() != llt) {
                varmem = p->ir->CreateBitCast(varmem, llt, "tmp");
            }
        }
        else {
            assert(0);
        }
        return new DFieldValue(type, varmem, true);
    }

    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AddrExp::toElem(IRState* p)
{
    Logger::println("AddrExp::toElem: %s | %s", toChars(), type->toChars());
    LOG_SCOPE;
    DValue* v = e1->toElem(p);
    if (v->isField()) {
        Logger::println("is field");
        return v;
    }
    else if (DFuncValue* fv = v->isFunc()) {
        Logger::println("is func");
        //Logger::println("FuncDeclaration");
        FuncDeclaration* fd = fv->func;
        assert(fd);
        DtoForceDeclareDsymbol(fd);
        return new DFuncValue(fd, fd->ir.irFunc->func);
    }
    else if (DImValue* im = v->isIm()) {
        Logger::println("is immediate");
        return v;
    }
    Logger::println("is nothing special");
    return new DFieldValue(type, v->getLVal(), false);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* PtrExp::toElem(IRState* p)
{
    Logger::println("PtrExp::toElem: %s | %s", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* a = e1->toElem(p);

    if (p->topexp() && p->topexp()->e1 == this) {
        Logger::println("lval PtrExp");
        return new DVarValue(type, a->getRVal(), true);
    }

    // this should be deterministic but right now lvalue casts don't propagate lvalueness !?!
    llvm::Value* lv = a->getRVal();
    llvm::Value* v = lv;
    if (DtoCanLoad(v))
        v = DtoLoad(v);
    return new DLRValue(e1->type, lv, type, v);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DotVarExp::toElem(IRState* p)
{
    Logger::print("DotVarExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    Type* t = DtoDType(type);
    Type* e1type = DtoDType(e1->type);

    //Logger::println("e1type=%s", e1type->toChars());
    //Logger::cout() << *DtoType(e1type) << '\n';

    if (VarDeclaration* vd = var->isVarDeclaration()) {
        llvm::Value* arrptr;
        if (e1type->ty == Tpointer) {
            assert(e1type->next->ty == Tstruct);
            TypeStruct* ts = (TypeStruct*)e1type->next;
            Logger::println("Struct member offset:%d", vd->offset);

            llvm::Value* src = l->getRVal();

            std::vector<unsigned> vdoffsets;
            arrptr = DtoIndexStruct(src, ts->sym, vd->type, vd->offset, vdoffsets);
        }
        else if (e1type->ty == Tclass) {
            TypeClass* tc = (TypeClass*)e1type;
            Logger::println("Class member offset: %d", vd->offset);

            llvm::Value* src = l->getRVal();

            std::vector<unsigned> vdoffsets;
            arrptr = DtoIndexClass(src, tc->sym, vd->type, vd->offset, vdoffsets);

            /*std::vector<unsigned> vdoffsets(1,0);
            tc->sym->offsetToIndex(vd->type, vd->offset, vdoffsets);

            llvm::Value* src = l->getRVal();

            Logger::println("indices:");
            for (size_t i=0; i<vdoffsets.size(); ++i)
                Logger::println("%d", vdoffsets[i]);

            Logger::cout() << "src: " << *src << '\n';
            arrptr = DtoGEP(src,vdoffsets,"tmp",p->scopebb());
            Logger::cout() << "dst: " << *arrptr << '\n';*/
        }
        else
            assert(0);

        //Logger::cout() << "mem: " << *arrptr << '\n';
        return new DVarValue(vd, arrptr, true);
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        DtoResolveDsymbol(fdecl);

        llvm::Value* funcval;
        llvm::Value* vthis2 = 0;
        if (e1type->ty == Tclass) {
            TypeClass* tc = (TypeClass*)e1type;
            if (tc->sym->isInterfaceDeclaration()) {
                vthis2 = DtoCastInterfaceToObject(l, NULL)->getRVal();
            }
        }
        llvm::Value* vthis = l->getRVal();
        if (!vthis2) vthis2 = vthis;
        //unsigned cc = (unsigned)-1;

        // super call
        if (e1->op == TOKsuper) {
            DtoForceDeclareDsymbol(fdecl);
            funcval = fdecl->ir.irFunc->func;
            assert(funcval);
        }
        // normal virtual call
        else if (fdecl->isAbstract() || (!fdecl->isFinal() && fdecl->isVirtual())) {
            assert(fdecl->vtblIndex > 0);
            assert(e1type->ty == Tclass);

            llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            llvm::Value* vtblidx = llvm::ConstantInt::get(llvm::Type::Int32Ty, (size_t)fdecl->vtblIndex, false);
            //Logger::cout() << "vthis: " << *vthis << '\n';
            funcval = DtoGEP(vthis, zero, zero, "tmp", p->scopebb());
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
            funcval = DtoGEP(funcval, zero, vtblidx, toChars(), p->scopebb());
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
        #if OPAQUE_VTBLS
            funcval = DtoBitCast(funcval, getPtrToType(DtoType(fdecl->type)));
            Logger::cout() << "funcval casted: " << *funcval << '\n';
        #endif
            //assert(funcval->getType() == DtoType(fdecl->type));
            //cc = DtoCallingConv(fdecl->linkage);
        }
        // static call
        else {
            DtoForceDeclareDsymbol(fdecl);
            funcval = fdecl->ir.irFunc->func;
            assert(funcval);
            //assert(funcval->getType() == DtoType(fdecl->type));
        }
        return new DFuncValue(fdecl, funcval, vthis2);
    }
    else {
        printf("unsupported dotvarexp: %s\n", var->toChars());
    }

    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ThisExp::toElem(IRState* p)
{
    Logger::print("ThisExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    if (VarDeclaration* vd = var->isVarDeclaration()) {
        llvm::Value* v;
        v = p->func()->decl->ir.irFunc->thisVar;
        if (llvm::isa<llvm::AllocaInst>(v))
            v = new llvm::LoadInst(v, "tmp", p->scopebb());
        const llvm::Type* t = DtoType(type);
        if (v->getType() != t)
            v = DtoBitCast(v, t, "tmp");
        return new DThisValue(vd, v);
    }

    assert(0);
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* IndexExp::toElem(IRState* p)
{
    Logger::print("IndexExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    Type* e1type = DtoDType(e1->type);

    p->arrays.push_back(l); // if $ is used it must be an array so this is fine.
    DValue* r = e2->toElem(p);
    p->arrays.pop_back();

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    llvm::Value* arrptr = 0;
    if (e1type->ty == Tpointer) {
        arrptr = llvm::GetElementPtrInst::Create(l->getRVal(),r->getRVal(),"tmp",p->scopebb());
    }
    else if (e1type->ty == Tsarray) {
        arrptr = DtoGEP(l->getRVal(), zero, r->getRVal(),"tmp",p->scopebb());
    }
    else if (e1type->ty == Tarray) {
        arrptr = DtoGEP(l->getRVal(),zero,one,"tmp",p->scopebb());
        arrptr = new llvm::LoadInst(arrptr,"tmp",p->scopebb());
        arrptr = llvm::GetElementPtrInst::Create(arrptr,r->getRVal(),"tmp",p->scopebb());
    }
    else if (e1type->ty == Taarray) {
        return DtoAAIndex(type, l, r);
    }
    else {
        Logger::println("invalid index exp! e1type: %s", e1type->toChars());
        assert(0);
    }
    return new DVarValue(type, arrptr, true);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* SliceExp::toElem(IRState* p)
{
    Logger::print("SliceExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* t = DtoDType(type);
    Type* e1type = DtoDType(e1->type);

    DValue* v = e1->toElem(p);
    llvm::Value* vmem = v->getRVal();
    assert(vmem);

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    llvm::Value* emem = 0;
    llvm::Value* earg = 0;

    // partial slice
    if (lwr)
    {
        assert(upr);
        p->arrays.push_back(v);
        DValue* lo = lwr->toElem(p);

        bool lwr_is_zero = false;
        if (DConstValue* cv = lo->isConst())
        {
            assert(llvm::isa<llvm::ConstantInt>(cv->c));

            if (e1type->ty == Tpointer) {
                emem = v->getRVal();
            }
            else if (e1type->ty == Tarray) {
                llvm::Value* tmp = DtoGEP(vmem,zero,one,"tmp",p->scopebb());
                emem = new llvm::LoadInst(tmp,"tmp",p->scopebb());
            }
            else if (e1type->ty == Tsarray) {
                emem = DtoGEP(vmem,zero,zero,"tmp",p->scopebb());
            }
            else
            assert(emem);

            llvm::ConstantInt* c = llvm::cast<llvm::ConstantInt>(cv->c);
            if (!(lwr_is_zero = c->isZero())) {
                emem = llvm::GetElementPtrInst::Create(emem,cv->c,"tmp",p->scopebb());
            }
        }
        else
        {
            if (e1type->ty == Tarray) {
                llvm::Value* tmp = DtoGEP(vmem,zero,one,"tmp",p->scopebb());
                tmp = new llvm::LoadInst(tmp,"tmp",p->scopebb());
                emem = llvm::GetElementPtrInst::Create(tmp,lo->getRVal(),"tmp",p->scopebb());
            }
            else if (e1type->ty == Tsarray) {
                emem = DtoGEP(vmem,zero,lo->getRVal(),"tmp",p->scopebb());
            }
            else if (e1type->ty == Tpointer) {
                emem = llvm::GetElementPtrInst::Create(v->getRVal(),lo->getRVal(),"tmp",p->scopebb());
            }
            else {
                Logger::println("type = %s", e1type->toChars());
                assert(0);
            }
        }

        DValue* up = upr->toElem(p);
        p->arrays.pop_back();

        if (DConstValue* cv = up->isConst())
        {
            assert(llvm::isa<llvm::ConstantInt>(cv->c));
            if (lwr_is_zero) {
                earg = cv->c;
            }
            else {
                if (lo->isConst()) {
                    llvm::Constant* clo = llvm::cast<llvm::Constant>(lo->getRVal());
                    llvm::Constant* cup = llvm::cast<llvm::Constant>(cv->c);
                    earg = llvm::ConstantExpr::getSub(cup, clo);
                }
                else {
                    earg = llvm::BinaryOperator::createSub(cv->c, lo->getRVal(), "tmp", p->scopebb());
                }
            }
        }
        else
        {
            if (lwr_is_zero) {
                earg = up->getRVal();
            }
            else {
                earg = llvm::BinaryOperator::createSub(up->getRVal(), lo->getRVal(), "tmp", p->scopebb());
            }
        }
    }
    // full slice
    else
    {
        emem = vmem;
    }

    if (earg) Logger::cout() << "slice exp result, length = " << *earg << '\n';
    Logger::cout() << "slice exp result, ptr = " << *emem << '\n';

    return new DSliceValue(type,earg,emem);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CmpExp::toElem(IRState* p)
{
    Logger::print("CmpExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = DtoDType(e1->type);
    Type* e2t = DtoDType(e2->type);
    assert(DtoType(t) == DtoType(e2t));

    llvm::Value* eval = 0;

    if (t->isintegral() || t->ty == Tpointer)
    {
        llvm::ICmpInst::Predicate cmpop;
        bool skip = false;
        switch(op)
        {
        case TOKlt:
        case TOKul:
            cmpop = t->isunsigned() ? llvm::ICmpInst::ICMP_ULT : llvm::ICmpInst::ICMP_SLT;
            break;
        case TOKle:
        case TOKule:
            cmpop = t->isunsigned() ? llvm::ICmpInst::ICMP_ULE : llvm::ICmpInst::ICMP_SLE;
            break;
        case TOKgt:
        case TOKug:
            cmpop = t->isunsigned() ? llvm::ICmpInst::ICMP_UGT : llvm::ICmpInst::ICMP_SGT;
            break;
        case TOKge:
        case TOKuge:
            cmpop = t->isunsigned() ? llvm::ICmpInst::ICMP_UGE : llvm::ICmpInst::ICMP_SGE;
            break;
        case TOKue:
            cmpop = llvm::ICmpInst::ICMP_EQ;
            break;
        case TOKlg:
            cmpop = llvm::ICmpInst::ICMP_NE;
            break;
        case TOKleg:
            skip = true;
            eval = llvm::ConstantInt::getTrue();
            break;
        case TOKunord:
            skip = true;
            eval = llvm::ConstantInt::getFalse();
            break;

        default:
            assert(0);
        }
        if (!skip)
        {
            llvm::Value* a = l->getRVal();
            llvm::Value* b = r->getRVal();
            Logger::cout() << "type 1: " << *a << '\n';
            Logger::cout() << "type 2: " << *b << '\n';
            eval = new llvm::ICmpInst(cmpop, a, b, "tmp", p->scopebb());
        }
    }
    else if (t->isfloating())
    {
        llvm::FCmpInst::Predicate cmpop;
        switch(op)
        {
        case TOKlt:
            cmpop = llvm::FCmpInst::FCMP_OLT;break;
        case TOKle:
            cmpop = llvm::FCmpInst::FCMP_OLE;break;
        case TOKgt:
            cmpop = llvm::FCmpInst::FCMP_OGT;break;
        case TOKge:
            cmpop = llvm::FCmpInst::FCMP_OGE;break;
        case TOKunord:
            cmpop = llvm::FCmpInst::FCMP_UNO;break;
        case TOKule:
            cmpop = llvm::FCmpInst::FCMP_ULE;break;
        case TOKul:
            cmpop = llvm::FCmpInst::FCMP_ULT;break;
        case TOKuge:
            cmpop = llvm::FCmpInst::FCMP_UGE;break;
        case TOKug:
            cmpop = llvm::FCmpInst::FCMP_UGT;break;
        case TOKue:
            cmpop = llvm::FCmpInst::FCMP_UEQ;break;
        case TOKlg:
            cmpop = llvm::FCmpInst::FCMP_ONE;break;
        case TOKleg:
            cmpop = llvm::FCmpInst::FCMP_ORD;break;

        default:
            assert(0);
        }
        eval = new llvm::FCmpInst(cmpop, l->getRVal(), r->getRVal(), "tmp", p->scopebb());
    }
    else if (t->ty == Tsarray || t->ty == Tarray)
    {
        Logger::println("static or dynamic array");
        eval = DtoArrayCompare(op,l,r);
    }
    else
    {
        assert(0 && "Unsupported CmpExp type");
    }

    return new DImValue(type, eval);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* EqualExp::toElem(IRState* p)
{
    Logger::print("EqualExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    Type* t = DtoDType(e1->type);
    Type* e2t = DtoDType(e2->type);
    //assert(t == e2t);

    llvm::Value* eval = 0;

    if (t->isintegral() || t->ty == Tpointer)
    {
        Logger::println("integral or pointer");
        llvm::ICmpInst::Predicate cmpop;
        switch(op)
        {
        case TOKequal:
            cmpop = llvm::ICmpInst::ICMP_EQ;
            break;
        case TOKnotequal:
            cmpop = llvm::ICmpInst::ICMP_NE;
            break;
        default:
            assert(0);
        }
        llvm::Value* lv = l->getRVal();
        llvm::Value* rv = r->getRVal();
        if (rv->getType() != lv->getType()) {
            rv = DtoBitCast(rv, lv->getType());
        }
        eval = new llvm::ICmpInst(cmpop, lv, rv, "tmp", p->scopebb());
    }
    else if (t->iscomplex())
    {
        Logger::println("complex");
        eval = DtoComplexEquals(op, l, r);
    }
    else if (t->isfloating())
    {
        Logger::println("floating");
        llvm::FCmpInst::Predicate cmpop;
        switch(op)
        {
        case TOKequal:
            cmpop = llvm::FCmpInst::FCMP_OEQ;
            break;
        case TOKnotequal:
            cmpop = llvm::FCmpInst::FCMP_UNE;
            break;
        default:
            assert(0);
        }
        eval = new llvm::FCmpInst(cmpop, l->getRVal(), r->getRVal(), "tmp", p->scopebb());
    }
    else if (t->ty == Tsarray || t->ty == Tarray)
    {
        Logger::println("static or dynamic array");
        eval = DtoArrayEquals(op,l,r);
    }
    else if (t->ty == Tdelegate)
    {
        Logger::println("delegate");
        eval = DtoDelegateCompare(op,l->getRVal(),r->getRVal());
    }
    else
    {
        assert(0 && "Unsupported EqualExp type");
    }

    return new DImValue(type, eval);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* PostExp::toElem(IRState* p)
{
    Logger::print("PostExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);
    DValue* r = e2->toElem(p);

    llvm::Value* val = l->getRVal();
    llvm::Value* post = 0;

    Type* e1type = DtoDType(e1->type);
    Type* e2type = DtoDType(e2->type);

    if (e1type->isintegral())
    {
        assert(e2type->isintegral());
        llvm::Value* one = llvm::ConstantInt::get(val->getType(), 1, !e2type->isunsigned());
        if (op == TOKplusplus) {
            post = llvm::BinaryOperator::createAdd(val,one,"tmp",p->scopebb());
        }
        else if (op == TOKminusminus) {
            post = llvm::BinaryOperator::createSub(val,one,"tmp",p->scopebb());
        }
    }
    else if (e1type->ty == Tpointer)
    {
        assert(e2type->isintegral());
        llvm::Constant* minusone = llvm::ConstantInt::get(DtoSize_t(),(uint64_t)-1,true);
        llvm::Constant* plusone = llvm::ConstantInt::get(DtoSize_t(),(uint64_t)1,false);
        llvm::Constant* whichone = (op == TOKplusplus) ? plusone : minusone;
        post = llvm::GetElementPtrInst::Create(val, whichone, "tmp", p->scopebb());
    }
    else if (e1type->isfloating())
    {
        assert(e2type->isfloating());
        llvm::Value* one = DtoConstFP(e1type, 1.0);
        if (op == TOKplusplus) {
            post = llvm::BinaryOperator::createAdd(val,one,"tmp",p->scopebb());
        }
        else if (op == TOKminusminus) {
            post = llvm::BinaryOperator::createSub(val,one,"tmp",p->scopebb());
        }
    }
    else
    assert(post);

    DtoStore(post,l->getLVal());

    return new DImValue(type,val,true);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* NewExp::toElem(IRState* p)
{
    Logger::print("NewExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(!newargs && "arguments to new not yet supported");
    assert(newtype);
    assert(!allocator && "custom allocators not yet supported");

    Type* ntype = DtoDType(newtype);

    // new class
    if (ntype->ty == Tclass) {
        Logger::println("new class");
        return DtoNewClass((TypeClass*)ntype, this);
    }
    // new dynamic array
    else if (ntype->ty == Tarray)
    {
        Logger::println("new dynamic array: %s", newtype->toChars());
        // get dim
        assert(arguments);
        assert(arguments->dim == 1);
        DValue* sz = ((Expression*)arguments->data[0])->toElem(p);
        // allocate & init
        return DtoNewDynArray(newtype, sz, true);
    }
    // new static array
    else if (ntype->ty == Tsarray)
    {
        assert(0);
    }
    // new struct
    else if (ntype->ty == Tstruct)
    {
        // allocate
        llvm::Value* mem = DtoNew(newtype);
        // init
        TypeStruct* ts = (TypeStruct*)ntype;
        if (ts->isZeroInit()) {
            DtoStructZeroInit(mem);
        }
        else {
            assert(ts->sym);
            DtoStructCopy(mem,ts->sym->ir.irStruct->init);
        }
        return new DImValue(type, mem, false);
    }
    // new basic type
    else
    {
        // allocate
        llvm::Value* mem = DtoNew(newtype);
        // BUG: default initialize
        // return
        return new DImValue(type, mem, false);
    }

    assert(0);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DeleteExp::toElem(IRState* p)
{
    Logger::print("DeleteExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* dval = e1->toElem(p);
    Type* et = DtoDType(e1->type);

    // simple pointer
    if (et->ty == Tpointer)
    {
        llvm::Value* rval = dval->getRVal();
        DtoDeleteMemory(rval);
        if (dval->isVar() && dval->isVar()->lval)
            DtoStore(llvm::Constant::getNullValue(rval->getType()), dval->getLVal());
    }
    // class
    else if (et->ty == Tclass)
    {
        bool onstack = false;
        TypeClass* tc = (TypeClass*)et;
        if (tc->sym->isInterfaceDeclaration())
        {
            DtoDeleteInterface(dval->getRVal());
            onstack = true;
        }
        else if (DVarValue* vv = dval->isVar()) {
            if (vv->var && vv->var->onstack) {
                if (tc->sym->dtors.dim > 0) {
                    DtoFinalizeClass(dval->getRVal());
                    onstack = true;
                }
            }
        }
        if (!onstack) {
            llvm::Value* rval = dval->getRVal();
            DtoDeleteClass(rval);
        }
        if (!dval->isThis() && dval->isVar() && dval->isVar()->lval) {
            llvm::Value* lval = dval->getLVal();
            DtoStore(llvm::Constant::getNullValue(lval->getType()->getContainedType(0)), lval);
        }
    }
    // dyn array
    else if (et->ty == Tarray)
    {
        DtoDeleteArray(dval);
        if (!dval->isSlice())
            DtoSetArrayToNull(dval->getRVal());
    }
    // unknown/invalid
    else
    {
        assert(0 && "invalid delete");
    }

    // no value to return
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ArrayLengthExp::toElem(IRState* p)
{
    Logger::print("ArrayLengthExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);
    Logger::println("e1 = %s", e1->type->toChars());

    if (p->topexp() && p->topexp()->e1 == this)
    {
        return new DArrayLenValue(e1->type, u->getLVal());
    }
    else
    {
        return new DImValue(type, DtoArrayLen(u));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AssertExp::toElem(IRState* p)
{
    Logger::print("AssertExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // condition
    DValue* cond = e1->toElem(p);

    // create basic blocks
    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* assertbb = llvm::BasicBlock::Create("assert", p->topfunc(), oldend);
    llvm::BasicBlock* endbb = llvm::BasicBlock::Create("endassert", p->topfunc(), oldend);

    // test condition
    llvm::Value* condval = cond->getRVal();
    condval = DtoBoolean(condval);

    // branch
    llvm::BranchInst::Create(endbb, assertbb, condval, p->scopebb());

    // call assert runtime functions
    p->scope() = IRScope(assertbb,endbb);
    DtoAssert(&loc, msg ? msg->toElem(p) : NULL);

    if (!gIR->scopereturned())
        llvm::BranchInst::Create(endbb, p->scopebb());

    // rewrite the scope
    p->scope() = IRScope(endbb,oldend);

    // no meaningful return value
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* NotExp::toElem(IRState* p)
{
    Logger::print("NotExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);

    llvm::Value* b = DtoBoolean(u->getRVal());

    llvm::Constant* zero = llvm::ConstantInt::get(llvm::Type::Int1Ty, 0, true);
    b = p->ir->CreateICmpEQ(b,zero);

    return new DImValue(type, b);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AndAndExp::toElem(IRState* p)
{
    Logger::print("AndAndExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // allocate a temporary for the final result. failed to come up with a better way :/
    llvm::Value* resval = 0;
    llvm::BasicBlock* entryblock = &p->topfunc()->front();
    resval = new llvm::AllocaInst(llvm::Type::Int1Ty,"andandtmp",p->topallocapoint());

    DValue* u = e1->toElem(p);

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* andand = llvm::BasicBlock::Create("andand", gIR->topfunc(), oldend);
    llvm::BasicBlock* andandend = llvm::BasicBlock::Create("andandend", gIR->topfunc(), oldend);

    llvm::Value* ubool = DtoBoolean(u->getRVal());
    new llvm::StoreInst(ubool,resval,p->scopebb());
    llvm::BranchInst::Create(andand,andandend,ubool,p->scopebb());

    p->scope() = IRScope(andand, andandend);
    DValue* v = e2->toElem(p);

    llvm::Value* vbool = DtoBoolean(v->getRVal());
    llvm::Value* uandvbool = llvm::BinaryOperator::create(llvm::BinaryOperator::And, ubool, vbool,"tmp",p->scopebb());
    new llvm::StoreInst(uandvbool,resval,p->scopebb());
    llvm::BranchInst::Create(andandend,p->scopebb());

    p->scope() = IRScope(andandend, oldend);

    resval = new llvm::LoadInst(resval,"tmp",p->scopebb());
    return new DImValue(type, resval);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* OrOrExp::toElem(IRState* p)
{
    Logger::print("OrOrExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // allocate a temporary for the final result. failed to come up with a better way :/
    llvm::Value* resval = 0;
    llvm::BasicBlock* entryblock = &p->topfunc()->front();
    resval = new llvm::AllocaInst(llvm::Type::Int1Ty,"orortmp",p->topallocapoint());

    DValue* u = e1->toElem(p);

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* oror = llvm::BasicBlock::Create("oror", gIR->topfunc(), oldend);
    llvm::BasicBlock* ororend = llvm::BasicBlock::Create("ororend", gIR->topfunc(), oldend);

    llvm::Value* ubool = DtoBoolean(u->getRVal());
    new llvm::StoreInst(ubool,resval,p->scopebb());
    llvm::BranchInst::Create(ororend,oror,ubool,p->scopebb());

    p->scope() = IRScope(oror, ororend);
    DValue* v = e2->toElem(p);

    llvm::Value* vbool = DtoBoolean(v->getRVal());
    new llvm::StoreInst(vbool,resval,p->scopebb());
    llvm::BranchInst::Create(ororend,p->scopebb());

    p->scope() = IRScope(ororend, oldend);

    resval = new llvm::LoadInst(resval,"tmp",p->scopebb());
    return new DImValue(type, resval);
}

//////////////////////////////////////////////////////////////////////////////////////////

#define BinBitExp(X,Y) \
DValue* X##Exp::toElem(IRState* p) \
{ \
    Logger::print("%sExp::toElem: %s | %s\n", #X, toChars(), type->toChars()); \
    LOG_SCOPE; \
    DValue* u = e1->toElem(p); \
    DValue* v = e2->toElem(p); \
    llvm::Value* x = llvm::BinaryOperator::create(llvm::Instruction::Y, u->getRVal(), v->getRVal(), "tmp", p->scopebb()); \
    return new DImValue(type, x); \
} \
\
DValue* X##AssignExp::toElem(IRState* p) \
{ \
    Logger::print("%sAssignExp::toElem: %s | %s\n", #X, toChars(), type->toChars()); \
    LOG_SCOPE; \
    p->exps.push_back(IRExp(e1,e2,NULL)); \
    DValue* u = e1->toElem(p); \
    p->topexp()->v = u; \
    DValue* v = e2->toElem(p); \
    p->exps.pop_back(); \
    llvm::Value* uval = u->getRVal(); \
    llvm::Value* vval = v->getRVal(); \
    llvm::Value* tmp = llvm::BinaryOperator::create(llvm::Instruction::Y, uval, vval, "tmp", p->scopebb()); \
    new llvm::StoreInst(DtoPointedType(u->getLVal(), tmp), u->getLVal(), p->scopebb()); \
    return u; \
}

BinBitExp(And,And);
BinBitExp(Or,Or);
BinBitExp(Xor,Xor);
BinBitExp(Shl,Shl);
BinBitExp(Shr,AShr);
BinBitExp(Ushr,LShr);

//////////////////////////////////////////////////////////////////////////////////////////

DValue* HaltExp::toElem(IRState* p)
{
    Logger::print("HaltExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DtoAssert(&loc, NULL);

    new llvm::UnreachableInst(p->scopebb());
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DelegateExp::toElem(IRState* p)
{
    Logger::print("DelegateExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    const llvm::PointerType* int8ptrty = getPtrToType(llvm::Type::Int8Ty);

    llvm::Value* lval;
    bool inplace = false;
    if (p->topexp() && p->topexp()->e2 == this) {
        assert(p->topexp()->v);
        lval = p->topexp()->v->getLVal();
        inplace = true;
    }
    else {
        lval = new llvm::AllocaInst(DtoType(type), "tmpdelegate", p->topallocapoint());
    }

    DValue* u = e1->toElem(p);
    llvm::Value* uval;
    if (DFuncValue* f = u->isFunc()) {
        //assert(f->vthis);
        //uval = f->vthis;
        llvm::Value* nestvar = p->func()->decl->ir.irFunc->nestedVar;
        if (nestvar)
            uval = nestvar;
        else
            uval = llvm::ConstantPointerNull::get(int8ptrty);
    }
    else {
        DValue* src = u;
        if (ClassDeclaration* cd = u->getType()->isClassHandle())
        {
            Logger::println("context type is class handle");
            if (cd->isInterfaceDeclaration())
            {
                Logger::println("context type is interface");
                src = DtoCastInterfaceToObject(u, ClassDeclaration::object->type);
            }
        }
        uval = src->getRVal();
    }

    Logger::cout() << "context = " << *uval << '\n';

    llvm::Value* context = DtoGEPi(lval,0,0,"tmp");
    llvm::Value* castcontext = DtoBitCast(uval, int8ptrty);
    DtoStore(castcontext, context);

    llvm::Value* fptr = DtoGEPi(lval,0,1,"tmp");

    Logger::println("func: '%s'", func->toPrettyChars());

    llvm::Value* castfptr;
    if (func->isVirtual())
        castfptr = DtoVirtualFunctionPointer(u, func);
    else if (func->isAbstract())
        assert(0 && "TODO delegate to abstract method");
    else if (func->toParent()->isInterfaceDeclaration())
        assert(0 && "TODO delegate to interface method");
    else
    {
        DtoForceDeclareDsymbol(func);
        castfptr = func->ir.irFunc->func;
    }

    castfptr = DtoBitCast(castfptr, fptr->getType()->getContainedType(0));
    DtoStore(castfptr, fptr);

    return new DImValue(type, lval, inplace);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* IdentityExp::toElem(IRState* p)
{
    Logger::print("IdentityExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);
    DValue* v = e2->toElem(p);

    llvm::Value* l = u->getRVal();
    llvm::Value* r = v->getRVal();

    Type* t1 = DtoDType(e1->type);

    llvm::Value* eval = 0;

    if (t1->ty == Tarray) {
        if (v->isNull()) {
            r = NULL;
        }
        else {
            assert(l->getType() == r->getType());
        }
        eval = DtoDynArrayIs(op,l,r);
    }
    else if (t1->ty == Tdelegate) {
        if (v->isNull()) {
            r = NULL;
        }
        else {
            assert(l->getType() == r->getType());
        }
        eval = DtoDelegateCompare(op,l,r);
    }
    else if (t1->isfloating())
    {
        llvm::FCmpInst::Predicate pred = (op == TOKidentity) ? llvm::FCmpInst::FCMP_OEQ : llvm::FCmpInst::FCMP_ONE;
        eval = new llvm::FCmpInst(pred, l, r, "tmp", p->scopebb());
    }
    else if (t1->ty == Tpointer)
    {
        if (l->getType() != r->getType()) {
            if (v->isNull())
                r = llvm::ConstantPointerNull::get(isaPointer(l->getType()));
            else
                r = DtoBitCast(r, l->getType(), "tmp");
        }
        llvm::ICmpInst::Predicate pred = (op == TOKidentity) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
        eval = new llvm::ICmpInst(pred, l, r, "tmp", p->scopebb());
    }
    else {
        llvm::ICmpInst::Predicate pred = (op == TOKidentity) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
        //Logger::cout() << "l = " << *l << " r = " << *r << '\n';
        eval = new llvm::ICmpInst(pred, l, r, "tmp", p->scopebb());
    }
    return new DImValue(type, eval);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CommaExp::toElem(IRState* p)
{
    Logger::print("CommaExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);
    DValue* v = e2->toElem(p);
    assert(e2->type == type);
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CondExp::toElem(IRState* p)
{
    Logger::print("CondExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* dtype = DtoDType(type);
    const llvm::Type* resty = DtoType(dtype);

    // allocate a temporary for the final result. failed to come up with a better way :/
    llvm::BasicBlock* entryblock = &p->topfunc()->front();
    llvm::Value* resval = new llvm::AllocaInst(resty,"condtmp",p->topallocapoint());
    DVarValue* dvv = new DVarValue(type, resval, true);

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* condtrue = llvm::BasicBlock::Create("condtrue", gIR->topfunc(), oldend);
    llvm::BasicBlock* condfalse = llvm::BasicBlock::Create("condfalse", gIR->topfunc(), oldend);
    llvm::BasicBlock* condend = llvm::BasicBlock::Create("condend", gIR->topfunc(), oldend);

    DValue* c = econd->toElem(p);
    llvm::Value* cond_val = DtoBoolean(c->getRVal());
    llvm::BranchInst::Create(condtrue,condfalse,cond_val,p->scopebb());

    p->scope() = IRScope(condtrue, condfalse);
    DValue* u = e1->toElem(p);
    DtoAssign(dvv, u);
    llvm::BranchInst::Create(condend,p->scopebb());

    p->scope() = IRScope(condfalse, condend);
    DValue* v = e2->toElem(p);
    DtoAssign(dvv, v);
    llvm::BranchInst::Create(condend,p->scopebb());

    p->scope() = IRScope(condend, oldend);
    return dvv;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ComExp::toElem(IRState* p)
{
    Logger::print("ComExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* u = e1->toElem(p);

    llvm::Value* value = u->getRVal();
    llvm::Value* minusone = llvm::ConstantInt::get(value->getType(), -1, true);
    value = llvm::BinaryOperator::create(llvm::Instruction::Xor, value, minusone, "tmp", p->scopebb());

    return new DImValue(type, value);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* NegExp::toElem(IRState* p)
{
    Logger::print("NegExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    if (type->iscomplex()) {
        return DtoComplexNeg(type, l);
    }

    llvm::Value* val = l->getRVal();
    Type* t = DtoDType(type);

    llvm::Value* zero = 0;
    if (t->isintegral())
        zero = llvm::ConstantInt::get(val->getType(), 0, true);
    else if (t->isfloating()) {
        zero = DtoConstFP(type, 0.0);
    }
    else
        assert(0);

    val = llvm::BinaryOperator::createSub(zero,val,"tmp",p->scopebb());
    return new DImValue(type, val);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CatExp::toElem(IRState* p)
{
    Logger::print("CatExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* t = DtoDType(type);

    bool arrNarr = DtoDType(e1->type) == DtoDType(e2->type);

    // array ~ array
    if (arrNarr)
    {
        return DtoCatArrays(type, e1, e2);
    }
    // array ~ element
    // element ~ array
    else
    {
        return DtoCatArrayElement(type, e1, e2);
    }

    /*
    IRExp* ex = p->topexp();
    if (ex && ex->e2 == this) {
        assert(ex->v);
        if (arrNarr)
            DtoCatArrays(ex->v->getLVal(),e1,e2);
        else
            DtoCatArrayElement(ex->v->getLVal(),e1,e2);
        return new DImValue(type, ex->v->getLVal(), true);
    }
    else {
        assert(t->ty == Tarray);
        const llvm::Type* arrty = DtoType(t);
        llvm::Value* dst = new llvm::AllocaInst(arrty, "tmpmem", p->topallocapoint());
        if (arrNarr)
            DtoCatAr
            DtoCatArrays(dst,e1,e2);
        else
            DtoCatArrayElement(dst,e1,e2);
        return new DVarValue(type, dst, true);
    }
    */
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* CatAssignExp::toElem(IRState* p)
{
    Logger::print("CatAssignExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* l = e1->toElem(p);

    Type* e1type = DtoDType(e1->type);
    Type* elemtype = DtoDType(e1type->next);
    Type* e2type = DtoDType(e2->type);

    if (e2type == elemtype) {
        DSliceValue* slice = DtoCatAssignElement(l,e2);
        DtoAssign(l, slice);
    }
    else if (e1type == e2type) {
        DSliceValue* slice = DtoCatAssignArray(l,e2);
        DtoAssign(l, slice);
    }
    else
        assert(0 && "only one element at a time right now");

    return l;
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* FuncExp::toElem(IRState* p)
{
    Logger::print("FuncExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(fd);

    if (fd->isNested()) Logger::println("nested");
    Logger::println("kind = %s\n", fd->kind());

    DtoForceDefineDsymbol(fd);

    bool temp = false;
    llvm::Value* lval = NULL;
    if (p->topexp() && p->topexp()->e2 == this) {
        assert(p->topexp()->v);
        lval = p->topexp()->v->getLVal();
    }
    else {
        const llvm::Type* dgty = DtoType(type);
        Logger::cout() << "delegate without explicit storage:" << '\n' << *dgty << '\n';
        lval = new llvm::AllocaInst(dgty,"dgstorage",p->topallocapoint());
        temp = true;
    }

    llvm::Value* context = DtoGEPi(lval,0,0,"tmp",p->scopebb());
    const llvm::PointerType* pty = isaPointer(context->getType()->getContainedType(0));
    llvm::Value* llvmNested = p->func()->decl->ir.irFunc->nestedVar;
    if (llvmNested == NULL) {
        llvm::Value* nullcontext = llvm::ConstantPointerNull::get(pty);
        p->ir->CreateStore(nullcontext, context);
    }
    else {
        llvm::Value* nestedcontext = p->ir->CreateBitCast(llvmNested, pty, "tmp");
        p->ir->CreateStore(nestedcontext, context);
    }

    llvm::Value* fptr = DtoGEPi(lval,0,1,"tmp",p->scopebb());

    assert(fd->ir.irFunc->func);
    llvm::Value* castfptr = new llvm::BitCastInst(fd->ir.irFunc->func,fptr->getType()->getContainedType(0),"tmp",p->scopebb());
    new llvm::StoreInst(castfptr, fptr, p->scopebb());

    if (temp)
        return new DVarValue(type, lval, true);
    else
        return new DImValue(type, lval, true);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* ArrayLiteralExp::toElem(IRState* p)
{
    Logger::print("ArrayLiteralExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // D types
    Type* arrayType = type->toBasetype();
    Type* elemType = arrayType->nextOf()->toBasetype();

    // is dynamic ?
    bool dyn = (arrayType->ty == Tarray);
    // length
    size_t len = elements->dim;
    // store into slice?
    bool sliceInPlace = false;

    // llvm target type
    const llvm::Type* llType = DtoType(arrayType);
    Logger::cout() << (dyn?"dynamic":"static") << " array literal with length " << len << " of D type: '" << arrayType->toChars() << "' has llvm type: '" << *llType << "'\n";

    // llvm storage type
    const llvm::Type* llStoType = llvm::ArrayType::get(DtoType(elemType), len);
    Logger::cout() << "llvm storage type: '" << *llStoType << "'\n";

    // dst pointer
    llvm::Value* dstMem = 0;

    // rvalue of assignment
    if (p->topexp() && p->topexp()->e2 == this)
    {
        DValue* topval = p->topexp()->v;
        // slice assignment (copy)
        if (DSliceValue* s = topval->isSlice())
        {
            dstMem = s->ptr;
            sliceInPlace = true;
            assert(s->len == NULL);
        }
        // static array assignment
        else if (topval->getType()->toBasetype()->ty == Tsarray)
        {
            dstMem = topval->getLVal();
        }
        // otherwise we still need to alloca storage
    }

    // alloca storage if not found already
    if (!dstMem)
    {
        dstMem = new llvm::AllocaInst(llStoType, "arrayliteral", p->topallocapoint());
    }
    Logger::cout() << "using dest mem: " << *dstMem << '\n';

    // store elements
    for (size_t i=0; i<len; ++i)
    {
        Expression* expr = (Expression*)elements->data[i];
        llvm::Value* elemAddr = DtoGEPi(dstMem,0,i,"tmp",p->scopebb());

        // emulate assignment
        DVarValue* vv = new DVarValue(expr->type, elemAddr, true);
        p->exps.push_back(IRExp(NULL, expr, vv));
        DValue* e = expr->toElem(p);
        p->exps.pop_back();
        DImValue* im = e->isIm();
        if (!im || !im->inPlace()) {
            DtoAssign(vv, e);
        }
    }

    // return storage directly ?
    if (!dyn || (dyn && sliceInPlace))
        return new DImValue(type, dstMem, true);
    // wrap in a slice
    return new DSliceValue(type, DtoConstSize_t(len), DtoGEPi(dstMem,0,0,"tmp"));
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* ArrayLiteralExp::toConstElem(IRState* p)
{
    Logger::print("ArrayLiteralExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    const llvm::Type* t = DtoType(type);
    Logger::cout() << "array literal has llvm type: " << *t << '\n';
    assert(isaArray(t));
    const llvm::ArrayType* arrtype = isaArray(t);

    assert(arrtype->getNumElements() == elements->dim);
    std::vector<llvm::Constant*> vals(elements->dim, NULL);
    for (unsigned i=0; i<elements->dim; ++i)
    {
        Expression* expr = (Expression*)elements->data[i];
        vals[i] = expr->toConstElem(p);
    }

    return llvm::ConstantArray::get(arrtype, vals);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* StructLiteralExp::toElem(IRState* p)
{
    Logger::print("StructLiteralExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    llvm::Value* sptr;
    const llvm::Type* llt = DtoType(type);

    llvm::Value* mem = 0;

    // temporary struct literal
    if (!p->topexp() || p->topexp()->e2 != this)
    {
        sptr = new llvm::AllocaInst(llt,"tmpstructliteral",p->topallocapoint());
    }
    // already has memory
    else
    {
        assert(p->topexp()->e2 == this);
        sptr = p->topexp()->v->getLVal();
    }

    // num elements in literal
    unsigned n = elements->dim;

    // unions might have different types for each literal
    if (sd->ir.irStruct->hasUnions) {
        // build the type of the literal
        std::vector<const llvm::Type*> tys;
        for (unsigned i=0; i<n; ++i) {
            Expression* vx = (Expression*)elements->data[i];
            if (!vx) continue;
            tys.push_back(DtoType(vx->type));
        }
        const llvm::StructType* t = llvm::StructType::get(tys);
        if (t != llt) {
            if (getABITypeSize(t) != getABITypeSize(llt)) {
                Logger::cout() << "got size " << getABITypeSize(t) << ", expected " << getABITypeSize(llt) << '\n';
                assert(0 && "type size mismatch");
            }
            sptr = DtoBitCast(sptr, getPtrToType(t));
            Logger::cout() << "sptr type is now: " << *t << '\n';
        }
    }

    // build
    unsigned j = 0;
    for (unsigned i=0; i<n; ++i)
    {
        Expression* vx = (Expression*)elements->data[i];
        if (!vx) continue;

        Logger::cout() << "getting index " << j << " of " << *sptr << '\n';
        llvm::Value* arrptr = DtoGEPi(sptr,0,j,"tmp",p->scopebb());
        DValue* darrptr = new DVarValue(vx->type, arrptr, true);

        p->exps.push_back(IRExp(NULL,vx,darrptr));
        DValue* ve = vx->toElem(p);
        p->exps.pop_back();

        if (!ve->inPlace())
            DtoAssign(darrptr, ve);

        j++;
    }

    return new DImValue(type, sptr, true);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* StructLiteralExp::toConstElem(IRState* p)
{
    Logger::print("StructLiteralExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    unsigned n = elements->dim;
    std::vector<llvm::Constant*> vals(n, NULL);

    for (unsigned i=0; i<n; ++i)
    {
        Expression* vx = (Expression*)elements->data[i];
        vals[i] = vx->toConstElem(p);
    }

    assert(DtoDType(type)->ty == Tstruct);
    const llvm::Type* t = DtoType(type);
    const llvm::StructType* st = isaStruct(t);
    return llvm::ConstantStruct::get(st,vals);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* InExp::toElem(IRState* p)
{
    Logger::print("InExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    DValue* key = e1->toElem(p);
    DValue* aa = e2->toElem(p);

    return DtoAAIn(type, aa, key);
}

DValue* RemoveExp::toElem(IRState* p)
{
    Logger::print("RemoveExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    DValue* aa = e1->toElem(p);
    DValue* key = e2->toElem(p);

    DtoAARemove(aa, key);

    return NULL; // does not produce anything useful
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* AssocArrayLiteralExp::toElem(IRState* p)
{
    Logger::print("AssocArrayLiteralExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(keys);
    assert(values);
    assert(keys->dim == values->dim);

    Type* aatype = DtoDType(type);
    Type* vtype = aatype->next;

    DValue* aa;
    if (p->topexp() && p->topexp()->e2 == this)
    {
        aa = p->topexp()->v;
    }
    else
    {
        llvm::Value* tmp = new llvm::AllocaInst(DtoType(type),"aaliteral",p->topallocapoint());
        aa = new DVarValue(type, tmp, true);
    }

    const size_t n = keys->dim;
    for (size_t i=0; i<n; ++i)
    {
        Expression* ekey = (Expression*)keys->data[i];
        Expression* eval = (Expression*)values->data[i];

        Logger::println("(%u) aa[%s] = %s", i, ekey->toChars(), eval->toChars());

        // index
        DValue* key = ekey->toElem(p);
        DValue* mem = DtoAAIndex(vtype, aa, key);

        // store
        DValue* val = eval->toElem(p);
        DtoAssign(mem, val);
    }

    return aa;
}

//////////////////////////////////////////////////////////////////////////////////////////

#define STUB(x) DValue *x::toElem(IRState * p) {error("Exp type "#x" not implemented: %s", toChars()); fatal(); return 0; }
//STUB(IdentityExp);
//STUB(CondExp);
//STUB(EqualExp);
//STUB(InExp);
//STUB(CmpExp);
//STUB(AndAndExp);
//STUB(OrOrExp);
//STUB(AndExp);
//STUB(AndAssignExp);
//STUB(OrExp);
//STUB(OrAssignExp);
//STUB(XorExp);
//STUB(XorAssignExp);
//STUB(ShrExp);
//STUB(ShrAssignExp);
//STUB(ShlExp);
//STUB(ShlAssignExp);
//STUB(UshrExp);
//STUB(UshrAssignExp);
//STUB(DivExp);
//STUB(DivAssignExp);
//STUB(MulExp);
//STUB(MulAssignExp);
//STUB(ModExp);
//STUB(ModAssignExp);
//STUB(CatExp);
//STUB(CatAssignExp);
//STUB(AddExp);
//STUB(AddAssignExp);
STUB(Expression);
//STUB(MinExp);
//STUB(MinAssignExp);
//STUB(PostExp);
//STUB(NullExp);
//STUB(ThisExp);
//STUB(CallExp);
STUB(DotTypeExp);
STUB(TypeDotIdExp);
//STUB(DotVarExp);
//STUB(AssertExp);
//STUB(FuncExp);
//STUB(DelegateExp);
//STUB(VarExp);
//STUB(DeclarationExp);
//STUB(NewExp);
//STUB(SymOffExp);
STUB(ScopeExp);
//STUB(AssignExp);

STUB(TypeExp);
//STUB(RealExp);
//STUB(ComplexExp);
//STUB(StringExp);
//STUB(IntegerExp);
STUB(BoolExp);

//STUB(NotExp);
//STUB(ComExp);
//STUB(NegExp);
//STUB(PtrExp);
//STUB(AddrExp);
//STUB(SliceExp);
//STUB(CastExp);
//STUB(DeleteExp);
//STUB(IndexExp);
//STUB(CommaExp);
//STUB(ArrayLengthExp);
//STUB(HaltExp);
//STUB(RemoveExp);
//STUB(ArrayLiteralExp);
//STUB(AssocArrayLiteralExp);
//STUB(StructLiteralExp);
STUB(TupleExp);

#define CONSTSTUB(x) llvm::Constant* x::toConstElem(IRState * p) {error("const Exp type "#x" not implemented: '%s' type: '%s'", toChars(), type->toChars()); fatal(); return NULL; }
CONSTSTUB(Expression);
//CONSTSTUB(IntegerExp);
//CONSTSTUB(RealExp);
//CONSTSTUB(NullExp);
//CONSTSTUB(ComplexExp);
//CONSTSTUB(StringExp);
//CONSTSTUB(VarExp);
//CONSTSTUB(ArrayLiteralExp);
CONSTSTUB(AssocArrayLiteralExp);
//CONSTSTUB(StructLiteralExp);

unsigned Type::totym() { return 0; }

type * Type::toCtype()
{
    assert(0);
    return 0;
}

type * Type::toCParamtype()
{
    assert(0);
    return 0;
}
Symbol * Type::toSymbol()
{
    assert(0);
    return 0;
}

type *
TypeTypedef::toCtype()
{
    assert(0);
    return 0;
}

type *
TypeTypedef::toCParamtype()
{
    assert(0);
    return 0;
}

void
TypedefDeclaration::toDebug()
{
    assert(0);
}


type *
TypeEnum::toCtype()
{
    assert(0);
    return 0;
}

type *
TypeStruct::toCtype()
{
    assert(0);
    return 0;
}

void
StructDeclaration::toDebug()
{
    assert(0);
}

Symbol * TypeClass::toSymbol()
{
    assert(0);
    return 0;
}

unsigned TypeFunction::totym()
{
    assert(0);
    return 0;
}

type * TypeFunction::toCtype()
{
    assert(0);
    return 0;
}

type * TypeSArray::toCtype()
{
    assert(0);
    return 0;
}

type *TypeSArray::toCParamtype()
{
    assert(0);
    return 0;
}

type * TypeDArray::toCtype()
{
    assert(0);
    return 0;
}

type * TypeAArray::toCtype()
{
    assert(0);
    return 0;
}

type * TypePointer::toCtype()
{
    assert(0);
    return 0;
}

type * TypeDelegate::toCtype()
{
    assert(0);
    return 0;
}

type * TypeClass::toCtype()
{
    assert(0);
    return 0;
}

void ClassDeclaration::toDebug()
{
    assert(0);
}

//////////////////////////////////////////////////////////////////////////////

void
EnumDeclaration::toDebug()
{
    assert(0);
}

int Dsymbol::cvMember(unsigned char*)
{
    assert(0);
    return 0;
}
int EnumDeclaration::cvMember(unsigned char*)
{
    assert(0);
    return 0;
}
int FuncDeclaration::cvMember(unsigned char*)
{
    assert(0);
    return 0;
}
int VarDeclaration::cvMember(unsigned char*)
{
    assert(0);
    return 0;
}
int TypedefDeclaration::cvMember(unsigned char*)
{
    assert(0);
    return 0;
}

void obj_includelib(char*){}

AsmStatement::AsmStatement(Loc loc, Token *tokens) :
    Statement(loc)
{
    this->tokens = tokens;
}
Statement *AsmStatement::syntaxCopy()
{
    /*error("%s: inline asm is not yet implemented", loc.toChars());
    fatal();
    assert(0);*/
    return 0;
}

Statement *AsmStatement::semantic(Scope *sc)
{
    return Statement::semantic(sc);
}


void AsmStatement::toCBuffer(OutBuffer *buf, HdrGenState *hgs)
{
    Statement::toCBuffer(buf, hgs);
}

int AsmStatement::comeFrom()
{
    /*error("%s: inline asm is not yet implemented", loc.toChars());
    fatal();
    assert(0);*/
    return 0;
}

void
backend_init()
{
    // now lazily loaded
    //LLVM_D_InitRuntime();
}

void
backend_term()
{
    LLVM_D_FreeRuntime();
}
