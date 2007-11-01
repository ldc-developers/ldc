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

#include "total.h"
#include "init.h"
#include "symbol.h"
#include "mtype.h"
#include "hdrgen.h"
#include "port.h"

#include "gen/irstate.h"
#include "gen/elem.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/runtime.h"
#include "gen/arrays.h"

//////////////////////////////////////////////////////////////////////////////////////////

elem* DeclarationExp::toElem(IRState* p)
{
    Logger::print("DeclarationExp::toElem: %s | T=%s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;

    // variable declaration
    if (VarDeclaration* vd = declaration->isVarDeclaration())
    {
        Logger::println("VarDeclaration");

        // static
        if (vd->isDataseg())
        {
            vd->toObjFile();
        }
        else
        {
            Logger::println("vdtype = %s", vd->type->toChars());
            // referenced by nested delegate?
            if (vd->nestedref) {
                Logger::println("has nestedref set");
                vd->llvmValue = p->func().decl->llvmNested;
                assert(vd->llvmValue);
                assert(vd->llvmNestedIndex >= 0);
            }
            // normal stack variable
            else {
                // allocate storage on the stack
                const llvm::Type* lltype = DtoType(vd->type);
                llvm::AllocaInst* allocainst = new llvm::AllocaInst(lltype, vd->toChars(), p->topallocapoint());
                //allocainst->setAlignment(vd->type->alignsize()); // TODO
                vd->llvmValue = allocainst;
            }
            DtoInitializer(vd->init);
        }
    }
    // struct declaration
    else if (StructDeclaration* s = declaration->isStructDeclaration())
    {
        Logger::println("StructDeclaration");
        s->toObjFile();
    }
    // function declaration
    else if (FuncDeclaration* f = declaration->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        f->toObjFile();
    }
    // alias declaration
    else if (AliasDeclaration* a = declaration->isAliasDeclaration())
    {
        Logger::println("AliasDeclaration - no work");
        // do nothing
    }
    else if (EnumDeclaration* e = declaration->isEnumDeclaration())
    {
        Logger::println("EnumDeclaration - no work");
        // do nothing
    }
    // unsupported declaration
    else
    {
        error("Only Var/Struct-Declaration is supported for DeclarationExp");
        assert(0);
    }
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* VarExp::toElem(IRState* p)
{
    Logger::print("VarExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;

    assert(var);
    if (VarDeclaration* vd = var->isVarDeclaration())
    {
        Logger::println("VarDeclaration %s", vd->toChars());

        // _arguments
        if (vd->ident == Id::_arguments)
        {
            Logger::println("Id::_arguments");
            vd->llvmValue = p->func().decl->llvmArguments;
            assert(vd->llvmValue);
            e->mem = vd->llvmValue;
            e->type = elem::VAR;
        }
        // _argptr
        else if (vd->ident == Id::_argptr)
        {
            Logger::println("Id::_argptr");
            vd->llvmValue = p->func().decl->llvmArgPtr;
            assert(vd->llvmValue);
            e->mem = vd->llvmValue;
            e->type = elem::VAR;
        }
        // _dollar
        else if (vd->ident == Id::dollar)
        {
            Logger::println("Id::dollar");
            assert(!p->arrays.empty());
            llvm::Value* tmp = DtoGEPi(p->arrays.back(),0,0,"tmp",p->scopebb());
            e->val = new llvm::LoadInst(tmp,"tmp",p->scopebb());
            e->type = elem::VAL;
        }
        // typeinfo
        else if (TypeInfoDeclaration* tid = vd->isTypeInfoDeclaration())
        {
            Logger::println("TypeInfoDeclaration");
            tid->toObjFile();
            assert(tid->llvmValue);
            const llvm::Type* vartype = DtoType(type);
            if (tid->llvmValue->getType() != llvm::PointerType::get(vartype))
                e->mem = p->ir->CreateBitCast(tid->llvmValue, vartype, "tmp");
            else
                e->mem = tid->llvmValue;
            e->type = elem::VAR;
        }
        // nested variable
        else if (vd->nestedref) {
            Logger::println("nested variable");
            e->mem = DtoNestedVariable(vd);
            e->type = elem::VAR;
            e->vardecl = vd;
        }
        // function parameter
        else if (vd->isParameter()) {
            Logger::println("function param");
            if (!vd->llvmValue) {
                // TODO: determine this properly
                // this happens when the DMD frontend generates by pointer wrappers for struct opEquals(S) and opCmp(S)
                vd->llvmValue = &p->func().func->getArgumentList().back();
            }
            if (vd->isRef() || vd->isOut()) {
                e->mem = vd->llvmValue;
                e->type = elem::VAR;
            }
            else {
                if (DtoIsPassedByRef(vd->type)) {
                    e->mem = vd->llvmValue;
                    e->type = elem::VAR;
                }
                else {
                    if (llvm::isa<llvm::Argument>(vd->llvmValue)) {
                        e->val = vd->llvmValue;
                        e->type = elem::VAL;
                        e->vardecl = vd;
                    }
                    else if (llvm::isa<llvm::AllocaInst>(vd->llvmValue)) {
                        e->mem = vd->llvmValue;
                        e->type = elem::VAR;
                    }
                    else
                    assert(0);
                }
            }
        }
        else {
            // take care of forward references of global variables
            if (!vd->llvmTouched && vd->isDataseg())
                vd->toObjFile();
            assert(vd->llvmValue);
            e->mem = vd->llvmValue;
            e->vardecl = vd;
            e->type = elem::VAR;
        }

        assert(e->mem || e->val);
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        if (fdecl->llvmInternal != LLVMva_arg && fdecl->llvmValue == 0)
            fdecl->toObjFile();
        e->val = fdecl->llvmValue;
        e->type = elem::FUNC;
        e->funcdecl = fdecl;
    }
    else if (SymbolDeclaration* sdecl = var->isSymbolDeclaration())
    {
        // this seems to be the static initialiser for structs
        Type* sdecltype = DtoDType(sdecl->type);
        Logger::print("Sym: type=%s\n", sdecltype->toChars());
        assert(sdecltype->ty == Tstruct);
        TypeStruct* ts = (TypeStruct*)sdecltype;
        e->mem = ts->llvmInit;
        assert(e->mem);
        e->type = elem::VAR;
    }
    else
    {
        assert(0 && "Unimplemented VarExp type");
    }

    return e;
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
        assert(ts->sym->llvmInitZ);
        return ts->sym->llvmInitZ;
    }
    assert(0 && "Only support const var exp is SymbolDeclaration");
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* IntegerExp::toElem(IRState* p)
{
    Logger::print("IntegerExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    e->val = toConstElem(p);
    e->type = elem::CONST;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* IntegerExp::toConstElem(IRState* p)
{
    Logger::print("IntegerExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    const llvm::Type* t = DtoType(type);
    if (llvm::isa<llvm::PointerType>(t)) {
        llvm::Constant* i = llvm::ConstantInt::get(DtoSize_t(),(uint64_t)value,false);
        return llvm::ConstantExpr::getIntToPtr(i, t);
    }
    else if (llvm::isa<llvm::IntegerType>(t)) {
        return llvm::ConstantInt::get(t,(uint64_t)value,!type->isunsigned());
    }
    assert(0);
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* RealExp::toElem(IRState* p)
{
    Logger::print("RealExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    e->val = toConstElem(p);
    e->type = elem::CONST;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* RealExp::toConstElem(IRState* p)
{
    Logger::print("RealExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    Type* t = DtoDType(type);
    const llvm::Type* fty = DtoType(t);
    if (t->ty == Tfloat32 || t->ty == Timaginary32)
        return llvm::ConstantFP::get(fty,float(value));
    else if (t->ty == Tfloat64 || t->ty == Timaginary64 || t->ty == Tfloat80 || t->ty == Timaginary80)
        return llvm::ConstantFP::get(fty,double(value));
    assert(0);
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* NullExp::toElem(IRState* p)
{
    Logger::print("NullExp::toElem(type=%s): %s\n", type->toChars(),toChars());
    LOG_SCOPE;
    elem* e = new elem;
    e->val = toConstElem(p);
    e->type = elem::NUL;
    //Logger::cout() << "null value is now " << *e->val << '\n';
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* NullExp::toConstElem(IRState* p)
{
    Logger::print("NullExp::toConstElem(type=%s): %s\n", type->toChars(),toChars());
    LOG_SCOPE;
    const llvm::Type* t = DtoType(type);
    if (type->ty == Tarray) {
        assert(llvm::isa<llvm::StructType>(t));
        return llvm::ConstantAggregateZero::get(t);
    }
    else {
        return llvm::Constant::getNullValue(t);
    }
    assert(0);
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* StringExp::toElem(IRState* p)
{
    Logger::print("StringExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* dtype = DtoDType(type);

    assert(dtype->next->ty == Tchar && "Only char is supported");
    assert(sz == 1);

    const llvm::Type* ct = DtoType(dtype->next);
    //printf("ct = %s\n", type->next->toChars());
    const llvm::ArrayType* at = llvm::ArrayType::get(ct,len+1);

    uint8_t* str = (uint8_t*)string;
    std::string cont((char*)str, len);

    llvm::Constant* _init = llvm::ConstantArray::get(cont,true);

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;//WeakLinkage;
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(at,true,_linkage,_init,"stringliteral",gIR->module);

    llvm::ConstantInt* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Constant* idxs[2] = { zero, zero };
    llvm::Constant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);

    elem* e = new elem;

    if (dtype->ty == Tarray) {
        llvm::Constant* clen = llvm::ConstantInt::get(DtoSize_t(),len,false);
        if (!p->topexp() || p->topexp()->e2 != this) {
            llvm::Value* tmpmem = new llvm::AllocaInst(DtoType(dtype),"tmp",p->topallocapoint());
            DtoSetArray(tmpmem, clen, arrptr);
            e->mem = tmpmem;
        }
        else if (p->topexp()->e2 == this) {
            llvm::Value* arr = p->topexp()->v;
            assert(arr);
            DtoSetArray(arr, clen, arrptr);
            e->inplace = true;
        }
        else
        assert(0);
    }
    else if (dtype->ty == Tsarray) {
        const llvm::Type* dstType = llvm::PointerType::get(llvm::ArrayType::get(ct, len));
        e->mem = new llvm::BitCastInst(gvar, dstType, "tmp", gIR->scopebb());
    }
    else if (dtype->ty == Tpointer) {
        e->mem = arrptr;
    }
    else {
        assert(0);
    }

    e->type = elem::VAL;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* StringExp::toConstElem(IRState* p)
{
    Logger::print("StringExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    uint8_t* str = (uint8_t*)string;
    std::string cont((char*)str, len);

    Type* t = DtoDType(type);

    if (t->ty == Tsarray) {
        return llvm::ConstantArray::get(cont,false);
    }
    llvm::Constant* _init = llvm::ConstantArray::get(cont,true);

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;//WeakLinkage;
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(_init->getType(),true,_linkage,_init,"stringliteral",gIR->module);

    llvm::ConstantInt* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Constant* idxs[2] = { zero, zero };
    llvm::Constant* arrptr = llvm::ConstantExpr::getGetElementPtr(gvar,idxs,2);

    if (t->ty == Tpointer) {
        return arrptr;
    }

    if (t->ty == Tarray) {
        llvm::Constant* clen = llvm::ConstantInt::get(DtoSize_t(),len,false);
        return DtoConstSlice(clen, arrptr);
    }

    assert(0);
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AssignExp::toElem(IRState* p)
{
    Logger::print("AssignExp::toElem: %s | %s = %s\n", toChars(), e1->type->toChars(), e2->type ? e2->type->toChars() : 0);
    LOG_SCOPE;

    p->exps.push_back(IRExp(e1,e2,NULL));

    elem* l = e1->toElem(p);
    p->topexp()->v = l->mem;
    elem* r = e2->toElem(p);

    p->exps.pop_back();

    if (l->type == elem::ARRAYLEN)
    {
        DtoResizeDynArray(l->mem, r->getValue());
        delete r;
        delete l;
        return 0;
    }

    // handle function argument - allocate temp storage for it :/ annoying
    if (l->mem == 0) {
        assert(l->val);
        if (llvm::isa<llvm::Argument>(l->val))
            DtoGiveArgumentStorage(l);
        else {
            Logger::cout() << "here it comes... " << *l->val << '\n';
            assert(0);
        }
    }
    //e->val = l->store(r->getValue());

    Type* e1type = DtoDType(e1->type);
    Type* e2type = DtoDType(e2->type);
    TY e1ty = e1type->ty;
    TY e2ty = e2type->ty;

    elem* e = new elem;
    e->type = elem::VAR;

    // struct
    if (e1ty == Tstruct) {
        e->mem = l->mem;
        // struct + struct
        if (e2ty == Tstruct) {
            // struct literals do the assignment themselvs (in place)
            if (!r->inplace) {
                DtoStructCopy(l->mem,r->getValue());
            }
            else {
                e->inplace = true;
            }
        }
        // struct + const int
        else if (e2type->isintegral()){
            IntegerExp* iexp = (IntegerExp*)e2;
            assert(iexp->value == 0 && "Only integral struct initializer allowed is zero");
            DtoStructZeroInit(l->mem);
        }
        // :x
        else
        assert(0 && "struct = unknown");
    }
    else if (e1ty == Tsarray) {
        assert(0 && "static array not supported");
    }
    else if (e1ty == Tarray) {
        if (e2type->isscalar() || e2type->ty == Tclass){
            DtoArrayInit(l->mem, r->getValue());
        }
        else if (e2ty == Tarray) {
            //new llvm::StoreInst(r->val,l->val,p->scopebb());
            if (r->type == elem::NUL) {
                llvm::Constant* c = llvm::cast<llvm::Constant>(r->val);
                assert(c->isNullValue());
                DtoNullArray(l->mem);
                e->mem = l->mem;
            }
            else if (r->type == elem::SLICE) {
                if (l->type == elem::SLICE) {
                    DtoArrayCopy(l,r);
                    e->type = elem::SLICE;
                    e->mem = l->mem;
                    e->arg = l->arg;
                }
                else {
                    DtoSetArray(l->mem,r->arg,r->mem);
                    e->mem = l->mem;
                }
            }
            else {
                // new expressions write directly to the array reference
                // so do string literals
                e->mem = l->mem;
                if (!r->inplace) {
                    assert(r->mem);
                    DtoArrayAssign(l->mem, r->mem);
                }
                else {
                    e->inplace = true;
                }
            }
        }
        else
        assert(0);
    }
    else if (e1ty == Tpointer) {
        e->mem = l->mem;
        if (e2ty == Tpointer) {
            llvm::Value* v = r->field ? r->mem : r->getValue();
            Logger::cout() << "*=*: " << *v << ", " << *l->mem << '\n';
            new llvm::StoreInst(v, l->mem, p->scopebb());
        }
        else
        assert(0);
    }
    else if (e1ty == Tclass) {
        if (e2ty == Tclass) {
            llvm::Value* tmp = r->getValue();
            Logger::cout() << "tmp: " << *tmp << " ||| " << *l->mem << '\n';
            // assignment to this in constructor special case
            if (l->isthis) {
                FuncDeclaration* fdecl = p->func().decl;
                // respecify the this param
                if (!llvm::isa<llvm::AllocaInst>(fdecl->llvmThisVar))
                    fdecl->llvmThisVar = new llvm::AllocaInst(tmp->getType(), "newthis", p->topallocapoint());
                new llvm::StoreInst(tmp, fdecl->llvmThisVar, p->scopebb());
                e->mem = fdecl->llvmThisVar;
            }
            // regular class ref -> class ref assignment
            else {
                new llvm::StoreInst(tmp, l->mem, p->scopebb());
                e->mem = l->mem;
            }
        }
        else
        assert(0);
    }
    else if (e1ty == Tdelegate) {
        Logger::println("Assigning to delegate");
        if (e2ty == Tdelegate) {
            if (r->type == elem::NUL) {
                llvm::Constant* c = llvm::cast<llvm::Constant>(r->val);
                if (c->isNullValue()) {
                    DtoNullDelegate(l->mem);
                    e->mem = l->mem;
                }
                else
                assert(0);
            }
            else if (r->inplace) {
                // do nothing
                e->inplace = true;
                e->mem = l->mem;
            }
            else {
                DtoDelegateCopy(l->mem, r->getValue());
                e->mem = l->mem;
            }
        }
        else
        assert(0);
    }
    // !struct && !array && !pointer && !class
    else {
        Logger::cout() << *l->mem << '\n';
        new llvm::StoreInst(r->getValue(),l->mem,p->scopebb());
        e->mem = l->mem;
    }

    delete r;
    delete l;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AddExp::toElem(IRState* p)
{
    Logger::print("AddExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* t = DtoDType(type);
    Type* e1type = DtoDType(e1->type);
    Type* e2type = DtoDType(e2->type);

    if (e1type != e2type) {
        if (e1type->ty == Tpointer && e1type->next->ty == Tstruct) {
            //assert(l->field);
            assert(r->type == elem::CONST);
            llvm::ConstantInt* cofs = llvm::cast<llvm::ConstantInt>(r->val);

            TypeStruct* ts = (TypeStruct*)e1type->next;
            std::vector<unsigned> offsets;
            e->mem = DtoIndexStruct(l->getValue(), ts->sym, t->next, cofs->getZExtValue(), offsets);
            e->type = elem::VAR;
            e->field = true;
        }
        else if (e1->type->ty == Tpointer) {
            e->val = new llvm::GetElementPtrInst(l->getValue(), r->getValue(), "tmp", p->scopebb());
            e->type = elem::VAR;
        }
        else {
            assert(0);
        }
    }
    else {
        e->val = llvm::BinaryOperator::createAdd(l->getValue(), r->getValue(), "tmp", p->scopebb());
        e->type = elem::VAL;
    }
    delete l;
    delete r;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AddAssignExp::toElem(IRState* p)
{
    Logger::print("AddAssignExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* e1type = DtoDType(e1->type);

    elem* e = new elem;
    llvm::Value* val = 0;
    if (e1type->ty == Tpointer) {
        val = e->mem = new llvm::GetElementPtrInst(l->getValue(),r->getValue(),"tmp",p->scopebb());
    }
    else {
        val = e->val = llvm::BinaryOperator::createAdd(l->getValue(),r->getValue(),"tmp",p->scopebb());
    }

    /*llvm::Value* storeVal = l->storeVal ? l->storeVal : l->val;
    if (llvm::isa<llvm::PointerType>(storeVal->getType()) && storeVal->getType()->getContainedType(0) != tmp->getType())
    {
        tmp = DtoPointedType(storeVal, tmp);
    }*/

    if (l->mem == 0)
        DtoGiveArgumentStorage(l);
    new llvm::StoreInst(val,l->mem,p->scopebb());
    e->type = elem::VAR;

    delete l;
    delete r;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* MinExp::toElem(IRState* p)
{
    Logger::print("MinExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    llvm::Value* left = l->getValue();
    if (llvm::isa<llvm::PointerType>(left->getType()))
        left = new llvm::PtrToIntInst(left,DtoSize_t(),"tmp",p->scopebb());

    llvm::Value* right = r->getValue();
    if (llvm::isa<llvm::PointerType>(right->getType()))
        right = new llvm::PtrToIntInst(right,DtoSize_t(),"tmp",p->scopebb());

    e->val = llvm::BinaryOperator::createSub(left,right,"tmp",p->scopebb());
    e->type = elem::VAL;

    const llvm::Type* totype = DtoType(type);
    if (e->val->getType() != totype) {
        assert(0);
        assert(llvm::isa<llvm::PointerType>(e->val->getType()));
        assert(llvm::isa<llvm::IntegerType>(totype));
        e->val = new llvm::IntToPtrInst(e->val,totype,"tmp",p->scopebb());
    }

    delete l;
    delete r;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* MinAssignExp::toElem(IRState* p)
{
    Logger::print("MinAssignExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* e1type = DtoDType(e1->type);

    llvm::Value* tmp = 0;
    if (e1type->ty == Tpointer) {
        tmp = r->getValue();
        llvm::Value* zero = llvm::ConstantInt::get(tmp->getType(),0,false);
        tmp = llvm::BinaryOperator::createSub(zero,tmp,"tmp",p->scopebb());
        tmp = new llvm::GetElementPtrInst(l->getValue(),tmp,"tmp",p->scopebb());
    }
    else {
        tmp = llvm::BinaryOperator::createSub(l->getValue(),r->getValue(),"tmp",p->scopebb());
    }

    /*llvm::Value* storeVal = l->storeVal ? l->storeVal : l->val;
    if (storeVal->getType()->getContainedType(0) != tmp->getType())
    {
        tmp = DtoPointedType(storeVal, tmp);
    }*/

    if (l->mem == 0)
        DtoGiveArgumentStorage(l);
    new llvm::StoreInst(tmp, l->mem, p->scopebb());

    delete l;
    delete r;

    elem* e = new elem;
    e->val = tmp;
    e->type = elem::VAR;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* MulExp::toElem(IRState* p)
{
    Logger::print("MulExp::toElem: %s\n", toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);
    llvm::Value* vl = l->getValue();
    llvm::Value* vr = r->getValue();
    Logger::cout() << "mul: " << *vl << ", " << *vr << '\n';
    e->val = llvm::BinaryOperator::createMul(vl,vr,"tmp",p->scopebb());
    e->type = elem::VAL;
    delete l;
    delete r;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* MulAssignExp::toElem(IRState* p)
{
    Logger::print("MulAssignExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);
    llvm::Value* vl = l->getValue();
    llvm::Value* vr = r->getValue();
    Logger::cout() << "mulassign: " << *vl << ", " << *vr << '\n';
    llvm::Value* tmp = llvm::BinaryOperator::createMul(vl,vr,"tmp",p->scopebb());

    /*llvm::Value* storeVal = l->storeVal ? l->storeVal : l->val;
    if (storeVal->getType()->getContainedType(0) != tmp->getType())
    {
        tmp = DtoPointedType(storeVal, tmp);
    }*/

    if (l->mem == 0)
        DtoGiveArgumentStorage(l);
    new llvm::StoreInst(tmp,l->mem,p->scopebb());

    delete l;
    delete r;

    elem* e = new elem;
    e->val = tmp;
    e->type = elem::VAR;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* DivExp::toElem(IRState* p)
{
    Logger::print("DivExp::toElem: %s\n", toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* t = DtoDType(type);

    if (t->isunsigned())
        e->val = llvm::BinaryOperator::createUDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (t->isintegral())
        e->val = llvm::BinaryOperator::createSDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (t->isfloating())
        e->val = llvm::BinaryOperator::createFDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else
        assert(0);
    e->type = elem::VAL;
    delete l;
    delete r;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* DivAssignExp::toElem(IRState* p)
{
    Logger::print("DivAssignExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* t = DtoDType(type);

    llvm::Value* tmp;
    if (t->isunsigned())
        tmp = llvm::BinaryOperator::createUDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (t->isintegral())
        tmp = llvm::BinaryOperator::createSDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (t->isfloating())
        tmp = llvm::BinaryOperator::createFDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else
        assert(0);

    /*llvm::Value* storeVal = l->storeVal ? l->storeVal : l->val;
    if (storeVal->getType()->getContainedType(0) != tmp->getType())
    {
        tmp = DtoPointedType(storeVal, tmp);
    }*/

    if (l->mem == 0)
        DtoGiveArgumentStorage(l);
    new llvm::StoreInst(tmp,l->mem,p->scopebb());

    delete l;
    delete r;

    elem* e = new elem;
    e->val = tmp;
    e->type = elem::VAR;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* ModExp::toElem(IRState* p)
{
    Logger::print("ModExp::toElem: %s\n", toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* t = DtoDType(type);

    if (t->isunsigned())
        e->val = llvm::BinaryOperator::createURem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (t->isintegral())
        e->val = llvm::BinaryOperator::createSRem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (t->isfloating())
        e->val = llvm::BinaryOperator::createFRem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else
        assert(0);
    e->type = elem::VAL;
    delete l;
    delete r;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* ModAssignExp::toElem(IRState* p)
{
    Logger::print("ModAssignExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* t = DtoDType(type);

    llvm::Value* tmp;
    if (t->isunsigned())
        tmp = llvm::BinaryOperator::createURem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (t->isintegral())
        tmp = llvm::BinaryOperator::createSRem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (t->isfloating())
        tmp = llvm::BinaryOperator::createFRem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else
        assert(0);

    /*llvm::Value* storeVal = l->storeVal ? l->storeVal : l->val;
    if (storeVal->getType()->getContainedType(0) != tmp->getType())
    {
        tmp = DtoPointedType(storeVal, tmp);
    }*/

    if (l->mem == 0)
        DtoGiveArgumentStorage(l);
    new llvm::StoreInst(tmp,l->mem,p->scopebb());

    delete l;
    delete r;

    elem* e = new elem;
    e->val = tmp;
    e->type = elem::VAR;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* CallExp::toElem(IRState* p)
{
    Logger::print("CallExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    elem* e = new elem;
    elem* fn = e1->toElem(p);

    TypeFunction* tf = 0;
    Type* e1type = DtoDType(e1->type);

    bool delegateCall = false;
    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty,0,false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty,1,false);
    LINK dlink = LINKdefault;

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

    // va args
    bool va_magic = false;
    bool va_intrinsic = false;
    if (fn->funcdecl) {
        if (fn->funcdecl->llvmInternal == LLVMva_intrinsic) {
            va_magic = true;
            va_intrinsic = true;
        }
        else if (fn->funcdecl->llvmInternal == LLVMva_start) {
            va_magic = true;
        }
        else if (fn->funcdecl->llvmInternal == LLVMva_arg) {
            Argument* fnarg = Argument::getNth(tf->parameters, 0);
            Expression* exp = (Expression*)arguments->data[0];
            elem* expelem = exp->toElem(p);
            assert(expelem->mem);
            elem* e = new elem;
            Type* t = DtoDType(type);
            const llvm::Type* llt = DtoType(type);
            if (DtoIsPassedByRef(t))
                llt = llvm::PointerType::get(llt);
            e->type = elem::VAL;
            e->val = p->ir->CreateVAArg(expelem->mem,llt,"tmp");
            delete expelem;
            return e;
        }
    }

    // args
    size_t n = arguments->dim;
    if (fn->funcdecl && fn->funcdecl->llvmInternal == LLVMva_start)
        n = 1;
    if (fn->arg || delegateCall) n++;
    if (retinptr) n++;
    if (tf->linkage == LINKd && tf->varargs == 1) n+=2;
    if (fn->funcdecl && fn->funcdecl->isNested()) n++;

    llvm::Value* funcval = fn->getValue();
    assert(funcval != 0);
    std::vector<llvm::Value*> llargs(n, 0);

    const llvm::FunctionType* llfnty = 0;

    // normal function call
    if (llvm::isa<llvm::FunctionType>(funcval->getType())) {
        llfnty = llvm::cast<llvm::FunctionType>(funcval->getType());
    }
    // pointer to something
    else if (llvm::isa<llvm::PointerType>(funcval->getType())) {
        // pointer to function pointer - I think this not really supposed to happen, but does :/
        // seems like sometimes we get a func* other times a func**
        if (llvm::isa<llvm::PointerType>(funcval->getType()->getContainedType(0))) {
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
        }
        // function pointer
        if (llvm::isa<llvm::FunctionType>(funcval->getType()->getContainedType(0))) {
            //Logger::cout() << "function pointer type:\n" << *funcval << '\n';
            llfnty = llvm::cast<llvm::FunctionType>(funcval->getType()->getContainedType(0));
        }
        // struct pointer - delegate
        else if (llvm::isa<llvm::StructType>(funcval->getType()->getContainedType(0))) {
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

    // hidden struct return arguments
    if (retinptr) {
        if (topexp && topexp->e2 == this) {
            assert(topexp->v);
            assert(llvm::isa<llvm::StructType>(topexp->v->getType()->getContainedType(0)));
            llargs[j] = topexp->v;
            if (DtoIsPassedByRef(tf->next)) {
                e->inplace = true;
            }
            else
            assert(0);
        }
        else {
            llargs[j] = new llvm::AllocaInst(argiter->get()->getContainedType(0),"rettmp",p->topallocapoint());
        }
        ++j;
        ++argiter;
        e->type = elem::VAR;
    }
    else {
        e->type = elem::VAL;
    }

    // this arguments
    if (fn->arg) {
        Logger::println("This Call");
        if (fn->arg->getType() != argiter->get()) {
            //Logger::cout() << *fn->thisparam << '|' << *argiter->get() << '\n';
            llargs[j] = new llvm::BitCastInst(fn->arg, argiter->get(), "tmp", p->scopebb());
        }
        else {
            llargs[j] = fn->arg;
        }
        ++j;
        ++argiter;
    }
    // delegate context arguments
    else if (delegateCall) {
        Logger::println("Delegate Call");
        llvm::Value* contextptr = DtoGEP(fn->mem,zero,zero,"tmp",p->scopebb());
        llargs[j] = new llvm::LoadInst(contextptr,"tmp",p->scopebb());
        ++j;
        ++argiter;
    }
    // nested call
    else if (fn->funcdecl && fn->funcdecl->isNested()) {
        Logger::println("Nested Call");
        llvm::Value* contextptr = p->func().decl->llvmNested;
        assert(contextptr);
        llargs[j] = p->ir->CreateBitCast(contextptr, llvm::PointerType::get(llvm::Type::Int8Ty), "tmp");
        ++j;
        ++argiter;
    }

    // va arg function special argument passing
    if (va_magic) {
        size_t n = va_intrinsic ? arguments->dim : 1;
        for (int i=0; i<n; i++,j++)
        {
            Argument* fnarg = Argument::getNth(tf->parameters, i);
            Expression* exp = (Expression*)arguments->data[i];
            elem* expelem = exp->toElem(p);
            assert(expelem->mem);
            llargs[j] = p->ir->CreateBitCast(expelem->mem, llvm::PointerType::get(llvm::Type::Int8Ty), "tmp");
            delete expelem;
        }
    }
    // regular arguments
    else {
        if (tf->linkage == LINKd && tf->varargs == 1)
        {
            Logger::println("doing d-style variadic arguments");

            std::vector<const llvm::Type*> vtypes;
            std::vector<llvm::Value*> vvalues;
            std::vector<llvm::Value*> vtypeinfos;

            for (int i=0; i<arguments->dim; i++) {
                Argument* fnarg = Argument::getNth(tf->parameters, i);
                Expression* argexp = (Expression*)arguments->data[i];
                vvalues.push_back(DtoArgument(NULL, fnarg, argexp));
                vtypes.push_back(vvalues.back()->getType());

                TypeInfoDeclaration* tidecl = argexp->type->getTypeInfoDeclaration();
                tidecl->toObjFile();
                assert(tidecl->llvmValue);
                vtypeinfos.push_back(tidecl->llvmValue);
            }

            const llvm::StructType* vtype = llvm::StructType::get(vtypes);
            llvm::Value* mem = new llvm::AllocaInst(vtype,"_argptr_storage",p->topallocapoint());
            for (unsigned i=0; i<vtype->getNumElements(); ++i)
                p->ir->CreateStore(vvalues[i], DtoGEPi(mem,0,i,"tmp"));

            //llvm::Constant* typeinfoparam = llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(llfnty->getParamType(j)));
            assert(Type::typeinfo->llvmInitZ);
            const llvm::Type* typeinfotype = llvm::PointerType::get(Type::typeinfo->llvmInitZ->getType());
            Logger::cout() << "typeinfo ptr type: " << *typeinfotype << '\n';
            const llvm::ArrayType* typeinfoarraytype = llvm::ArrayType::get(typeinfotype,vtype->getNumElements());
            llvm::Value* typeinfomem = new llvm::AllocaInst(typeinfoarraytype,"_arguments_storage",p->topallocapoint());
            for (unsigned i=0; i<vtype->getNumElements(); ++i) {
                llvm::Value* v = p->ir->CreateBitCast(vtypeinfos[i], typeinfotype, "tmp");
                p->ir->CreateStore(v, DtoGEPi(typeinfomem,0,i,"tmp"));
            }

            llvm::Value* typeinfoarrayparam = new llvm::AllocaInst(llfnty->getParamType(j)->getContainedType(0),"_arguments_array",p->topallocapoint());
            p->ir->CreateStore(DtoConstSize_t(vtype->getNumElements()), DtoGEPi(typeinfoarrayparam,0,0,"tmp"));
            llvm::Value* casttypeinfomem = p->ir->CreateBitCast(typeinfomem, llvm::PointerType::get(typeinfotype), "tmp");
            p->ir->CreateStore(casttypeinfomem, DtoGEPi(typeinfoarrayparam,0,1,"tmp"));

            llargs[j] = typeinfoarrayparam;;
            j++;
            llargs[j] = p->ir->CreateBitCast(mem, llvm::PointerType::get(llvm::Type::Int8Ty), "tmp");
            j++;
            llargs.resize(2);
        }
        else {
            Logger::println("doing normal arguments");
            for (int i=0; i<arguments->dim; i++,j++) {
                Argument* fnarg = Argument::getNth(tf->parameters, i);
                llargs[j] = DtoArgument(llfnty->getParamType(j), fnarg, (Expression*)arguments->data[i]);
            }
            Logger::println("%d params passed", n);
            for (int i=0; i<n; ++i) {
                Logger::cout() << *llargs[i] << '\n';
            }
        }
    }

    // void returns cannot not be named
    const char* varname = "";
    if (llfnty->getReturnType() != llvm::Type::VoidTy)
        varname = "tmp";

    Logger::cout() << "Calling: " << *funcval->getType() << '\n';

    // call the function
    llvm::CallInst* call = new llvm::CallInst(funcval, llargs.begin(), llargs.end(), varname, p->scopebb());
    if (retinptr)
        e->mem = llargs[0];
    else
        e->val = call;

    // set calling convention
    if (fn->funcdecl) {
        int li = fn->funcdecl->llvmInternal;
        if (li != LLVMintrinsic && li != LLVMva_start && li != LLVMva_intrinsic) {
            call->setCallingConv(DtoCallingConv(dlink));
        }
    }
    else if (delegateCall) {
        call->setCallingConv(DtoCallingConv(dlink));
    }
    else if (fn->callconv != (unsigned)-1) {
        call->setCallingConv(fn->callconv);
    }

    delete fn;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* CastExp::toElem(IRState* p)
{
    Logger::print("CastExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* u = e1->toElem(p);
    const llvm::Type* tolltype = DtoType(to);
    Type* fromtype = DtoDType(e1->type);
    Type* totype = DtoDType(to);
    int lsz = fromtype->size();
    int rsz = totype->size();

    // this makes sure the strange lvalue casts don't screw things up
    e->mem = u->mem;

    if (fromtype->isintegral()) {
        if (totype->isintegral()) {
            if (lsz < rsz) {
                Logger::cout() << *tolltype << '\n';
                if (fromtype->isunsigned() || fromtype->ty == Tbool) {
                    e->val = new llvm::ZExtInst(u->getValue(), tolltype, "tmp", p->scopebb());
                } else {
                    e->val = new llvm::SExtInst(u->getValue(), tolltype, "tmp", p->scopebb());
                }
            }
            else if (lsz > rsz) {
                e->val = new llvm::TruncInst(u->getValue(), tolltype, "tmp", p->scopebb());
            }
            else {
                e->val = new llvm::BitCastInst(u->getValue(), tolltype, "tmp", p->scopebb());
            }
        }
        else if (totype->isfloating()) {
            if (fromtype->isunsigned()) {
                e->val = new llvm::UIToFPInst(u->getValue(), tolltype, "tmp", p->scopebb());
            }
            else {
                e->val = new llvm::SIToFPInst(u->getValue(), tolltype, "tmp", p->scopebb());
            }
        }
        else if (totype->ty == Tpointer) {
            e->val = p->ir->CreateIntToPtr(u->getValue(), tolltype, "tmp");
        }
        else {
            assert(0);
        }
        //e->storeVal = u->storeVal ? u->storeVal : u->val;
        e->type = elem::VAL;
    }
    else if (fromtype->isfloating()) {
        if (totype->isfloating()) {
            if ((fromtype->ty == Tfloat80 || fromtype->ty == Tfloat64) && (totype->ty == Tfloat80 || totype->ty == Tfloat64)) {
                e->val = u->getValue();
            }
            else if (lsz < rsz) {
                e->val = new llvm::FPExtInst(u->getValue(), tolltype, "tmp", p->scopebb());
            }
            else if (lsz > rsz) {
                e->val = new llvm::FPTruncInst(u->getValue(), tolltype, "tmp", p->scopebb());
            }
            else {
                assert(0);
            }
        }
        else if (totype->isintegral()) {
            if (totype->isunsigned()) {
                e->val = new llvm::FPToUIInst(u->getValue(), tolltype, "tmp", p->scopebb());
            }
            else {
                e->val = new llvm::FPToSIInst(u->getValue(), tolltype, "tmp", p->scopebb());
            }
        }
        else {
            assert(0);
        }
        e->type = elem::VAL;
    }
    else if (fromtype->ty == Tclass) {
        //assert(to->ty == Tclass);
        e->val = new llvm::BitCastInst(u->getValue(), tolltype, "tmp", p->scopebb());
        e->type = elem::VAL;
    }
    else if (fromtype->ty == Tarray || fromtype->ty == Tsarray) {
        Logger::cout() << "from array or sarray" << '\n';
        if (totype->ty == Tpointer) {
            Logger::cout() << "to pointer" << '\n';
            assert(fromtype->next == totype->next || totype->next->ty == Tvoid);
            llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
            llvm::Value* ptr = DtoGEP(u->getValue(),zero,one,"tmp",p->scopebb());
            e->val = new llvm::LoadInst(ptr, "tmp", p->scopebb());
            if (fromtype->next != totype->next)
                e->val = p->ir->CreateBitCast(e->val, llvm::PointerType::get(llvm::Type::Int8Ty), "tmp");
            e->type = elem::VAL;
        }
        else if (totype->ty == Tarray) {
            Logger::cout() << "to array" << '\n';
            const llvm::Type* ptrty = DtoType(totype->next);
            if (ptrty == llvm::Type::VoidTy)
                ptrty = llvm::Type::Int8Ty;
            ptrty = llvm::PointerType::get(ptrty);

            const llvm::Type* ety = DtoType(fromtype->next);
            if (ety == llvm::Type::VoidTy)
                ety = llvm::Type::Int8Ty;

            if (u->type == elem::SLICE) {
                e->mem = new llvm::BitCastInst(u->mem, ptrty, "tmp", p->scopebb());
                if (fromtype->next->size() == totype->next->size())
                    e->arg = u->arg;
                else
                    e->arg = DtoArrayCastLength(u->arg, ety, ptrty->getContainedType(0));
            }
            else {
                llvm::Value* uval = u->getValue();
                if (fromtype->ty == Tsarray) {
                    Logger::cout() << "uvalTy = " << *uval->getType() << '\n';
                    assert(llvm::isa<llvm::PointerType>(uval->getType()));
                    const llvm::ArrayType* arrty = llvm::cast<llvm::ArrayType>(uval->getType()->getContainedType(0));
                    e->arg = llvm::ConstantInt::get(DtoSize_t(), arrty->getNumElements(), false);
                    e->arg = DtoArrayCastLength(e->arg, ety, ptrty->getContainedType(0));
                    e->mem = new llvm::BitCastInst(uval, ptrty, "tmp", p->scopebb());
                }
                else {
                    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
                    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
                    e->arg = DtoGEP(uval,zero,zero,"tmp",p->scopebb());
                    e->arg = new llvm::LoadInst(e->arg, "tmp", p->scopebb());
                    e->arg = DtoArrayCastLength(e->arg, ety, ptrty->getContainedType(0));

                    e->mem = DtoGEP(uval,zero,one,"tmp",p->scopebb());
                    e->mem = new llvm::LoadInst(e->mem, "tmp", p->scopebb());
                    //Logger::cout() << *e->mem->getType() << '|' << *ptrty << '\n';
                    e->mem = new llvm::BitCastInst(e->mem, ptrty, "tmp", p->scopebb());
                }
            }
            e->type = elem::SLICE;
        }
        else if (totype->ty == Tsarray) {
            Logger::cout() << "to sarray" << '\n';
            assert(0);
        }
        else {
            assert(0);
        }
    }
    else if (fromtype->ty == Tpointer) {
        if (totype->ty == Tpointer || totype->ty == Tclass) {
            llvm::Value* src = u->getValue();
            Logger::cout() << *src << '|' << *tolltype << '\n';
            e->val = new llvm::BitCastInst(src, tolltype, "tmp", p->scopebb());
        }
        else if (totype->isintegral()) {
            e->val = new llvm::PtrToIntInst(u->getValue(), tolltype, "tmp", p->scopebb());
        }
        else
        assert(0);
        e->type = elem::VAL;
    }
    else {
        assert(0);
    }
    delete u;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* SymOffExp::toElem(IRState* p)
{
    Logger::print("SymOffExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = 0;
    if (VarDeclaration* vd = var->isVarDeclaration())
    {
        Logger::println("VarDeclaration");
        if (!vd->llvmTouched && vd->isDataseg())
            vd->toObjFile();

        if (vd->isTypedefDeclaration()) {
            e->istypeinfo = true;
        }

        assert(vd->llvmValue);
        Type* t = DtoDType(type);
        Type* tnext = DtoDType(t->next);
        Type* vdtype = DtoDType(vd->type);

        llvm::Value* llvalue = vd->nestedref ? DtoNestedVariable(vd) : vd->llvmValue;

        if (vdtype->ty == Tstruct && !(t->ty == Tpointer && t->next == vdtype)) {
            Logger::println("struct");
            TypeStruct* vdt = (TypeStruct*)vdtype;
            assert(vdt->sym);
            e = new elem;
            const llvm::Type* llt = DtoType(t);
            if (offset == 0) {
                e->mem = p->ir->CreateBitCast(llvalue, llt, "tmp");
            }
            else {
                std::vector<unsigned> dst;
                e->mem = DtoIndexStruct(llvalue,vdt->sym, tnext, offset, dst);
            }
            e->type = elem::VAL;
            e->field = true;
        }
        else if (vdtype->ty == Tsarray) {
            Logger::println("sarray");
            e = new elem;
            assert(llvalue);
            e->arg = llvalue;
            e->type = elem::VAL;

            const llvm::Type* llt = DtoType(t);
            llvm::Value* off = 0;
            if (offset != 0) {
                Logger::println("offset = %d\n", offset);
            }
            if (llvalue->getType() != llt) {
                e->mem = p->ir->CreateBitCast(llvalue, llt, "tmp");
                if (offset != 0)
                    e->mem = DtoGEPi(e->mem, offset, "tmp");
            }
            else {
                assert(offset == 0);
                e->mem = DtoGEPi(llvalue,0,0,"tmp");
            }
        }
        else if (offset == 0) {
            Logger::println("normal symoff");
            e = new elem;
            e->type = elem::VAL;
            assert(llvalue);
            e->mem = llvalue;

            const llvm::Type* llt = DtoType(t);
            if (llvalue->getType() != llt) {
                e->mem = p->ir->CreateBitCast(e->mem, llt, "tmp");
            }
        }
        else {
            assert(0);
        }
    }
    else if (FuncDeclaration* fd = var->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        e = new elem;
        if (fd->llvmValue == 0)
            fd->toObjFile();
        e->val = fd->llvmValue;
        e->type = elem::FUNC;
    }
    assert(e != 0);
    assert(e->type != elem::NONE);
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* PtrExp::toElem(IRState* p)
{
    Logger::print("PtrExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* a = e1->toElem(p);

    if (a->mem)
        Logger::cout() << "mem: " << *a->mem << '\n';
    if (a->val)
        Logger::cout() << "val: " << *a->val << '\n';

    if (a->field)
        e->mem = a->mem;
    else
        e->mem = a->getValue();
    e->type = elem::VAR;

    delete a;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* DotVarExp::toElem(IRState* p)
{
    Logger::print("DotVarExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;

    elem* l = e1->toElem(p);

    Type* t = DtoDType(type);
    Type* e1type = DtoDType(e1->type);

    Logger::print("e1->type=%s\n", e1type->toChars());

    if (VarDeclaration* vd = var->isVarDeclaration()) {
        llvm::Value* arrptr;
        if (e1type->ty == Tpointer) {
            assert(e1type->next->ty == Tstruct);
            TypeStruct* ts = (TypeStruct*)e1type->next;
            Logger::println("Struct member offset:%d", vd->offset);
            llvm::Value* src = l->val ? l->val : l->mem;
            std::vector<unsigned> vdoffsets;
            arrptr = DtoIndexStruct(src, ts->sym, vd->type, vd->offset, vdoffsets);
        }
        else if (e1->type->ty == Tclass) {
            TypeClass* tc = (TypeClass*)e1type;
            Logger::println("Class member offset: %d", vd->offset);
            std::vector<unsigned> vdoffsets(1,0);
            tc->sym->offsetToIndex(vd->type, vd->offset, vdoffsets);
            llvm::Value* src = l->getValue();
            Logger::cout() << "src: " << *src << '\n';
            arrptr = DtoGEP(src,vdoffsets,"tmp",p->scopebb());
        }
        else
            assert(0);
        e->mem = arrptr;
        Logger::cout() << "mem: " << *e->mem << '\n';
        e->type = elem::VAR;
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        if (fdecl->llvmValue == 0)
        {
            fdecl->toObjFile();
        }

        llvm::Value* funcval = fdecl->llvmValue;
        e->arg = l->getValue();

        // virtual call
        if (fdecl->isVirtual()) {
            assert(fdecl->vtblIndex > 0);
            assert(e1type->ty == Tclass);

            llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            llvm::Value* vtblidx = llvm::ConstantInt::get(llvm::Type::Int32Ty, (size_t)fdecl->vtblIndex, false);
            funcval = DtoGEP(e->arg, zero, zero, "tmp", p->scopebb());
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
            funcval = DtoGEP(funcval, zero, vtblidx, toChars(), p->scopebb());
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
            assert(funcval->getType() == fdecl->llvmValue->getType());
            e->callconv = DtoCallingConv(fdecl->linkage);
        }
        e->val = funcval;
        e->type = elem::FUNC;
        e->funcdecl = fdecl;
    }
    else {
        printf("unknown: %s\n", var->toChars());
        assert(0);
    }

    delete l;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* ThisExp::toElem(IRState* p)
{
    Logger::print("ThisExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;

    if (VarDeclaration* vd = var->isVarDeclaration()) {
        llvm::Value* v = p->func().decl->llvmThisVar;
        if (llvm::isa<llvm::AllocaInst>(v))
            v = new llvm::LoadInst(v, "tmp", p->scopebb());
        e->mem = v;
        e->type = elem::VAL;
        e->isthis = true;
    }
    else {
        assert(0);
    }

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AddrExp::toElem(IRState* p)
{
    Logger::print("AddrExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = e1->toElem(p);
    e->field = true;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* StructLiteralExp::toElem(IRState* p)
{
    Logger::print("StructLiteralExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;

    llvm::Value* sptr;
    const llvm::Type* llt = DtoType(type);

    // temporary struct literal
    if (!p->topexp() || p->topexp()->e2 != this)
    {
        sptr = new llvm::AllocaInst(llt,"tmpstructliteral",p->topallocapoint());
        e->mem = sptr;
        e->type = elem::VAR;
    }
    // already has memory
    else
    {
        assert(p->topexp()->e2 == this);
        sptr = p->topexp()->v;
    }

    // num elements in literal
    unsigned n = elements->dim;

    // unions might have different types for each literal
    if (sd->llvmHasUnions) {
        // build the type of the literal
        std::vector<const llvm::Type*> tys;
        for (unsigned i=0; i<n; ++i) {
            Expression* vx = (Expression*)elements->data[i];
            if (!vx) continue;
            tys.push_back(DtoType(vx->type));
        }
        const llvm::StructType* t = llvm::StructType::get(tys);
        if (t != llt) {
            if (gTargetData->getTypeSize(t) != gTargetData->getTypeSize(llt)) { 
                Logger::cout() << "got size " << gTargetData->getTypeSize(t) << ", expected " << gTargetData->getTypeSize(llt) << '\n';
                assert(0 && "type size mismatch");
            }
            sptr = p->ir->CreateBitCast(sptr, llvm::PointerType::get(t), "tmp");
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

        p->exps.push_back(IRExp(NULL,vx,arrptr));
        elem* ve = vx->toElem(p);
        p->exps.pop_back();

        if (!ve->inplace) {
            llvm::Value* val = ve->getValue();
            Logger::cout() << *val << " | " << *arrptr << '\n';

            Type* vxtype = DtoDType(vx->type);
            DtoAssign(vxtype, arrptr, val);
        }
        delete ve;

        j++;
    }

    e->inplace = true;

    return e;
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
    const llvm::StructType* st = llvm::cast<llvm::StructType>(t);
    return llvm::ConstantStruct::get(st,vals);
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* IndexExp::toElem(IRState* p)
{
    Logger::print("IndexExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;

    elem* l = e1->toElem(p);

    Type* e1type = DtoDType(e1->type);

    p->arrays.push_back(l->mem); // if $ is used it must be an array so this is fine.
    elem* r = e2->toElem(p);
    p->arrays.pop_back();

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    llvm::Value* arrptr = 0;
    if (e1type->ty == Tpointer) {
        arrptr = new llvm::GetElementPtrInst(l->getValue(),r->getValue(),"tmp",p->scopebb());
    }
    else if (e1type->ty == Tsarray) {
        arrptr = DtoGEP(l->getValue(), zero, r->getValue(),"tmp",p->scopebb());
    }
    else if (e1type->ty == Tarray) {
        arrptr = DtoGEP(l->mem,zero,one,"tmp",p->scopebb());
        arrptr = new llvm::LoadInst(arrptr,"tmp",p->scopebb());
        arrptr = new llvm::GetElementPtrInst(arrptr,r->getValue(),"tmp",p->scopebb());
    }
    assert(arrptr);

    e->mem = arrptr;
    e->type = elem::VAR;

    delete l;
    delete r;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* SliceExp::toElem(IRState* p)
{
    Logger::print("SliceExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* t = DtoDType(type);
    assert(t->ty == Tarray);

    elem* v = e1->toElem(p);
    Type* e1type = DtoDType(e1->type);

    elem* e = new elem;
    assert(v->mem);
    e->type = elem::SLICE;

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    // partial slice
    if (lwr)
    {
        assert(upr);
        p->arrays.push_back(v->mem);
        elem* lo = lwr->toElem(p);

        bool lwr_is_zero = false;
        if (lo->type == elem::CONST)
        {
            assert(lo->val);
            assert(llvm::isa<llvm::ConstantInt>(lo->val));

            if (e1type->ty == Tpointer) {
                e->mem = v->getValue();
            }
            else if (e1type->ty == Tarray) {
                llvm::Value* tmp = DtoGEP(v->mem,zero,one,"tmp",p->scopebb());
                e->mem = new llvm::LoadInst(tmp,"tmp",p->scopebb());
            }
            else if (e1type->ty == Tsarray) {
                e->mem = DtoGEP(v->mem,zero,zero,"tmp",p->scopebb());
            }
            else
            assert(e->mem);

            llvm::ConstantInt* c = llvm::cast<llvm::ConstantInt>(lo->val);
            if (!(lwr_is_zero = c->isZero())) {
                e->mem = new llvm::GetElementPtrInst(e->mem,lo->val,"tmp",p->scopebb());
            }
        }
        else
        {
            if (e1type->ty == Tarray) {
                llvm::Value* tmp = DtoGEP(v->mem,zero,one,"tmp",p->scopebb());
                tmp = new llvm::LoadInst(tmp,"tmp",p->scopebb());
                e->mem = new llvm::GetElementPtrInst(tmp,lo->getValue(),"tmp",p->scopebb());
            }
            else if (e1type->ty == Tsarray) {
                e->mem = DtoGEP(v->mem,zero,lo->getValue(),"tmp",p->scopebb());
            }
            else if (e1type->ty == Tpointer) {
                e->mem = new llvm::GetElementPtrInst(v->getValue(),lo->getValue(),"tmp",p->scopebb());
            }
            else {
                Logger::println("type = %s", e1type->toChars());
                assert(0);
            }
        }

        elem* up = upr->toElem(p);
        p->arrays.pop_back();

        if (up->type == elem::CONST)
        {
            assert(up->val);
            assert(llvm::isa<llvm::ConstantInt>(up->val));
            if (lwr_is_zero) {
                e->arg = up->val;
            }
            else {
                if (lo->type == elem::CONST) {
                    llvm::Constant* clo = llvm::cast<llvm::Constant>(lo->val);
                    llvm::Constant* cup = llvm::cast<llvm::Constant>(up->val);
                    e->arg = llvm::ConstantExpr::getSub(cup, clo);
                }
                else {
                    e->arg = llvm::BinaryOperator::createSub(up->val, lo->getValue(), "tmp", p->scopebb());
                }
            }
        }
        else
        {
            if (lwr_is_zero) {
                e->arg = up->getValue();
            }
            else {
                e->arg = llvm::BinaryOperator::createSub(up->getValue(), lo->getValue(), "tmp", p->scopebb());
            }
        }

        delete lo;
        delete up;

        /*
        llvm::Value* tmpmem = new llvm::AllocaInst(DtoType(t),"tmp",p->topallocapoint());
        llvm::Value* ptr = DtoGEPi(tmpmem,0,0,"tmp");
        p->ir->CreateStore(e->arg, ptr);
        ptr = DtoGEPi(tmpmem,0,1,"tmp");
        p->ir->CreateStore(e->mem, ptr);
        e->arg = NULL;
        e->mem = tmpmem;
        */
    }
    // full slice
    else
    {
        e->mem = v->mem;
    }

    delete v;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* CmpExp::toElem(IRState* p)
{
    Logger::print("CmpExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;

    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* t = DtoDType(e1->type);
    Type* e2t = DtoDType(e2->type);
    assert(t == e2t);

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
            e->val = llvm::ConstantInt::getTrue();
            break;
        case TOKunord:
            skip = true;
            e->val = llvm::ConstantInt::getFalse();
            break;

        default:
            assert(0);
        }
        if (!skip)
        {
            e->val = new llvm::ICmpInst(cmpop, l->getValue(), r->getValue(), "tmp", p->scopebb());
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
        e->val = new llvm::FCmpInst(cmpop, l->getValue(), r->getValue(), "tmp", p->scopebb());
    }
    else
    {
        assert(0 && "Unsupported CmpExp type");
    }

    delete l;
    delete r;

    e->type = elem::VAL;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* EqualExp::toElem(IRState* p)
{
    Logger::print("EqualExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;

    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    Type* t = DtoDType(e1->type);
    Type* e2t = DtoDType(e2->type);
    assert(t == e2t);

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
        e->val = new llvm::ICmpInst(cmpop, l->getValue(), r->getValue(), "tmp", p->scopebb());
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
        e->val = new llvm::FCmpInst(cmpop, l->getValue(), r->getValue(), "tmp", p->scopebb());
    }
    else if (t->ty == Tsarray)
    {
        Logger::println("static array");
        e->val = DtoStaticArrayCompare(op,l->mem,r->mem);
    }
    else if (t->ty == Tarray)
    {
        Logger::println("dynamic array");
        e->val = DtoDynArrayCompare(op,l->mem,r->mem);
    }
    else if (t->ty == Tdelegate)
    {
        Logger::println("delegate");
        e->val = DtoCompareDelegate(op,l->mem,r->mem);
    }
    else
    {
        assert(0 && "Unsupported EqualExp type");
    }

    delete l;
    delete r;

    e->type = elem::VAL;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* PostExp::toElem(IRState* p)
{
    Logger::print("PostExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    elem* e = new elem;
    e->mem = l->mem;
    e->val = l->getValue();
    e->type = elem::VAL;

    llvm::Value* val = e->val;
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
        post = new llvm::GetElementPtrInst(val, whichone, "tmp", p->scopebb());
    }
    else if (e1type->isfloating())
    {
        assert(e2type->isfloating());
        llvm::Value* one = llvm::ConstantFP::get(val->getType(), 1.0f);
        if (op == TOKplusplus) {
            post = llvm::BinaryOperator::createAdd(val,one,"tmp",p->scopebb());
        }
        else if (op == TOKminusminus) {
            post = llvm::BinaryOperator::createSub(val,one,"tmp",p->scopebb());
        }
    }
    else
    assert(post);

    if (l->mem == 0)
        DtoGiveArgumentStorage(l);
    new llvm::StoreInst(post,l->mem,p->scopebb());

    delete l;
    delete r;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* NewExp::toElem(IRState* p)
{
    Logger::print("NewExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(!thisexp);
    assert(!newargs);
    assert(newtype);
    assert(!allocator);

    elem* e = new elem;
    e->inplace = true;

    Type* ntype = DtoDType(newtype);

    const llvm::Type* t = DtoType(ntype);

    if (onstack) {
        assert(ntype->ty == Tclass);
        e->mem = new llvm::AllocaInst(t->getContainedType(0),"tmp",p->topallocapoint());
    }
    else {
        if (ntype->ty == Tclass) {
            e->mem = new llvm::MallocInst(t->getContainedType(0),"tmp",p->scopebb());
        }
        else if (ntype->ty == Tarray) {
            assert(arguments);
            if (arguments->dim == 1) {
                elem* sz = ((Expression*)arguments->data[0])->toElem(p);
                llvm::Value* dimval = sz->getValue();
                Type* nnt = DtoDType(ntype->next);
                if (nnt->ty == Tvoid)
                    nnt = Type::tint8;
                if (!p->topexp() || p->topexp()->e2 != this) {
                    const llvm::Type* restype = DtoType(type);
                    Logger::cout() << "restype = " << *restype << '\n';
                    e->mem = new llvm::AllocaInst(restype,"tmp",p->topallocapoint());
                    DtoNewDynArray(e->mem, dimval, nnt);
                    e->inplace = false;
                }
                else if (p->topexp() || p->topexp()->e2 != this) {
                    assert(p->topexp()->v);
                    e->mem = p->topexp()->v;
                    DtoNewDynArray(e->mem, dimval, nnt);
                }
                else
                assert(0);
                delete sz;
            }
            else {
                assert(0);
            }
        }
        else {
            e->mem = new llvm::MallocInst(t,"tmp",p->scopebb());
        }
    }

    if (ntype->ty == Tclass) {
        // first apply the static initializer
        assert(e->mem);
        DtoInitClass((TypeClass*)ntype, e->mem);

        // then call constructor
        if (arguments) {
            assert(member);
            assert(member->llvmValue);
            llvm::Function* fn = llvm::cast<llvm::Function>(member->llvmValue);
            TypeFunction* tf = (TypeFunction*)DtoDType(member->type);

            std::vector<llvm::Value*> ctorargs;
            ctorargs.push_back(e->mem);
            for (size_t i=0; i<arguments->dim; ++i)
            {
                Expression* ex = (Expression*)arguments->data[i];
                Argument* fnarg = Argument::getNth(tf->parameters, i);
                llvm::Value* a = DtoArgument(fn->getFunctionType()->getParamType(i+1), fnarg, ex);
                ctorargs.push_back(a);
            }
            e->mem = new llvm::CallInst(fn, ctorargs.begin(), ctorargs.end(), "tmp", p->scopebb());
        }
    }
    else if (ntype->ty == Tstruct) {
        TypeStruct* ts = (TypeStruct*)ntype;
        if (ts->isZeroInit()) {
            DtoStructZeroInit(e->mem);
        }
        else {
            DtoStructCopy(e->mem,ts->llvmInit);
        }
    }

    e->type = elem::VAR;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* DeleteExp::toElem(IRState* p)
{
    Logger::print("DeleteExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    //assert(e1->type->ty != Tclass);

    elem* v = e1->toElem(p);
    llvm::Value* val = v->getValue();
    llvm::Value* ldval = 0;

    const llvm::Type* t = val->getType();
    llvm::Constant* z = llvm::Constant::getNullValue(t);

    Type* e1type = DtoDType(e1->type);

    if (e1type->ty == Tpointer) {
        ldval = v->getValue();
        new llvm::FreeInst(ldval, p->scopebb());

        Logger::cout() << *z << '\n';
        Logger::cout() << *val << '\n';
        new llvm::StoreInst(z, v->mem, p->scopebb());
    }
    else if (e1type->ty == Tclass) {
        TypeClass* tc = (TypeClass*)e1type;
        DtoCallClassDtors(tc, val);

        if (v->vardecl && !v->vardecl->onstack) {
            new llvm::FreeInst(val, p->scopebb());
        }
        new llvm::StoreInst(z, v->mem, p->scopebb());
    }
    else if (e1type->ty == Tarray) {
        // must be on the heap (correct?)
        ldval = v->getValue();
        
        llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
        llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
        
        llvm::Value* ptr = DtoGEP(ldval,zero,one,"tmp",p->scopebb());
        ptr = new llvm::LoadInst(ptr,"tmp",p->scopebb());
        new llvm::FreeInst(ptr, p->scopebb());
        DtoNullArray(val);
    }
    else {
        assert(0);
    }

    delete v;

    // this expression produces no useful data
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* ArrayLengthExp::toElem(IRState* p)
{
    Logger::print("ArrayLengthExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;
    elem* u = e1->toElem(p);

    if (p->topexp() && p->topexp()->e1 == this)
    {
        e->mem = u->mem;
        e->type = elem::ARRAYLEN;
    }
    else
    {
        llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
        llvm::Value* ptr = DtoGEP(u->mem,zero,zero,"tmp",p->scopebb());
        e->val = new llvm::LoadInst(ptr, "tmp", p->scopebb());
        e->type = elem::VAL;
    }
    delete u;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AssertExp::toElem(IRState* p)
{
    Logger::print("AssertExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* u = e1->toElem(p);
    elem* m = msg ? msg->toElem(p) : NULL;

    llvm::Value* loca = llvm::ConstantInt::get(llvm::Type::Int32Ty, loc.linnum, false);
    DtoAssert(u->getValue(), loca, m ? m->val : NULL);

    delete m;
    delete u;

    return new elem;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* NotExp::toElem(IRState* p)
{
    Logger::print("NotExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;
    elem* u = e1->toElem(p);

    llvm::Value* b = DtoBoolean(u->getValue());

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int1Ty, 0, true);
    e->val = p->ir->CreateICmpEQ(b,zero);
    //e->val = new llvm::ICmpInst(llvm::ICmpInst::ICMP_EQ,b,zero,"tmp",p->scopebb());
    e->type = elem::VAL;

    delete u;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AndAndExp::toElem(IRState* p)
{
    Logger::print("AndAndExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // allocate a temporary for the final result. failed to come up with a better way :/
    llvm::Value* resval = 0;
    llvm::BasicBlock* entryblock = &p->topfunc()->front();
    resval = new llvm::AllocaInst(llvm::Type::Int1Ty,"andandtmp",p->topallocapoint());
    
    elem* e = new elem;
    elem* u = e1->toElem(p);

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* andand = new llvm::BasicBlock("andand", gIR->topfunc(), oldend);
    llvm::BasicBlock* andandend = new llvm::BasicBlock("andandend", gIR->topfunc(), oldend);
    
    llvm::Value* ubool = DtoBoolean(u->getValue());
    new llvm::StoreInst(ubool,resval,p->scopebb());
    new llvm::BranchInst(andand,andandend,ubool,p->scopebb());
    
    p->scope() = IRScope(andand, andandend);
    elem* v = e2->toElem(p);

    llvm::Value* vbool = DtoBoolean(v->getValue());
    llvm::Value* uandvbool = llvm::BinaryOperator::create(llvm::BinaryOperator::And, ubool, vbool,"tmp",p->scopebb());
    new llvm::StoreInst(uandvbool,resval,p->scopebb());
    new llvm::BranchInst(andandend,p->scopebb());

    delete u;
    delete v;

    p->scope() = IRScope(andandend, oldend);
    
    e->val = new llvm::LoadInst(resval,"tmp",p->scopebb());
    e->type = elem::VAL;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* OrOrExp::toElem(IRState* p)
{
    Logger::print("OrOrExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    // allocate a temporary for the final result. failed to come up with a better way :/
    llvm::Value* resval = 0;
    llvm::BasicBlock* entryblock = &p->topfunc()->front();
    resval = new llvm::AllocaInst(llvm::Type::Int1Ty,"orortmp",p->topallocapoint());
    
    elem* e = new elem;
    elem* u = e1->toElem(p);

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* oror = new llvm::BasicBlock("oror", gIR->topfunc(), oldend);
    llvm::BasicBlock* ororend = new llvm::BasicBlock("ororend", gIR->topfunc(), oldend);
    
    llvm::Value* ubool = DtoBoolean(u->getValue());
    new llvm::StoreInst(ubool,resval,p->scopebb());
    new llvm::BranchInst(ororend,oror,ubool,p->scopebb());
    
    p->scope() = IRScope(oror, ororend);
    elem* v = e2->toElem(p);

    llvm::Value* vbool = DtoBoolean(v->getValue());
    new llvm::StoreInst(vbool,resval,p->scopebb());
    new llvm::BranchInst(ororend,p->scopebb());

    delete u;
    delete v;

    p->scope() = IRScope(ororend, oldend);
    
    e->val = new llvm::LoadInst(resval,"tmp",p->scopebb());
    e->type = elem::VAL;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

#define BinBitExp(X,Y) \
elem* X##Exp::toElem(IRState* p) \
{ \
    Logger::print("%sExp::toElem: %s | %s\n", #X, toChars(), type->toChars()); \
    LOG_SCOPE; \
    elem* e = new elem; \
    elem* u = e1->toElem(p); \
    elem* v = e2->toElem(p); \
    e->val = llvm::BinaryOperator::create(llvm::Instruction::Y, u->getValue(), v->getValue(), "tmp", p->scopebb()); \
    e->type = elem::VAL; \
    delete u; \
    delete v; \
    return e; \
} \
\
elem* X##AssignExp::toElem(IRState* p) \
{ \
    Logger::print("%sAssignExp::toElem: %s | %s\n", #X, toChars(), type->toChars()); \
    LOG_SCOPE; \
    elem* u = e1->toElem(p); \
    elem* v = e2->toElem(p); \
    llvm::Value* uval = u->getValue(); \
    assert(uval); \
    llvm::Value* vval = v->getValue(); \
    assert(vval); \
    llvm::Value* tmp = llvm::BinaryOperator::create(llvm::Instruction::Y, uval, vval, "tmp", p->scopebb()); \
    if (u->mem == 0) \
        DtoGiveArgumentStorage(u); \
    Logger::cout() << *tmp << '|' << *u->mem << '\n'; \
    new llvm::StoreInst(DtoPointedType(u->mem, tmp), u->mem, p->scopebb()); \
    delete u; \
    delete v; \
    elem* e = new elem; \
    e->mem = u->mem; \
    e->type = elem::VAR; \
    return e; \
}

BinBitExp(And,And);
BinBitExp(Or,Or);
BinBitExp(Xor,Xor);
BinBitExp(Shl,Shl);
BinBitExp(Shr,AShr);
BinBitExp(Ushr,LShr);

//////////////////////////////////////////////////////////////////////////////////////////

elem* HaltExp::toElem(IRState* p)
{
    Logger::print("HaltExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    llvm::Value* loca = llvm::ConstantInt::get(llvm::Type::Int32Ty, loc.linnum, false);
    DtoAssert(llvm::ConstantInt::getFalse(), loca, NULL);

    new llvm::UnreachableInst(p->scopebb());
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* DelegateExp::toElem(IRState* p)
{
    Logger::print("DelegateExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;
    elem* u = e1->toElem(p);

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    const llvm::Type* int8ptrty = llvm::PointerType::get(llvm::Type::Int8Ty);

    assert(p->topexp() && p->topexp()->e2 == this && p->topexp()->v);
    llvm::Value* lval = p->topexp()->v;

    llvm::Value* context = DtoGEP(lval,zero,zero,"tmp",p->scopebb());
    llvm::Value* castcontext = new llvm::BitCastInst(u->getValue(),int8ptrty,"tmp",p->scopebb());
    new llvm::StoreInst(castcontext, context, p->scopebb());

    llvm::Value* fptr = DtoGEP(lval,zero,one,"tmp",p->scopebb());

    assert(func->llvmValue);
    llvm::Value* castfptr = new llvm::BitCastInst(func->llvmValue,fptr->getType()->getContainedType(0),"tmp",p->scopebb());
    new llvm::StoreInst(castfptr, fptr, p->scopebb());

    e->inplace = true;

    delete u;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* IdentityExp::toElem(IRState* p)
{
    Logger::print("IdentityExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* u = e1->toElem(p);
    elem* v = e2->toElem(p);

    elem* e = new elem;

    llvm::Value* l = u->field ? u->mem : u->getValue();
    llvm::Value* r = v->field ? v->mem : v->getValue();

    Type* t1 = DtoDType(e1->type);

    if (t1->ty == Tarray) {
        if (v->type == elem::NUL) {
            r = NULL;
        }
        else {
            assert(l->getType() == r->getType());
        }
        e->val = DtoDynArrayIs(op,l,r);
    }
    else {
        llvm::ICmpInst::Predicate pred = (op == TOKidentity) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
        if (t1->ty == Tpointer && v->type == elem::NUL && l->getType() != r->getType()) {
            r = llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(l->getType()));
        }
        Logger::cout() << "l = " << *l << " r = " << *r << '\n';
        e->val = new llvm::ICmpInst(pred, l, r, "tmp", p->scopebb());
    }
    e->type = elem::VAL;

    delete u;
    delete v;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* CommaExp::toElem(IRState* p)
{
    Logger::print("CommaExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* u = e1->toElem(p);
    elem* v = e2->toElem(p);
    delete u;
    return v;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* CondExp::toElem(IRState* p)
{
    Logger::print("CondExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* dtype = DtoDType(type);
    const llvm::Type* resty = DtoType(dtype);

    // allocate a temporary for the final result. failed to come up with a better way :/
    llvm::BasicBlock* entryblock = &p->topfunc()->front();
    llvm::Value* resval = new llvm::AllocaInst(resty,"condtmp",p->topallocapoint());

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* condtrue = new llvm::BasicBlock("condtrue", gIR->topfunc(), oldend);
    llvm::BasicBlock* condfalse = new llvm::BasicBlock("condfalse", gIR->topfunc(), oldend);
    llvm::BasicBlock* condend = new llvm::BasicBlock("condend", gIR->topfunc(), oldend);

    elem* c = econd->toElem(p);
    llvm::Value* cond_val = DtoBoolean(c->getValue());
    delete c;
    new llvm::BranchInst(condtrue,condfalse,cond_val,p->scopebb());

    p->scope() = IRScope(condtrue, condfalse);
    elem* u = e1->toElem(p);
    DtoAssign(dtype, resval, u->getValue());
    new llvm::BranchInst(condend,p->scopebb());
    delete u;

    p->scope() = IRScope(condfalse, condend);
    elem* v = e2->toElem(p);
    DtoAssign(dtype, resval, v->getValue());
    new llvm::BranchInst(condend,p->scopebb());
    delete v;

    p->scope() = IRScope(condend, oldend);

    elem* e = new elem;
    e->mem = resval;
    e->type = elem::VAR;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* ComExp::toElem(IRState* p)
{
    Logger::print("ComExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;
    elem* u = e1->toElem(p);

    llvm::Value* value = u->getValue();
    llvm::Value* minusone = llvm::ConstantInt::get(value->getType(), -1, true);
    e->val = llvm::BinaryOperator::create(llvm::Instruction::Xor, value, minusone, "tmp", p->scopebb());

    delete u;

    e->type = elem::VAL;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* NegExp::toElem(IRState* p)
{
    Logger::print("NegExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* l = e1->toElem(p);
    llvm::Value* val = l->getValue();
    delete l;

    Type* t = DtoDType(type);

    llvm::Value* zero = 0;
    if (t->isintegral())
        zero = llvm::ConstantInt::get(val->getType(), 0, true);
    else if (t->isfloating()) {
        if (t->ty == Tfloat32)
            zero = llvm::ConstantFP::get(val->getType(), float(0));
        else if (t->ty == Tfloat64 || t->ty == Tfloat80)
            zero = llvm::ConstantFP::get(val->getType(), double(0));
        else
        assert(0);
    }
    else
        assert(0);

    e->val = llvm::BinaryOperator::createSub(zero,val,"tmp",p->scopebb());
    e->type = elem::VAL;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* CatExp::toElem(IRState* p)
{
    Logger::print("CatExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    Type* t = DtoDType(type);

    bool inplace = false;
    llvm::Value* dst = 0;
    IRExp* ex = p->topexp();
    if (ex && ex->e2 == this) {
        assert(ex->v);
        dst = ex->v;
        inplace = true;
    }
    else {
        assert(t->ty == Tarray);
        const llvm::Type* arrty = DtoType(t);
        dst = new llvm::AllocaInst(arrty, "tmpmem", p->topallocapoint());
    }

    DtoCatArrays(dst,e1,e2);

    elem* e = new elem;
    e->mem = dst;
    e->type = elem::VAR;
    e->inplace = inplace;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* CatAssignExp::toElem(IRState* p)
{
    Logger::print("CatAssignExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* l = e1->toElem(p);
    assert(l->mem);

    Type* e1type = DtoDType(e1->type);
    Type* elemtype = DtoDType(e1type->next);
    Type* e2type = DtoDType(e2->type);

    if (e2type == elemtype) {
        DtoCatAssignElement(l->mem,e2);
    }
    else
        assert(0 && "only one element at a time right now");

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* ArrayLiteralExp::toElem(IRState* p)
{
    Logger::print("ArrayLiteralExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    const llvm::Type* t = DtoType(type);
    Logger::cout() << "array literal has llvm type: " << *t << '\n';

    llvm::Value* mem = 0;
    if (!p->topexp() || p->topexp()->e2 != this) {
        assert(DtoDType(type)->ty == Tsarray);
        mem = new llvm::AllocaInst(t,"tmparrayliteral",p->topallocapoint());
    }
    else if (p->topexp()->e2 == this) {
        mem = p->topexp()->v;
        assert(mem);
        if (!llvm::isa<llvm::PointerType>(mem->getType()) ||
            !llvm::isa<llvm::ArrayType>(mem->getType()->getContainedType(0)))
        {
            error("TODO array literals can currently only be used to initialise static arrays");
            fatal();
        }
    }
    else
    assert(0);

    for (unsigned i=0; i<elements->dim; ++i)
    {
        Expression* expr = (Expression*)elements->data[i];
        llvm::Value* elemAddr = DtoGEPi(mem,0,i,"tmp",p->scopebb());
        elem* e = expr->toElem(p);
        new llvm::StoreInst(e->getValue(), elemAddr, p->scopebb());
    }

    elem* e = new elem;
    e->mem = mem;
    e->type = elem::VAL;
    e->inplace = true;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Constant* ArrayLiteralExp::toConstElem(IRState* p)
{
    Logger::print("ArrayLiteralExp::toConstElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    const llvm::Type* t = DtoType(type);
    Logger::cout() << "array literal has llvm type: " << *t << '\n';
    assert(llvm::isa<llvm::ArrayType>(t));
    const llvm::ArrayType* arrtype = llvm::cast<llvm::ArrayType>(t);

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

elem* FuncExp::toElem(IRState* p)
{
    Logger::print("FuncExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    assert(fd);

    if (fd->isNested()) Logger::println("nested");
    Logger::println("kind = %s\n", fd->kind());

    fd->toObjFile();

    llvm::Value* lval = NULL;
    if (!p->topexp() || p->topexp()->e2 != this) {
        const llvm::Type* dgty = DtoType(type);
        Logger::cout() << "delegate without explicit storage:" << '\n' << *dgty << '\n';
        lval = new llvm::AllocaInst(dgty,"dgstorage",p->topallocapoint());
    }
    else if (p->topexp()->e2 == this) {
        lval = p->topexp()->v;
        assert(lval);
    }
    else
    assert(0);

    elem* e = new elem;

    llvm::Value* context = DtoGEPi(lval,0,0,"tmp",p->scopebb());
    const llvm::PointerType* pty = llvm::cast<llvm::PointerType>(context->getType()->getContainedType(0));
    llvm::Value* llvmNested = p->func().decl->llvmNested;
    if (llvmNested == NULL) {
        llvm::Value* nullcontext = llvm::ConstantPointerNull::get(pty);
        p->ir->CreateStore(nullcontext, context);
    }
    else {
        llvm::Value* nestedcontext = p->ir->CreateBitCast(llvmNested, pty, "tmp");
        p->ir->CreateStore(nestedcontext, context);
    }

    llvm::Value* fptr = DtoGEPi(lval,0,1,"tmp",p->scopebb());

    assert(fd->llvmValue);
    llvm::Value* castfptr = new llvm::BitCastInst(fd->llvmValue,fptr->getType()->getContainedType(0),"tmp",p->scopebb());
    new llvm::StoreInst(castfptr, fptr, p->scopebb());

    e->inplace = true;
    e->mem = lval;
    e->type = elem::VAR;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

#define STUB(x) elem *x::toElem(IRState * p) {error("Exp type "#x" not implemented: %s", toChars()); fatal(); return 0; }
//STUB(IdentityExp);
//STUB(CondExp);
//STUB(EqualExp);
STUB(InExp);
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
STUB(ComplexExp);
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
STUB(RemoveExp);
//STUB(ArrayLiteralExp);
STUB(AssocArrayLiteralExp);
//STUB(StructLiteralExp);

#define CONSTSTUB(x) llvm::Constant* x::toConstElem(IRState * p) {error("const Exp type "#x" not implemented: '%s' type: '%s'", toChars(), type->toChars()); assert(0); fatal(); return NULL; }
CONSTSTUB(Expression);
//CONSTSTUB(IntegerExp);
//CONSTSTUB(RealExp);
//CONSTSTUB(NullExp);
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
    assert(0);
}
Statement *AsmStatement::syntaxCopy()
{
    assert(0);
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
    assert(0);
    return FALSE;
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
