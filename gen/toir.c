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

#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/CallingConv.h"

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

        // handle const
        // TODO probably not correct
        bool isconst = (vd->storage_class & STCconst) != 0;

        // allocate storage on the stack
        Logger::println("vdtype = %s", vd->type->toChars());
        const llvm::Type* lltype = LLVM_DtoType(vd->type);
        llvm::AllocaInst* allocainst = new llvm::AllocaInst(lltype, vd->toChars(), p->topallocapoint());
        //allocainst->setAlignment(vd->type->alignsize()); // TODO
        vd->llvmValue = allocainst;
        // e->val = really needed??

        LLVM_DtoInitializer(vd->type, vd->init);
    }
    // struct declaration
    else if (StructDeclaration* s = declaration->isStructDeclaration())
    {
        Logger::println("StructDeclaration");
        s->toObjFile();
    }
    // unsupported declaration
    else
    {
        error("Only Var/Struct-Declaration is supported for DeclarationExp");
        fatal();
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
        Logger::println("VarDeclaration");

        if (TypeInfoDeclaration* tid = vd->isTypeInfoDeclaration())
        {
            Logger::println("TypeInfoDeclaration");
        }

        // this must be a dollar expression or some other magic value
        if (!vd->llvmValue)
        {
            // dollar
            if (!p->arrays.empty())
            {
                llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
                //llvm::Value* tmp = new llvm::GetElementPtrInst(p->arrays.back(),zero,zero,"tmp",p->scopebb());
                llvm::Value* tmp = LLVM_DtoGEP(p->arrays.back(),zero,zero,"tmp",p->scopebb());
                e->val = new llvm::LoadInst(tmp,"tmp",p->scopebb());
                e->type = elem::VAL;
            }
            // magic
            else
            {
                if (TypeInfoDeclaration* tid = vd->isTypeInfoDeclaration())
                {
                    tid->toObjFile();
                    e->mem = tid->llvmValue;
                    e->type = elem::VAR;
                }
                else
                assert(0 && "only magic supported is typeinfo");
            }
            return e;
        }

        // function parameter
        if (vd->storage_class & STCparameter) {
            Logger::println("function param");
            if (vd->storage_class & (STCref | STCout)) {
                e->mem = vd->llvmValue;
                e->type = elem::VAR;
            }
            else {
                if (vd->type->ty == Tstruct || vd->type->ty == Tdelegate || vd->type->ty == Tarray) {
                    e->mem = vd->llvmValue;
                    e->type = elem::VAR;
                }
                else {
                    e->val = vd->llvmValue;
                    e->type = elem::VAL;
                }
            }
        }
        else {
            e->mem = vd->llvmValue;
            //e->mem->setName(toChars());
            e->vardecl = vd;
            e->type = elem::VAR;
        }
    }
    else if (FuncDeclaration* fdecl = var->isFuncDeclaration())
    {
        Logger::println("FuncDeclaration");
        if (fdecl->llvmValue == 0) {
            fdecl->toObjFile();
        }
        e->val = fdecl->llvmValue;
        e->type = elem::FUNC;
        e->funcdecl = fdecl;
    }
    else if (SymbolDeclaration* sdecl = var->isSymbolDeclaration())
    {
        // this seems to be the static initialiser for structs
        Logger::print("Sym: type=%s\n", sdecl->type->toChars());
        assert(sdecl->type->ty == Tstruct);
        //assert(sdecl->llvmInitZ);
        //e->val = sdecl->llvmInitZ;
        TypeStruct* ts = (TypeStruct*)sdecl->type;
        e->mem = ts->llvmInit;
        assert(e->mem);
        e->type = elem::VAR;
    }
    else
    {
        assert(0 && "Unimplemented VarExp type");
    }

    assert(e->mem || e->val);
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* IntegerExp::toElem(IRState* p)
{
    Logger::print("IntegerExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    const llvm::Type* t = LLVM_DtoType(type);
    if (llvm::isa<llvm::PointerType>(t)) {
        llvm::Constant* i = llvm::ConstantInt::get(LLVM_DtoSize_t(),(uint64_t)value,false);
        e->val = llvm::ConstantExpr::getIntToPtr(i, t);
    }
    else if (llvm::isa<llvm::IntegerType>(t)) {
        e->val = llvm::ConstantInt::get(t,(uint64_t)value,!type->isunsigned());
    }
    else {
        assert(0);
    }
    e->type = elem::CONST;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* RealExp::toElem(IRState* p)
{
    Logger::print("RealExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;
    elem* e = new elem;
    const llvm::Type* fty = LLVM_DtoType(type);
    if (type->ty == Tfloat32)
        e->val = llvm::ConstantFP::get(fty,float(value));
    else if (type->ty == Tfloat64 || type->ty == Tfloat80)
        e->val = llvm::ConstantFP::get(fty,double(value));
    else
    assert(0);
    e->type = elem::CONST;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* NullExp::toElem(IRState* p)
{
    Logger::print("NullExp::toElem(type=%s): %s\n", type->toChars(),toChars());
    LOG_SCOPE;
    elem* e = new elem;
    const llvm::Type* t = LLVM_DtoType(type);
    if (llvm::isa<llvm::StructType>(t))
        t = llvm::PointerType::get(t);
    Logger::cout() << *t << '\n';
    e->val = llvm::Constant::getNullValue(t);
    assert(e->val);
    Logger::cout() << *e->val << '\n';
    e->type = elem::NUL;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* StringExp::toElem(IRState* p)
{
    Logger::print("StringExp::toElem: %s\n", toChars());
    LOG_SCOPE;

    assert(type->next->ty == Tchar && "Only char is supported");
    assert(sz == 1);

    const llvm::Type* ct = LLVM_DtoType(type->next);
    //printf("ct = %s\n", type->next->toChars());
    const llvm::ArrayType* at = llvm::ArrayType::get(ct,len+1);

    uint8_t* str = (uint8_t*)string;
    std::string cont((char*)str, len);

    llvm::Constant* _init = llvm::ConstantArray::get(cont,true);

    llvm::GlobalValue::LinkageTypes _linkage = llvm::GlobalValue::InternalLinkage;//WeakLinkage;
    llvm::GlobalVariable* gvar = new llvm::GlobalVariable(at,true,_linkage,_init,"stringliteral",gIR->module);

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* arrptr = LLVM_DtoGEP(gvar,zero,zero,"tmp",p->scopebb());

    elem* e = new elem;

    if (type->ty == Tarray) {
        llvm::Constant* clen = llvm::ConstantInt::get(LLVM_DtoSize_t(),len,false);
        if (p->lvals.empty()) {
            e->type = elem::SLICE;
            e->arg = clen;
            e->mem = arrptr;
            return e;
        }
        else {
            llvm::Value* arr = p->toplval();
            LLVM_DtoSetArray(arr, clen, arrptr);
        }
    }
    else if (type->ty == Tpointer) {
        e->mem = arrptr;
    }
    else {
        assert(0);
    }

    e->inplace = true;
    e->type = elem::VAL;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AssignExp::toElem(IRState* p)
{
    Logger::print("AssignExp::toElem: %s | %s = %s\n", toChars(), e1->type->toChars(), e2->type->toChars());
    LOG_SCOPE;

    assert(e1 && e2);
    p->inLvalue = true;
        elem* l = e1->toElem(p);
    p->inLvalue = false;

    p->lvals.push_back(l->mem);
        elem* r = e2->toElem(p);
    p->lvals.pop_back();

    assert(l->mem);
    //e->val = l->store(r->getValue());

    TY e1ty = e1->type->ty;
    TY e2ty = e2->type->ty;

    elem* e = new elem;

    // struct
    if (e1ty == Tstruct) {
        // struct + struct
        if (e2ty == Tstruct) {
            // struct literals do the assignment themselvs (in place)
            if (!r->inplace) {
                TypeStruct* ts = (TypeStruct*)e2->type;
                assert(r->mem);
                LLVM_DtoStructCopy(ts,l->mem,r->mem);
            }
            else {
                e->inplace = true;
            }
        }
        // struct + const int
        else if (e2->type->isintegral()){
            IntegerExp* iexp = (IntegerExp*)e2;
            assert(iexp->value == 0 && "Only integral struct initializer allowed is zero");
            TypeStruct* st = (TypeStruct*)e1->type;
            LLVM_DtoStructZeroInit(st, l->mem);
        }
        // :x
        else
        assert(0 && "struct = unknown");
    }
    else if (e1ty == Tsarray) {
        assert(0 && "static array = not supported");
    }
    else if (e1ty == Tarray) {
        if (e2->type->isscalar() || e2->type->ty == Tclass){
            LLVM_DtoArrayInit(l->mem, r->getValue());
        }
        else if (e2ty == Tarray) {
            //new llvm::StoreInst(r->val,l->val,p->scopebb());
            if (r->type == elem::NUL) {
                llvm::Constant* c = llvm::cast<llvm::Constant>(r->val);
                assert(c->isNullValue());
                LLVM_DtoNullArray(l->mem);
            }
            else if (r->type == elem::SLICE) {
                if (l->type == elem::SLICE)
                LLVM_DtoArrayCopy(l,r);
                else
                LLVM_DtoSetArray(l->mem,r->arg,r->mem);
            }
            else {
                // new expressions write directly to the array reference
                // so do string literals
                if (!r->inplace) {
                    assert(r->mem);
                    LLVM_DtoArrayAssign(l->mem, r->mem);
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
            Logger::cout() << "tmp: " << *tmp << ", " << *l->mem << '\n';
            new llvm::StoreInst(tmp, l->mem, p->scopebb());
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
                    LLVM_DtoNullDelegate(l->mem);
                }
                else
                assert(0);
            }
            else if (r->inplace) {
                // do nothing
                e->inplace = true;
            }
            else
            assert(0);
        }
        else
        assert(0);
    }
    // !struct && !array && !pointer && !class
    else {
        Logger::cout() << *l->mem << '\n';
        new llvm::StoreInst(r->getValue(),l->mem,p->scopebb());
    }

    delete r;
    delete l;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AddExp::toElem(IRState* p)
{
    Logger::print("AddExp::toElem: %s\n", toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* l = e1->toElem(p);
    elem* r = e2->toElem(p);

    if (e1->type != e2->type) {
        if (e1->type->ty == Tpointer && e1->type->next->ty == Tstruct) {
            assert(l->field);
            llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            assert(r->type == elem::CONST);
            llvm::ConstantInt* cofs = llvm::cast<llvm::ConstantInt>(r->val);

            TypeStruct* ts = (TypeStruct*)e1->type->next;
            llvm::Value* offset = llvm::ConstantInt::get(llvm::Type::Int32Ty, ts->sym->offsetToIndex(cofs->getZExtValue()), false);

            e->mem = LLVM_DtoGEP(l->getValue(), zero, offset, "tmp", p->scopebb());
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

    elem* e = new elem;
    llvm::Value* val = 0;
    if (e1->type->ty == Tpointer) {
        val = e->mem = new llvm::GetElementPtrInst(l->getValue(),r->getValue(),"tmp",p->scopebb());
    }
    else {
        val = e->val = llvm::BinaryOperator::createAdd(l->getValue(),r->getValue(),"tmp",p->scopebb());
    }

    /*llvm::Value* storeVal = l->storeVal ? l->storeVal : l->val;
    if (llvm::isa<llvm::PointerType>(storeVal->getType()) && storeVal->getType()->getContainedType(0) != tmp->getType())
    {
        tmp = LLVM_DtoPointedType(storeVal, tmp);
    }*/

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
        left = new llvm::PtrToIntInst(left,LLVM_DtoSize_t(),"tmp",p->scopebb());

    llvm::Value* right = r->getValue();
    if (llvm::isa<llvm::PointerType>(right->getType()))
        right = new llvm::PtrToIntInst(right,LLVM_DtoSize_t(),"tmp",p->scopebb());

    e->val = llvm::BinaryOperator::createSub(left,right,"tmp",p->scopebb());
    e->type = elem::VAL;

    const llvm::Type* totype = LLVM_DtoType(type);
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

    llvm::Value* tmp = 0;
    if (e1->type->ty == Tpointer) {
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
        tmp = LLVM_DtoPointedType(storeVal, tmp);
    }*/

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
        tmp = LLVM_DtoPointedType(storeVal, tmp);
    }*/

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

    if (type->isunsigned())
        e->val = llvm::BinaryOperator::createUDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (type->isintegral())
        e->val = llvm::BinaryOperator::createSDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (type->isfloating())
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

    llvm::Value* tmp;
    if (type->isunsigned())
        tmp = llvm::BinaryOperator::createUDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (type->isintegral())
        tmp = llvm::BinaryOperator::createSDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (type->isfloating())
        tmp = llvm::BinaryOperator::createFDiv(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else
        assert(0);

    /*llvm::Value* storeVal = l->storeVal ? l->storeVal : l->val;
    if (storeVal->getType()->getContainedType(0) != tmp->getType())
    {
        tmp = LLVM_DtoPointedType(storeVal, tmp);
    }*/

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

    if (type->isunsigned())
        e->val = llvm::BinaryOperator::createURem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (type->isintegral())
        e->val = llvm::BinaryOperator::createSRem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (type->isfloating())
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

    llvm::Value* tmp;
    if (type->isunsigned())
        tmp = llvm::BinaryOperator::createURem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (type->isintegral())
        tmp = llvm::BinaryOperator::createSRem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else if (type->isfloating())
        tmp = llvm::BinaryOperator::createFRem(l->getValue(),r->getValue(),"tmp",p->scopebb());
    else
        assert(0);

    /*llvm::Value* storeVal = l->storeVal ? l->storeVal : l->val;
    if (storeVal->getType()->getContainedType(0) != tmp->getType())
    {
        tmp = LLVM_DtoPointedType(storeVal, tmp);
    }*/

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
    LINK dlink = LINKdefault;
    
    bool delegateCall = false;
    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty,0,false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty,1,false);

    // hidden struct return parameter handling
    bool retinptr = false;
    
    TypeFunction* tf = 0;
    
    // regular functions
    if (e1->type->ty == Tfunction) {
        tf = (TypeFunction*)e1->type;
        if (tf->llvmRetInPtr) {
            retinptr = true;
        }
        dlink = tf->linkage;
    }
    
    // delegates
    else if (e1->type->ty == Tdelegate) {
        Logger::println("delegateTy = %s\n", e1->type->toChars());
        assert(e1->type->next->ty == Tfunction);
        tf = (TypeFunction*)e1->type->next;
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

    size_t n = arguments->dim;
    if (fn->arg || delegateCall) n++;
    if (retinptr) n++;

    llvm::Value* funcval = fn->getValue();
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
            funcval = LLVM_DtoGEP(funcval,zero,one,"tmp",p->scopebb());
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
    Logger::cout() << "Function LLVM type: " << *llfnty << '\n';

    // argument handling
    llvm::FunctionType::param_iterator argiter = llfnty->param_begin();
    int j = 0;

    // hidden struct return parameter
    if (retinptr) {
        if (!p->lvals.empty()) {
            assert(llvm::isa<llvm::StructType>(p->toplval()->getType()->getContainedType(0)));
            llargs[j] = p->toplval();
            TY Dty = tf->next->ty;
            if (Dty == Tstruct || Dty == Tdelegate || Dty == Tarray) {
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

    // this parameter
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
    // delegate context parameter
    else if (delegateCall) {
        Logger::println("Delegate Call");
        llvm::Value* contextptr = LLVM_DtoGEP(fn->mem,zero,zero,"tmp",p->scopebb());
        llargs[j] = new llvm::LoadInst(contextptr,"tmp",p->scopebb());
        ++j;
        ++argiter;
    }

    // regular parameters
    for (int i=0; i<arguments->dim; i++,j++)
    {
        Expression* argexp = (Expression*)arguments->data[i];
        elem* arg = argexp->toElem(p);

        Argument* fnarg = Argument::getNth(tf->parameters, i);

        TY argty = argexp->type->ty;
        if (argty == Tstruct || argty == Tdelegate || argty == Tarray) {
            if (!fnarg || !fnarg->llvmCopy) {
                llargs[j] = arg->getValue();
                assert(llargs[j] != 0);
            }
            else {
                llvm::Value* allocaInst = 0;
                llvm::BasicBlock* entryblock = &p->topfunc()->front();
                const llvm::PointerType* pty = llvm::cast<llvm::PointerType>(arg->mem->getType());
                if (argty == Tstruct) {
                    allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", p->topallocapoint());
                    TypeStruct* ts = (TypeStruct*)argexp->type;
                    LLVM_DtoStructCopy(ts,allocaInst,arg->mem);
                }
                else if (argty == Tdelegate) {
                    allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", p->topallocapoint());
                    LLVM_DtoDelegateCopy(allocaInst,arg->mem);
                }
                else if (argty == Tarray) {
                    if (arg->type == elem::SLICE) {
                        allocaInst = new llvm::AllocaInst(LLVM_DtoType(argexp->type), "tmpparam", p->topallocapoint());
                        LLVM_DtoSetArray(allocaInst, arg->arg, arg->mem);
                    }
                    else {
                        allocaInst = new llvm::AllocaInst(pty->getElementType(), "tmpparam", p->topallocapoint());
                        LLVM_DtoArrayAssign(allocaInst,arg->mem);
                    }
                }
                else
                assert(0);

                llargs[j] = allocaInst;
                assert(llargs[j] != 0);
            }
        }
        else if (!fnarg || fnarg->llvmCopy) {
            llargs[j] = arg->getValue();
            assert(llargs[j] != 0);
        }
        else {
            llargs[j] = arg->mem ? arg->mem : arg->val;
            assert(llargs[j] != 0);
        }

        delete arg;
    }

    // void returns cannot not be named
    const char* varname = "";
    if (llfnty->getReturnType() != llvm::Type::VoidTy)
        varname = "tmp";

    Logger::println("%d params passed", n);
    for (int i=0; i<n; ++i)
    {
        Logger::cout() << *llargs[i] << '\n';
    }

    Logger::cout() << "Calling: " << *funcval->getType() << '\n';

    // call the function
    llvm::CallInst* call = new llvm::CallInst(funcval, llargs.begin(), llargs.end(), varname, p->scopebb());
    e->val = call;

    // set calling convention
    if ((fn->funcdecl && (fn->funcdecl->llvmInternal != LLVMintrinsic)) || delegateCall)
        call->setCallingConv(LLVM_DtoCallingConv(dlink));

    delete fn;
    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* CastExp::toElem(IRState* p)
{
    Logger::print("CastExp::toElem: %s\n", toChars());
    LOG_SCOPE;
    elem* e = new elem;
    elem* u = e1->toElem(p);
    const llvm::Type* totype = LLVM_DtoType(to);
    Type* from = e1->type;
    int lsz = from->size();
    int rsz = to->size();

    // this makes sure the strange lvalue casts don't screw things up
    e->mem = u->mem;

    if (from->isintegral()) {
        if (to->isintegral()) {
            if (lsz < rsz) {
                Logger::cout() << *totype << '\n';
                if (from->isunsigned() || from->ty == Tbool) {
                    e->val = new llvm::ZExtInst(u->getValue(), totype, "tmp", p->scopebb());
                } else {
                    e->val = new llvm::SExtInst(u->getValue(), totype, "tmp", p->scopebb());
                }
            }
            else if (lsz > rsz) {
                e->val = new llvm::TruncInst(u->getValue(), totype, "tmp", p->scopebb());
            }
            else {
                e->val = new llvm::BitCastInst(u->getValue(), totype, "tmp", p->scopebb());
            }
        }
        else if (to->isfloating()) {
            if (from->isunsigned()) {
                e->val = new llvm::UIToFPInst(u->getValue(), totype, "tmp", p->scopebb());
            }
            else {
                e->val = new llvm::SIToFPInst(u->getValue(), totype, "tmp", p->scopebb());
            }
        }
        else {
            assert(0);
        }
        //e->storeVal = u->storeVal ? u->storeVal : u->val;
        e->type = elem::VAL;
    }
    else if (from->isfloating()) {
        if (to->isfloating()) {
            if ((from->ty == Tfloat80 || from->ty == Tfloat64) && (to->ty == Tfloat80 || to->ty == Tfloat64)) {
                e->val = u->getValue();
            }
            else if (lsz < rsz) {
                e->val = new llvm::FPExtInst(u->getValue(), totype, "tmp", p->scopebb());
            }
            else if (lsz > rsz) {
                e->val = new llvm::FPTruncInst(u->getValue(), totype, "tmp", p->scopebb());
            }
            else {
                assert(0);
            }
        }
        else if (to->isintegral()) {
            if (to->isunsigned()) {
                e->val = new llvm::FPToUIInst(u->getValue(), totype, "tmp", p->scopebb());
            }
            else {
                e->val = new llvm::FPToSIInst(u->getValue(), totype, "tmp", p->scopebb());
            }
        }
        else {
            assert(0);
        }
        e->type = elem::VAL;
    }
    else if (from->ty == Tclass) {
        //assert(to->ty == Tclass);
        e->val = new llvm::BitCastInst(u->getValue(), totype, "tmp", p->scopebb());
        e->type = elem::VAL;
    }
    else if (from->ty == Tarray || from->ty == Tsarray) {
        Logger::cout() << "from array or sarray" << '\n';
        if (to->ty == Tpointer) {
            Logger::cout() << "to pointer" << '\n';
            assert(from->next == to->next);
            llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
            llvm::Value* ptr = LLVM_DtoGEP(u->getValue(),zero,one,"tmp",p->scopebb());
            e->val = new llvm::LoadInst(ptr, "tmp", p->scopebb());
            e->type = elem::VAL;
        }
        else if (to->ty == Tarray) {
            Logger::cout() << "to array" << '\n';
            assert(from->next->size() == to->next->size());
            const llvm::Type* ptrty = LLVM_DtoType(to->next);
            if (ptrty == llvm::Type::VoidTy)
                ptrty = llvm::Type::Int8Ty;
            ptrty = llvm::PointerType::get(ptrty);

            if (u->type == elem::SLICE) {
                e->mem = new llvm::BitCastInst(u->mem, ptrty, "tmp", p->scopebb());
                e->arg = u->arg;
            }
            else {
                llvm::Value* uval = u->getValue();
                if (from->ty == Tsarray) {
                    Logger::cout() << "uvalTy = " << *uval->getType() << '\n';
                    assert(llvm::isa<llvm::PointerType>(uval->getType()));
                    const llvm::ArrayType* arrty = llvm::cast<llvm::ArrayType>(uval->getType()->getContainedType(0));
                    e->arg = llvm::ConstantInt::get(LLVM_DtoSize_t(), arrty->getNumElements(), false);
                    e->mem = new llvm::BitCastInst(uval, ptrty, "tmp", p->scopebb());
                }
                else {
                    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
                    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
                    e->arg = LLVM_DtoGEP(uval,zero,zero,"tmp",p->scopebb());
                    e->arg = new llvm::LoadInst(e->arg, "tmp", p->scopebb());

                    e->mem = LLVM_DtoGEP(uval,zero,one,"tmp",p->scopebb());
                    e->mem = new llvm::LoadInst(e->mem, "tmp", p->scopebb());
                    //Logger::cout() << *e->mem->getType() << '|' << *ptrty << '\n';
                    e->mem = new llvm::BitCastInst(e->mem, ptrty, "tmp", p->scopebb());
                }
            }
            e->type = elem::SLICE;
        }
        else if (to->ty == Tsarray) {
            Logger::cout() << "to sarray" << '\n';
            assert(0);
        }
        else {
            assert(0);
        }
    }
    else if (from->ty == Tpointer) {
        if (to->ty == Tpointer || to->ty == Tclass) {
            llvm::Value* src = u->getValue();
            //Logger::cout() << *src << '|' << *totype << '\n';
            e->val = new llvm::BitCastInst(src, totype, "tmp", p->scopebb());
        }
        else if (to->isintegral()) {
            e->val = new llvm::PtrToIntInst(u->getValue(), totype, "tmp", p->scopebb());
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
        if (vd->type->ty == Tstruct && !(type->ty == Tpointer && type->next == vd->type)) {
            TypeStruct* vdt = (TypeStruct*)vd->type;
            e = new elem;
            llvm::Value* idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            llvm::Value* idx1 = llvm::ConstantInt::get(llvm::Type::Int32Ty, (uint64_t)vdt->sym->offsetToIndex(offset), false);
            //const llvm::Type* _typ = llvm::GetElementPtrInst::getIndexedType(LLVM_DtoType(type), idx1);
            llvm::Value* ptr = vd->llvmValue;
            assert(ptr);
            e->mem = LLVM_DtoGEP(ptr,idx0,idx1,"tmp",p->scopebb());
            e->type = elem::VAL;
            e->field = true;
        }
        else if (vd->type->ty == Tsarray) {
            /*e = new elem;
            llvm::Value* idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            e->val = new llvm::GetElementPtrInst(vd->llvmValue,idx0,idx0,"tmp",p->scopebb());*/
            e = new elem;
            llvm::Value* idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            //llvm::Value* idx1 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
            e->mem = LLVM_DtoGEP(vd->llvmValue,idx0,idx0,"tmp",p->scopebb());
            e->type = elem::VAL;
        }
        else if (offset == 0) {
            /*if (!vd->llvmValue)
                vd->toObjFile();*/
            assert(vd->llvmValue);
            e = new elem;
            e->mem = vd->llvmValue;
            //e->vardecl = vd;
            e->type = elem::VAL;
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
        //e->aspointer = true;
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

    Logger::print("e1->type=%s\n", e1->type->toChars());

    if (VarDeclaration* vd = var->isVarDeclaration()) {
        size_t vdoffset = (size_t)-1;
        llvm::Value* src = 0;
        if (e1->type->ty == Tpointer) {
            assert(e1->type->next->ty == Tstruct);
            TypeStruct* ts = (TypeStruct*)e1->type->next;
            vdoffset = ts->sym->offsetToIndex(vd->offset);
            Logger::println("Struct member offset:%d index:%d", vd->offset, vdoffset);
            src = l->val;
        }
        else if (e1->type->ty == Tclass) {
            TypeClass* tc = (TypeClass*)e1->type;
            Logger::println("Class member offset: %d", vd->offset);
            vdoffset = tc->sym->offsetToIndex(vd->offset);
            src = l->getValue();
        }
        assert(vdoffset != (size_t)-1);
        assert(src != 0);
        llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
        llvm::Value* offset = llvm::ConstantInt::get(llvm::Type::Int32Ty, vdoffset, false);
        llvm::Value* arrptr = LLVM_DtoGEP(src,zero,offset,"tmp",p->scopebb());
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
            assert(e1->type->ty == Tclass);

            const llvm::Type* vtbltype = llvm::PointerType::get(llvm::ArrayType::get(llvm::PointerType::get(llvm::Type::Int8Ty),0));

            llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
            llvm::Value* vtblidx = llvm::ConstantInt::get(llvm::Type::Int32Ty, (size_t)fdecl->vtblIndex, false);
            funcval = LLVM_DtoGEP(e->arg, zero, zero, "tmp", p->scopebb());
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
            funcval = new llvm::BitCastInst(funcval, vtbltype, "tmp", p->scopebb());
            funcval = LLVM_DtoGEP(funcval, zero, vtblidx, "tmp", p->scopebb());
            funcval = new llvm::LoadInst(funcval,"tmp",p->scopebb());
            funcval = new llvm::BitCastInst(funcval, fdecl->llvmValue->getType(), "tmp", p->scopebb());
        }
        e->val = funcval;
        e->type = elem::VAL;
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
        assert(vd->llvmValue == 0);

        llvm::Function* fn = p->topfunc();
        assert(fn);

        TypeFunction* tf = p->topfunctype();
        assert(tf);

        llvm::Value* v = 0;
        if (tf->llvmRetInPtr)
        v = ++fn->arg_begin();
        else
        v = fn->arg_begin();
        assert(v);

        e->val = v;
        e->type = elem::VAL;
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

    // if there is no lval, this must be a static initializer for a global. correct?
    if (p->lvals.empty())
    {
        // TODO
        assert(0);
    }
    // otherwise write directly in the lvalue
    else
    {
        llvm::Value* sptr = p->toplval();
        assert(sptr);

        llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

        unsigned n = elements->dim;
        for (unsigned i=0; i<n; ++i)
        {
            llvm::Value* offset = llvm::ConstantInt::get(llvm::Type::Int32Ty, i, false);
            llvm::Value* arrptr = LLVM_DtoGEP(sptr,zero,offset,"tmp",p->scopebb());

            Expression* vx = (Expression*)elements->data[i];
            if (vx != 0) {
                elem* ve = vx->toElem(p);
                //Logger::cout() << *ve->val << " | " << *arrptr << '\n';
                new llvm::StoreInst(ve->getValue(), arrptr, p->scopebb());
                delete ve;
            }
            else {
                assert(0);
            }
        }
    }

    e->inplace = true;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* IndexExp::toElem(IRState* p)
{
    Logger::print("IndexExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;

    elem* l = e1->toElem(p);

    p->arrays.push_back(l->mem); // if $ is used it must be an array so this is fine.
    elem* r = e2->toElem(p);
    p->arrays.pop_back();

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);

    llvm::Value* arrptr = 0;
    if (e1->type->ty == Tpointer) {
        arrptr = new llvm::GetElementPtrInst(l->getValue(),r->getValue(),"tmp",p->scopebb());
    }
    else if (e1->type->ty == Tsarray) {
        arrptr = LLVM_DtoGEP(l->mem, zero, r->getValue(),"tmp",p->scopebb());
    }
    else if (e1->type->ty == Tarray) {
        arrptr = LLVM_DtoGEP(l->mem,zero,one,"tmp",p->scopebb());
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

    elem* v = e1->toElem(p);

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

            if (e1->type->ty == Tpointer) {
                e->mem = v->getValue();
            }
            else if (e1->type->ty == Tarray) {
                llvm::Value* tmp = LLVM_DtoGEP(v->mem,zero,one,"tmp",p->scopebb());
                e->mem = new llvm::LoadInst(tmp,"tmp",p->scopebb());
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
            llvm::Value* tmp = LLVM_DtoGEP(v->mem,zero,one,"tmp",p->scopebb());
            tmp = new llvm::LoadInst(tmp,"tmp",p->scopebb());
            e->mem = new llvm::GetElementPtrInst(tmp,lo->getValue(),"tmp",p->scopebb());
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

    assert(e1->type == e2->type);

    Type* t = e1->type;

    if (t->isintegral())
    {
        llvm::ICmpInst::Predicate cmpop;
        switch(op)
        {
        case TOKlt:
            cmpop = t->isunsigned() ? llvm::ICmpInst::ICMP_ULT : llvm::ICmpInst::ICMP_SLT;
            break;
        case TOKle:
            cmpop = t->isunsigned() ? llvm::ICmpInst::ICMP_ULE : llvm::ICmpInst::ICMP_SLE;
            break;
        case TOKgt:
            cmpop = t->isunsigned() ? llvm::ICmpInst::ICMP_UGT : llvm::ICmpInst::ICMP_SGT;
            break;
        case TOKge:
            cmpop = t->isunsigned() ? llvm::ICmpInst::ICMP_UGE : llvm::ICmpInst::ICMP_SGE;
            break;
        default:
            assert(0);
        }
        e->val = new llvm::ICmpInst(cmpop, l->getValue(), r->getValue(), "tmp", p->scopebb());
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

    assert(e1->type == e2->type);

    Type* t = e1->type;

    if (t->isintegral() || t->ty == Tpointer)
    {
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
    else if (t->ty == Tarray)
    {
        // array comparison invokes the typeinfo runtime
        assert(0);
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

    if (e1->type->isintegral())
    {
        assert(e2->type->isintegral());
        llvm::Value* one = llvm::ConstantInt::get(val->getType(), 1, !e2->type->isunsigned());
        if (op == TOKplusplus) {
            post = llvm::BinaryOperator::createAdd(val,one,"tmp",p->scopebb());
        }
        else if (op == TOKminusminus) {
            post = llvm::BinaryOperator::createSub(val,one,"tmp",p->scopebb());
        }
    }
    else if (e1->type->ty == Tpointer)
    {
        assert(e2->type->isintegral());
        llvm::Constant* minusone = llvm::ConstantInt::get(LLVM_DtoSize_t(),(uint64_t)-1,true);
        llvm::Constant* plusone = llvm::ConstantInt::get(LLVM_DtoSize_t(),(uint64_t)1,false);
        llvm::Constant* whichone = (op == TOKplusplus) ? plusone : minusone;
        post = new llvm::GetElementPtrInst(val, whichone, "tmp", p->scopebb());
    }
    else if (e1->type->isfloating())
    {
        assert(e2->type->isfloating());
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

    //llvm::Value* tostore = l->storeVal ? l->storeVal : l->val;
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
    //assert(!arguments);
    //assert(!member);
    assert(!allocator);

    elem* e = new elem;

    const llvm::Type* t = LLVM_DtoType(newtype);

    if (onstack) {
        assert(newtype->ty == Tclass);
        e->mem = new llvm::AllocaInst(t->getContainedType(0),"tmp",p->topallocapoint());
    }
    else {
        if (newtype->ty == Tclass) {
            e->mem = new llvm::MallocInst(t->getContainedType(0),"tmp",p->scopebb());
        }
        else if (newtype->ty == Tarray) {
            t = LLVM_DtoType(newtype->next);
            assert(arguments);
            if (arguments->dim == 1) {
                elem* sz = ((Expression*)arguments->data[0])->toElem(p);
                llvm::Value* dimval = sz->getValue();
                llvm::Value* usedimval = dimval;
                if (dimval->getType() != llvm::Type::Int32Ty)
                    usedimval = new llvm::TruncInst(dimval, llvm::Type::Int32Ty,"tmp",p->scopebb());
                e->mem = new llvm::MallocInst(t,usedimval,"tmp",p->scopebb());

                LLVM_DtoSetArray(p->toplval(), dimval, e->mem);
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

    if (newtype->ty == Tclass) {
        // first apply the static initializer
        assert(e->mem);
        LLVM_DtoInitClass((TypeClass*)newtype, e->mem);

        // then call constructor
        if (arguments) {
            std::vector<llvm::Value*> ctorargs;
            ctorargs.push_back(e->mem);
            for (size_t i=0; i<arguments->dim; ++i)
            {
                Expression* ex = (Expression*)arguments->data[i];
                Logger::println("arg=%s", ex->toChars());
                elem* exe = ex->toElem(p);
                assert(exe->getValue());
                ctorargs.push_back(exe->getValue());
                delete exe;
            }
            assert(member);
            assert(member->llvmValue);
            new llvm::CallInst(member->llvmValue, ctorargs.begin(), ctorargs.end(), "", p->scopebb());
        }
    }
    else if (newtype->ty == Tstruct) {
        TypeStruct* ts = (TypeStruct*)newtype;
        if (ts->isZeroInit()) {
            LLVM_DtoStructZeroInit(ts,e->mem);
        }
        else {
            LLVM_DtoStructCopy(ts,e->mem,ts->llvmInit);
        }
    }

    e->inplace = true;
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

    if (e1->type->ty == Tpointer) {
        ldval = v->getValue();
        new llvm::FreeInst(ldval, p->scopebb());

        Logger::cout() << *z << '\n';
        Logger::cout() << *val << '\n';
        new llvm::StoreInst(z, v->mem, p->scopebb());
    }
    else if (e1->type->ty == Tclass) {
        TypeClass* tc = (TypeClass*)e1->type;
        LLVM_DtoCallClassDtors(tc, val);

        if (v->vardecl && !v->vardecl->onstack) {
            new llvm::FreeInst(val, p->scopebb());
        }
        new llvm::StoreInst(z, v->mem, p->scopebb());
    }
    else if (e1->type->ty == Tarray) {
        // must be on the heap (correct?)
        ldval = v->getValue();
        
        llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
        llvm::Value* one = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1, false);
        
        llvm::Value* ptr = LLVM_DtoGEP(ldval,zero,one,"tmp",p->scopebb());
        ptr = new llvm::LoadInst(ptr,"tmp",p->scopebb());
        new llvm::FreeInst(ptr, p->scopebb());
        LLVM_DtoNullArray(val);
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

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);
    llvm::Value* ptr = LLVM_DtoGEP(u->mem,zero,zero,"tmp",p->scopebb());
    e->val = new llvm::LoadInst(ptr, "tmp", p->scopebb());
    e->type = elem::VAL;

    delete u;

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* AssertExp::toElem(IRState* p)
{
    Logger::print("AssertExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;
    elem* u = e1->toElem(p);
    elem* m = msg ? msg->toElem(p) : 0;

    std::vector<llvm::Value*> llargs;
    llargs.resize(3);
    llargs[0] = LLVM_DtoBoolean(u->getValue());
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int32Ty, loc.linnum, false);
    llargs[2] = m ? m->val : llvm::ConstantPointerNull::get(llvm::PointerType::get(llvm::Type::Int8Ty));
    
    delete m;
    delete u;
    
    //Logger::cout() << *llargs[0] << '|' << *llargs[1] << '\n';
    
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(p->module, "_d_assert");
    assert(fn);
    llvm::CallInst* call = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", p->scopebb());
    call->setCallingConv(llvm::CallingConv::C);

    return e;
}

//////////////////////////////////////////////////////////////////////////////////////////

elem* NotExp::toElem(IRState* p)
{
    Logger::print("NotExp::toElem: %s | %s\n", toChars(), type->toChars());
    LOG_SCOPE;

    elem* e = new elem;
    elem* u = e1->toElem(p);

    llvm::Value* b = LLVM_DtoBoolean(u->getValue());

    llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::Int1Ty, 0, true);
    e->val = new llvm::ICmpInst(llvm::ICmpInst::ICMP_EQ,b,zero,"tmp",p->scopebb());
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
    
    llvm::Value* ubool = LLVM_DtoBoolean(u->getValue());
    new llvm::StoreInst(ubool,resval,p->scopebb());
    new llvm::BranchInst(andand,andandend,ubool,p->scopebb());
    
    p->scope() = IRScope(andand, andandend);
    elem* v = e2->toElem(p);

    llvm::Value* vbool = LLVM_DtoBoolean(v->getValue());
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
    
    llvm::Value* ubool = LLVM_DtoBoolean(u->getValue());
    new llvm::StoreInst(ubool,resval,p->scopebb());
    new llvm::BranchInst(ororend,oror,ubool,p->scopebb());
    
    p->scope() = IRScope(oror, ororend);
    elem* v = e2->toElem(p);

    llvm::Value* vbool = LLVM_DtoBoolean(v->getValue());
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
    llvm::Value* tmp = llvm::BinaryOperator::create(llvm::Instruction::Y, u->getValue(), v->getValue(), "tmp", p->scopebb()); \
    Logger::cout() << *tmp << '|' << *u->mem << '\n'; \
    new llvm::StoreInst(LLVM_DtoPointedType(u->mem, tmp), u->mem, p->scopebb()); \
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

    std::vector<llvm::Value*> llargs;
    llargs.resize(3);
    llargs[0] = llvm::ConstantInt::get(llvm::Type::Int1Ty, 0, false);
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int32Ty, loc.linnum, false);
    llargs[2] = llvm::ConstantPointerNull::get(llvm::PointerType::get(llvm::Type::Int8Ty));
    
    //Logger::cout() << *llargs[0] << '|' << *llargs[1] << '\n';
    
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(p->module, "_d_assert");
    assert(fn);
    llvm::CallInst* call = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", p->scopebb());
    call->setCallingConv(llvm::CallingConv::C);
    
    //new llvm::UnreachableInst(p->scopebb());

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

    llvm::Value* lval = p->toplval();

    llvm::Value* context = LLVM_DtoGEP(lval,zero,zero,"tmp",p->scopebb());
    llvm::Value* castcontext = new llvm::BitCastInst(u->getValue(),int8ptrty,"tmp",p->scopebb());
    new llvm::StoreInst(castcontext, context, p->scopebb());
    
    llvm::Value* fptr = LLVM_DtoGEP(lval,zero,one,"tmp",p->scopebb());
    
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

    llvm::Value* l = u->getValue();
    llvm::Value* r = 0;
    if (v->type == elem::NUL)
    r = llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(l->getType()));
    else
    r = v->getValue();

    llvm::ICmpInst::Predicate pred = (op == TOKidentity) ? llvm::ICmpInst::ICMP_EQ : llvm::ICmpInst::ICMP_NE;
    e->val = new llvm::ICmpInst(pred, l, r, "tmp", p->scopebb());
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

    const llvm::Type* resty = LLVM_DtoType(type);

    // allocate a temporary for the final result. failed to come up with a better way :/
    llvm::BasicBlock* entryblock = &p->topfunc()->front();
    llvm::Value* resval = new llvm::AllocaInst(resty,"condtmp",p->topallocapoint());

    llvm::BasicBlock* oldend = p->scopeend();
    llvm::BasicBlock* condtrue = new llvm::BasicBlock("condtrue", gIR->topfunc(), oldend);
    llvm::BasicBlock* condfalse = new llvm::BasicBlock("condfalse", gIR->topfunc(), oldend);
    llvm::BasicBlock* condend = new llvm::BasicBlock("condend", gIR->topfunc(), oldend);

    elem* c = econd->toElem(p);
    llvm::Value* cond_val = LLVM_DtoBoolean(c->getValue());
    delete c;
    new llvm::BranchInst(condtrue,condfalse,cond_val,p->scopebb());

    p->scope() = IRScope(condtrue, condfalse);
    elem* u = e1->toElem(p);
    Logger::cout() << *u->val << '|' << *resval << '\n'; \
    new llvm::StoreInst(u->getValue(),resval,p->scopebb());
    new llvm::BranchInst(condend,p->scopebb());
    delete u;

    p->scope() = IRScope(condfalse, condend);
    elem* v = e2->toElem(p);
    new llvm::StoreInst(v->getValue(),resval,p->scopebb());
    new llvm::BranchInst(condend,p->scopebb());
    delete v;

    p->scope() = IRScope(condend, oldend);

    elem* e = new elem;
    e->val = new llvm::LoadInst(resval,"tmp",p->scopebb());
    e->type = elem::VAL;
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
STUB(CatExp);
STUB(CatAssignExp);
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
STUB(FuncExp);
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
STUB(NegExp);
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
STUB(ArrayLiteralExp);
STUB(AssocArrayLiteralExp);
//STUB(StructLiteralExp);

unsigned Type::totym() { return 0; }

type *
Type::toCtype() {
    return 0;
}

type * Type::toCParamtype()
{
    return 0;
}
Symbol * Type::toSymbol()
{
    return 0;
}

type *
TypeTypedef::toCtype()
{
    return 0;
}

type *
TypeTypedef::toCParamtype()
{
    return 0;
}

void
TypedefDeclaration::toDebug()
{
}


type *
TypeEnum::toCtype()
{
    return 0;
}

type *
TypeStruct::toCtype()
{
    return 0;
}

void
StructDeclaration::toDebug()
{
}

Symbol * TypeClass::toSymbol()
{
    return 0;
}

unsigned TypeFunction::totym()
{
    return 0;
}

type *
TypeFunction::toCtype()
{
    return 0;
}

type *
TypeSArray::toCtype()
{
    return 0;
}

type *TypeSArray::toCParamtype() { return 0; }

type *
TypeDArray::toCtype()
{
    return 0;
}

type *
TypeAArray::toCtype()
{
    return 0;
}

type *
TypePointer::toCtype()
{
    return 0;
}

type *
TypeDelegate::toCtype()
{
    return 0;
}

type *
TypeClass::toCtype()
{
    return 0;
}

void
ClassDeclaration::toDebug()
{
}

/* --------------------------------------------------------------------------------------- */

void CompoundStatement::toIR(IRState* p)
{
    static int csi = 0;
    Logger::println("CompoundStatement::toIR(%d):\n<<<\n%s>>>", csi++, toChars());
    LOG_SCOPE;

    /*
    const char* labelname;
    bool insterm = false;

    if (!p->scopes()) {
        labelname = "bb";
        insterm = true;
    }
    else
        labelname = "entry";

    //if (!llvm::isa<llvm::TerminatorInst>(p->topfunc()->back().back()))
    //    insterm = true;

    llvm::BasicBlock* bb = new llvm::BasicBlock(labelname, p->topfunc());

    if (insterm) {
        new llvm::BranchInst(bb,p->topbb());
    }

    p->bbs.push(bb);
    */

    size_t n = statements->dim;
    for (size_t i=0; i<n; i++)
    {
        Statement* s = (Statement*)statements->data[i];
        if (s)
        s->toIR(p);
        else
        Logger::println("NULL statement found in CompoundStatement !! :S");
    }

    //p->bbs.pop();
}

void ReturnStatement::toIR(IRState* p)
{
    static int rsi = 0;
    Logger::println("ReturnStatement::toIR(%d): %s", rsi++, toChars());
    LOG_SCOPE;

    if (exp)
    {
        TY expty = exp->type->ty;
        if (p->topfunc()->getReturnType() == llvm::Type::VoidTy) {
            assert(expty == Tstruct || expty == Tdelegate || expty == Tarray);

            TypeFunction* f = p->topfunctype();
            assert(f->llvmRetInPtr && f->llvmRetArg);

            p->lvals.push_back(f->llvmRetArg);
            elem* e = exp->toElem(p);
            p->lvals.pop_back();

            // structliterals do this themselves
            // also they dont produce any value
            if (expty == Tstruct) {
                if (!e->inplace) {
                    TypeStruct* ts = (TypeStruct*)exp->type;
                    assert(e->mem);
                    LLVM_DtoStructCopy(ts,f->llvmRetArg,e->mem);
                }
            }
            else if (expty == Tdelegate) {
                // do nothing, handled by the DelegateExp
                LLVM_DtoDelegateCopy(f->llvmRetArg,e->mem);
            }
            else if (expty == Tarray) {
                if (e->type == elem::SLICE) {
                    LLVM_DtoSetArray(f->llvmRetArg,e->arg,e->mem);
                }
                // else the return value is a variable and should already have been assigned by now
            }
            else
            assert(0);

            new llvm::ReturnInst(p->scopebb());
            delete e;
        }
        else {
            elem* e = exp->toElem(p);
            llvm::Value* v = e->getValue();
            Logger::cout() << *v << '\n';
            new llvm::ReturnInst(v, p->scopebb());
            delete e;
        }
    }
    else
    {
        if (p->topfunc()->getReturnType() == llvm::Type::VoidTy)
            new llvm::ReturnInst(p->scopebb());
        else
            new llvm::UnreachableInst(p->scopebb());
    }

    p->scope().returned = true;
}

void ExpStatement::toIR(IRState* p)
{
    static int esi = 0;
    Logger::println("ExpStatement::toIR(%d): %s", esi++, toChars());
    LOG_SCOPE;

    if (exp != 0) {
        elem* e = exp->toElem(p);
        delete e;
    }
    /*elem* e = exp->toElem(p);
    p->buf.printf("%s", e->toChars());
    delete e;
    p->buf.writenl();*/
}

void IfStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("IfStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    elem* cond_e = condition->toElem(p);
    llvm::Value* cond_val = cond_e->getValue();
    delete cond_e;

    llvm::BasicBlock* oldend = gIR->scopeend();

    llvm::BasicBlock* ifbb = new llvm::BasicBlock("if", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("endif", gIR->topfunc(), oldend);
    llvm::BasicBlock* elsebb = 0;
    if (elsebody) {
        elsebb = new llvm::BasicBlock("else", gIR->topfunc(), endbb);
    }
    else {
        elsebb = endbb;
    }

    if (cond_val->getType() != llvm::Type::Int1Ty) {
        Logger::cout() << "if conditional: " << *cond_val << '\n';
        cond_val = LLVM_DtoBoolean(cond_val);
    }
    llvm::Value* ifgoback = new llvm::BranchInst(ifbb, elsebb, cond_val, gIR->scopebegin());

    // replace current scope
    gIR->scope() = IRScope(ifbb,elsebb);

    bool endifUsed = false;

    // do scoped statements
    ifbody->toIR(p);
    if (!gIR->scopereturned()) {
        new llvm::BranchInst(endbb,gIR->scopebegin());
        endifUsed = true;
    }

    if (elsebody) {
        //assert(0);
        gIR->scope() = IRScope(elsebb,endbb);
        elsebody->toIR(p);
        if (!gIR->scopereturned()) {
            new llvm::BranchInst(endbb,gIR->scopebegin());
            endifUsed = true;
        }
    }

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

void ScopeStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("ScopeStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();

    IRScope irs;
    irs.begin = new llvm::BasicBlock("scope", gIR->topfunc(), oldend);
    irs.end = new llvm::BasicBlock("endscope", gIR->topfunc(), oldend);

    // pass the previous BB into this
    new llvm::BranchInst(irs.begin, gIR->scopebegin());

    gIR->scope() = irs;

    statement->toIR(p);
    if (!gIR->scopereturned()) {
        new llvm::BranchInst(irs.end, gIR->scopebegin());
    }

    // rewrite the scope
    gIR->scope() = IRScope(irs.end,oldend);
}

void WhileStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("WhileStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* whilebb = new llvm::BasicBlock("whilecond", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("endwhile", gIR->topfunc(), oldend);

    // move into the while block
    new llvm::BranchInst(whilebb, gIR->scopebegin());

    // replace current scope
    gIR->scope() = IRScope(whilebb,endbb);

    // create the condition
    elem* cond_e = condition->toElem(p);
    llvm::Value* cond_val = LLVM_DtoBoolean(cond_e->getValue());
    delete cond_e;

    // while body block
    llvm::BasicBlock* whilebodybb = new llvm::BasicBlock("whilebody", gIR->topfunc(), endbb);

    // conditional branch
    llvm::Value* ifbreak = new llvm::BranchInst(whilebodybb, endbb, cond_val, whilebb);

    // rewrite scope
    gIR->scope() = IRScope(whilebodybb,endbb);

    // do while body code
    body->toIR(p);

    // loop
    new llvm::BranchInst(whilebb, gIR->scopebegin());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

void DoStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("DoStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    // create while blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* dowhilebb = new llvm::BasicBlock("dowhile", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("enddowhile", gIR->topfunc(), oldend);

    // move into the while block
    new llvm::BranchInst(dowhilebb, gIR->scopebegin());

    // replace current scope
    gIR->scope() = IRScope(dowhilebb,endbb);

    // do do-while body code
    body->toIR(p);

    // create the condition
    elem* cond_e = condition->toElem(p);
    llvm::Value* cond_val = LLVM_DtoBoolean(cond_e->getValue());
    delete cond_e;

    // conditional branch
    llvm::Value* ifbreak = new llvm::BranchInst(dowhilebb, endbb, cond_val, gIR->scopebegin());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

void ForStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("ForStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    // create for blocks
    llvm::BasicBlock* oldend = gIR->scopeend();
    llvm::BasicBlock* forbb = new llvm::BasicBlock("forcond", gIR->topfunc(), oldend);
    llvm::BasicBlock* forbodybb = new llvm::BasicBlock("forbody", gIR->topfunc(), oldend);
    llvm::BasicBlock* forincbb = new llvm::BasicBlock("forinc", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("endfor", gIR->topfunc(), oldend);

    // init
    if (init != 0)
    init->toIR(p);

    // move into the for condition block, ie. start the loop
    new llvm::BranchInst(forbb, gIR->scopebegin());

    IRScope loop;
    loop.begin = forincbb;
    loop.end = endbb;
    p->loopbbs.push_back(loop);

    // replace current scope
    gIR->scope() = IRScope(forbb,forbodybb);

    // create the condition
    elem* cond_e = condition->toElem(p);
    llvm::Value* cond_val = LLVM_DtoBoolean(cond_e->getValue());
    delete cond_e;

    // conditional branch
    llvm::Value* ifbreak = new llvm::BranchInst(forbodybb, endbb, cond_val, forbb);

    // rewrite scope
    gIR->scope() = IRScope(forbodybb,forincbb);

    // do for body code
    body->toIR(p);

    // move into the for increment block
    new llvm::BranchInst(forincbb, gIR->scopebegin());
    gIR->scope() = IRScope(forincbb, endbb);

    // increment
    if (increment) {
        elem* inc = increment->toElem(p);
        delete inc;
    }

    // loop
    new llvm::BranchInst(forbb, gIR->scopebegin());

    p->loopbbs.pop_back();

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

void BreakStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("BreakStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());
        assert(0);
    }
    else {
        new llvm::BranchInst(gIR->loopbbs.back().end, gIR->scopebegin());
    }
}

void ContinueStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("ContinueStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    if (ident != 0) {
        Logger::println("ident = %s", ident->toChars());
        assert(0);
    }
    else {
        new llvm::BranchInst(gIR->loopbbs.back().begin, gIR->scopebegin());
    }
}

void OnScopeStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("OnScopeStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    assert(statement);
    //statement->toIR(p); // this seems to be redundant
}

void TryFinallyStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("TryFinallyStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    llvm::BasicBlock* oldend = gIR->scopeend();

    llvm::BasicBlock* trybb = new llvm::BasicBlock("try", gIR->topfunc(), oldend);
    llvm::BasicBlock* finallybb = new llvm::BasicBlock("finally", gIR->topfunc(), oldend);
    llvm::BasicBlock* endbb = new llvm::BasicBlock("endtryfinally", gIR->topfunc(), oldend);

    // pass the previous BB into this
    new llvm::BranchInst(trybb, gIR->scopebegin());

    gIR->scope() = IRScope(trybb,finallybb);

    assert(body);
    body->toIR(p);
    new llvm::BranchInst(finallybb, gIR->scopebegin());

    // rewrite the scope
    gIR->scope() = IRScope(finallybb,endbb);

    assert(finalbody);
    finalbody->toIR(p);
    new llvm::BranchInst(endbb, gIR->scopebegin());

    // rewrite the scope
    gIR->scope() = IRScope(endbb,oldend);
}

void TryCatchStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("TryCatchStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    assert(0 && "try-catch is not properly");

    assert(body);
    body->toIR(p);

    assert(catches);
    for(size_t i=0; i<catches->dim; ++i)
    {
        Catch* c = (Catch*)catches->data[i];
        c->handler->toIR(p);
    }
}

void ThrowStatement::toIR(IRState* p)
{
    static int wsi = 0;
    Logger::println("ThrowStatement::toIR(%d): %s", wsi++, toChars());
    LOG_SCOPE;

    assert(0 && "throw is not implemented");

    assert(exp);
    elem* e = exp->toElem(p);
    delete e;
}

#define STUBST(x) void x::toIR(IRState * p) {error("Statement type "#x" not implemented: %s", toChars());fatal();}
//STUBST(BreakStatement);
//STUBST(ForStatement);
STUBST(WithStatement);
STUBST(SynchronizedStatement);
//STUBST(ReturnStatement);
//STUBST(ContinueStatement);
STUBST(DefaultStatement);
STUBST(CaseStatement);
STUBST(SwitchStatement);
STUBST(SwitchErrorStatement);
STUBST(Statement);
//STUBST(IfStatement);
STUBST(ForeachStatement);
//STUBST(DoStatement);
//STUBST(WhileStatement);
//STUBST(ExpStatement);
//STUBST(CompoundStatement);
//STUBST(ScopeStatement);
STUBST(AsmStatement);
//STUBST(TryCatchStatement);
//STUBST(TryFinallyStatement);
STUBST(VolatileStatement);
STUBST(LabelStatement);
//STUBST(ThrowStatement);
STUBST(GotoCaseStatement);
STUBST(GotoDefaultStatement);
STUBST(GotoStatement);
STUBST(UnrolledLoopStatement);
//STUBST(OnScopeStatement);


void
EnumDeclaration::toDebug()
{

}

int
Dsymbol::cvMember(unsigned char*)
{
    return 0;
}
int
EnumDeclaration::cvMember(unsigned char*)
{
    return 0;
}
int
FuncDeclaration::cvMember(unsigned char*)
{
    return 0;
}
int
VarDeclaration::cvMember(unsigned char*)
{
    return 0;
}
int
TypedefDeclaration::cvMember(unsigned char*)
{
    return 0;
}

void obj_includelib(char*){}

AsmStatement::AsmStatement(Loc loc, Token *tokens) :
    Statement(loc)
{
}
Statement *AsmStatement::syntaxCopy() {
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
    return FALSE;
}

void
backend_init()
{
    //LLVM_D_InitRuntime();
    // lazily loaded
}

void
backend_term()
{
    LLVM_D_FreeRuntime();
}
