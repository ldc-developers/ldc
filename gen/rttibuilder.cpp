//===-- rttibuilder.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/rttibuilder.h"
#include "aggregate.h"
#include "mtype.h"
#include "gen/arrays.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "ir/iraggr.h"

RTTIBuilder::RTTIBuilder(AggregateDeclaration* base_class)
{
    DtoResolveDsymbol(base_class);

    base = base_class;
    basetype = static_cast<TypeClass*>(base->type);

    baseir = getIrAggr(base);
    assert(baseir && "no IrStruct for TypeInfo base class");

    if (base->isClassDeclaration()) {
        // just start with adding the vtbl
        inits.push_back(baseir->getVtblSymbol());
        // and monitor
        push_null_vp();
    }
}

void RTTIBuilder::push(llvm::Constant* C)
{
    inits.push_back(C);
}

void RTTIBuilder::push_null(Type* T)
{
    inits.push_back(getNullValue(DtoType(T)));
}

void RTTIBuilder::push_null_vp()
{
    inits.push_back(getNullValue(getVoidPtrType()));
}

void RTTIBuilder::push_typeinfo(Type* t)
{
    inits.push_back(DtoTypeInfoOf(t, true));
}

void RTTIBuilder::push_classinfo(ClassDeclaration* cd)
{
    inits.push_back(getIrAggr(cd)->getClassInfoSymbol());
}

void RTTIBuilder::push_string(const char* str)
{
    inits.push_back(DtoConstString(str));
}

void RTTIBuilder::push_null_void_array()
{
    LLType* T = DtoType(Type::tvoid->arrayOf());
    inits.push_back(getNullValue(T));
}

void RTTIBuilder::push_void_array(uint64_t dim, llvm::Constant* ptr)
{
    inits.push_back(DtoConstSlice(
        DtoConstSize_t(dim),
        DtoBitCast(ptr, getVoidPtrType())
        ));
}

void RTTIBuilder::push_void_array(llvm::Constant* CI, Type* valtype, Dsymbol* mangle_sym)
{
    std::string initname(mangle(mangle_sym));
    initname.append(".rtti.voidarr.data");

    LLGlobalVariable* G = new LLGlobalVariable(
        *gIR->module, CI->getType(), true, TYPEINFO_LINKAGE_TYPE, CI, initname);
    G->setAlignment(valtype->alignsize());

    push_void_array(getTypePaddedSize(CI->getType()), G);
}

void RTTIBuilder::push_array(llvm::Constant * CI, uint64_t dim, Type* valtype, Dsymbol * mangle_sym)
{
    std::string tmpStr(valtype->arrayOf()->toChars());
    tmpStr.erase( remove( tmpStr.begin(), tmpStr.end(), '[' ), tmpStr.end() );
    tmpStr.erase( remove( tmpStr.begin(), tmpStr.end(), ']' ), tmpStr.end() );
    tmpStr.append("arr");

    std::string initname(mangle_sym ? mangle(mangle_sym) : ".ldc");
    initname.append(".rtti.");
    initname.append(tmpStr);
    initname.append(".data");

    LLGlobalVariable* G = new LLGlobalVariable(
        *gIR->module, CI->getType(), true, TYPEINFO_LINKAGE_TYPE, CI, initname);
    G->setAlignment(valtype->alignsize());

    push_array(dim, DtoBitCast(G, DtoType(valtype->pointerTo())));
}

void RTTIBuilder::push_array(uint64_t dim, llvm::Constant * ptr)
{
    inits.push_back(DtoConstSlice(DtoConstSize_t(dim), ptr));
}

void RTTIBuilder::push_uint(unsigned u)
{
    inits.push_back(DtoConstUint(u));
}

void RTTIBuilder::push_size(uint64_t s)
{
    inits.push_back(DtoConstSize_t(s));
}

void RTTIBuilder::push_size_as_vp(uint64_t s)
{
    inits.push_back(llvm::ConstantExpr::getIntToPtr(DtoConstSize_t(s), getVoidPtrType()));
}

void RTTIBuilder::push_funcptr(FuncDeclaration* fd, Type* castto)
{
    if (fd)
    {
        DtoResolveFunction(fd);
        LLConstant* F = getIrFunc(fd)->func;
        if (castto)
            F = DtoBitCast(F, DtoType(castto));
        inits.push_back(F);
    }
    else if (castto)
    {
        push_null(castto);
    }
    else
    {
        push_null_vp();
    }
}

void RTTIBuilder::finalize(IrGlobal* tid)
{
    finalize(tid->type, tid->value);
}

void RTTIBuilder::finalize(LLType* type, LLValue* value)
{
    llvm::ArrayRef<LLConstant*> inits = llvm::makeArrayRef(this->inits);
    LLStructType *st = isaStruct(type);
    assert(st);

    // set struct body
    if (st->isOpaque()) {
        const int n = inits.size();
        std::vector<LLType*> types;
        types.reserve(n);
        for (int i = 0; i < n; ++i)
            types.push_back(inits[i]->getType());
        st->setBody(types);
    }

    // create the inititalizer
    LLConstant* tiInit = LLConstantStruct::get(st, inits);

    // set the initializer
    llvm::GlobalVariable* gvar = llvm::cast<llvm::GlobalVariable>(value);
    gvar->setInitializer(tiInit);
    gvar->setLinkage(TYPEINFO_LINKAGE_TYPE);
}

LLConstant* RTTIBuilder::get_constant(LLStructType *initType)
{
    // just return the inititalizer
    return LLConstantStruct::get(initType, inits);
}
