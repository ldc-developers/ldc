#include "gen/llvm.h"

#include "aggregate.h"
#include "mtype.h"

#include "gen/arrays.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvmhelpers.h"
#include "gen/rttibuilder.h"
#include "gen/tollvm.h"

#include "ir/irstruct.h"

RTTIBuilder::RTTIBuilder(ClassDeclaration* base_class)
{
    // make sure the base typeinfo class has been processed
    base_class->codegen(Type::sir);

    base = base_class;
    basetype = (TypeClass*)base->type;

    baseir = base->ir.irStruct;
    assert(baseir && "no IrStruct for TypeInfo base class");

    // just start with adding the vtbl
    inits.push_back(baseir->getVtblSymbol());
    // and monitor
    push_null_vp();
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
    inits.push_back(cd->ir.irStruct->getClassInfoSymbol());
}

void RTTIBuilder::push_string(const char* str)
{
    inits.push_back(DtoConstString(str));
}

void RTTIBuilder::push_null_void_array()
{
    const llvm::Type* T = DtoType(Type::tvoid->arrayOf());
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
    std::string initname(mangle_sym->mangle());
    initname.append(".rtti.void[].data");

    LLGlobalVariable* G = new llvm::GlobalVariable(
        CI->getType(), true, TYPEINFO_LINKAGE_TYPE, CI, initname, gIR->module);
    G->setAlignment(valtype->alignsize());

    push_void_array(getTypePaddedSize(CI->getType()), G);
}

void RTTIBuilder::push_array(llvm::Constant * CI, uint64_t dim, Type* valtype, Dsymbol * mangle_sym)
{
    std::string initname(mangle_sym?mangle_sym->mangle():".ldc");
    initname.append(".rtti.");
    initname.append(valtype->arrayOf()->toChars());
    initname.append(".data");

    LLGlobalVariable* G = new llvm::GlobalVariable(
        CI->getType(), true, TYPEINFO_LINKAGE_TYPE, CI, initname, gIR->module);
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

void RTTIBuilder::push_funcptr(FuncDeclaration* fd, Type* castto)
{
    if (fd)
    {
        fd->codegen(Type::sir);
        LLConstant* F = fd->ir.irFunc->func;
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
    // create the inititalizer
    LLConstant* tiInit = llvm::ConstantStruct::get(&inits[0], inits.size(), false);

    // refine global type
    llvm::cast<llvm::OpaqueType>(tid->type.get())->refineAbstractTypeTo(tiInit->getType());

    // set the initializer
    isaGlobalVar(tid->value)->setInitializer(tiInit);
}

LLConstant* RTTIBuilder::get_constant()
{
    // just return the inititalizer
    return llvm::ConstantStruct::get(&inits[0], inits.size(), false);
}
