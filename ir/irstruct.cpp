#include "gen/llvm.h"
#include "mtype.h"
#include "aggregate.h"
#include "ir/irstruct.h"

IrInterface::IrInterface(BaseClass* b, const llvm::StructType* vt)
{
    base = b;
    decl = b->base;
    vtblTy = vt;
    vtblInit = NULL;
    vtbl = NULL;
    infoTy = NULL;
    infoInit = NULL;
    info = NULL;
}

IrInterface::~IrInterface()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrStruct::IrStruct(Type* t)
 : recty((t->llvmType != NULL) ? *t->llvmType : llvm::OpaqueType::get())
{
    type = t;
    defined = false;
    constinited = false;
    interfaceInfosTy = NULL;
    interfaceInfos = NULL;
}

IrStruct::~IrStruct()
{
}
