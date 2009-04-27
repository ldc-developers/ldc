/// Support for callbacks when an abstract type becomes more concrete.

#include "llvm/Support/Streams.h"
#include "llvm/Type.h"
#include "llvm-c/Core.h"

using namespace llvm;

extern "C" typedef int (*RefineCallback)(void *handle, LLVMTypeRef newT);

class TypeMonitor : AbstractTypeUser {
    void *handle_;
    RefineCallback callback_;
    
    void onRefineType(const Type* oldT, const Type* newT) {
        callback_(handle_, wrap(newT));
        oldT->removeAbstractTypeUser(this);
        delete this;
    }
    
    public:
    
    TypeMonitor(Type* T, void *handle, RefineCallback callback)
    : handle_(handle), callback_(callback) {
        T->addAbstractTypeUser(this);
    }
    
    virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
        onRefineType(OldTy, NewTy);
    }
    
    virtual void typeBecameConcrete(const DerivedType *AbsTy) {
        onRefineType(AbsTy, AbsTy);
    }
    
    virtual void dump() const {
        cerr << "<TypeMonitor>";
    }
};

extern "C" void LLVMRegisterAbstractTypeCallback(LLVMTypeRef T,
                                                 void *handle,
                                                 RefineCallback callback)
{
    new TypeMonitor(unwrap(T), handle, callback);
}
