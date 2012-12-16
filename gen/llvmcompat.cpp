//===-- llvmcompat.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvmcompat.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#if LDC_LLVM_VER >= 302
#include "llvm/IRBuilder.h"
#else
#include "llvm/Support/IRBuilder.h"
#endif
#include <string>

#if LDC_LLVM_VER == 300
using namespace llvm;

std::string llvm::sys::getDefaultTargetTriple() {
    return LLVM_HOSTTRIPLE;
}

Triple llvm::Triple__get32BitArchVariant(const std::string& triple) {
    Triple T(triple);
    switch (T.getArch()) {
        case Triple::UnknownArch:
        case Triple::msp430:
            T.setArch(Triple::UnknownArch);
            break;

        case Triple::amdil:
        case Triple::arm:
        case Triple::cellspu:
        case Triple::le32:
        case Triple::mblaze:
        case Triple::mips:
        case Triple::mipsel:
        case Triple::ppc:
        case Triple::sparc:
        case Triple::tce:
        case Triple::thumb:
        case Triple::x86:
        case Triple::xcore:
            // Already 32-bit.
            break;

        case Triple::mips64:    T.setArch(Triple::mips);    break;
        case Triple::mips64el:  T.setArch(Triple::mipsel);  break;
        case Triple::ppc64:     T.setArch(Triple::ppc);   break;
        case Triple::sparcv9:   T.setArch(Triple::sparc);   break;
        case Triple::x86_64:    T.setArch(Triple::x86);     break;
    }
    return T;
}

Triple llvm::Triple__get64BitArchVariant(const std::string& triple) {
    Triple T(triple);
    switch (T.getArch()) {
        case Triple::UnknownArch:
        case Triple::amdil:
        case Triple::arm:
        case Triple::cellspu:
        case Triple::le32:
        case Triple::mblaze:
        case Triple::msp430:
        case Triple::tce:
        case Triple::thumb:
        case Triple::xcore:
            T.setArch(Triple::UnknownArch);
            break;

        case Triple::mips64:
        case Triple::mips64el:
        case Triple::ppc64:
        case Triple::sparcv9:
        case Triple::x86_64:
            // Already 64-bit.
            break;

        case Triple::mips:    T.setArch(Triple::mips64);    break;
        case Triple::mipsel:  T.setArch(Triple::mips64el);  break;
        case Triple::ppc:     T.setArch(Triple::ppc64);     break;
        case Triple::sparc:   T.setArch(Triple::sparcv9);   break;
        case Triple::x86:     T.setArch(Triple::x86_64);    break;
    }
    return T;
}

static void appendToGlobalArray(const char *Array,
                                Module &M, Function *F, int Priority) {
    IRBuilder<> IRB(M.getContext());
    FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);
    StructType *Ty = StructType::get(
        IRB.getInt32Ty(), PointerType::getUnqual(FnTy), NULL);

    Constant *RuntimeCtorInit = ConstantStruct::get(
        Ty, IRB.getInt32(Priority), F, NULL);

    // Get the current set of static global constructors and add the new ctor
    // to the list.
    SmallVector<Constant *, 16> CurrentCtors;
    if (GlobalVariable * GVCtor = M.getNamedGlobal(Array)) {
    if (Constant *Init = GVCtor->getInitializer()) {
        unsigned n = Init->getNumOperands();
        CurrentCtors.reserve(n + 1);
        for (unsigned i = 0; i != n; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
    }

    CurrentCtors.push_back(RuntimeCtorInit);

    // Create a new initializer.
    ArrayType *AT = ArrayType::get(RuntimeCtorInit->getType(),
                                    CurrentCtors.size());
    Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

    // Create the new global variable and replace all uses of
    // the old global variable with the new one.
    (void)new GlobalVariable(M, NewInit->getType(), false,
                            GlobalValue::AppendingLinkage, NewInit, Array);
}

void llvm::appendToGlobalCtors(Module &M, Function *F, int Priority) {
    appendToGlobalArray("llvm.global_ctors", M, F, Priority);
}

void llvm::appendToGlobalDtors(Module &M, Function *F, int Priority) {
    appendToGlobalArray("llvm.global_dtors", M, F, Priority);
}

#endif


