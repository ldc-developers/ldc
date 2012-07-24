#ifdef _MSC_VER
#pragma once
#endif

#ifndef LDC_LLVMCOMPAT_H
#define LDC_LLVMCOMPAT_H

#include "llvm/ADT/Triple.h"
#include <string>

#if !defined(LDC_LLVM_VER)
#error "Please specify value for LDC_LLVM_VER."
#endif

#if LDC_LLVM_VER == 300
namespace llvm {
    namespace sys {
        std::string getDefaultTargetTriple();
    }

    Triple Triple__get32BitArchVariant(const std::string&_this);
    Triple Triple__get64BitArchVariant(const std::string& _this);
}
#endif

#endif
