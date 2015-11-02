//===-- gen/llvmcompat.h - LLVM API compatibilty shims ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Provides a central place to handle API changes between supported LLVM
// versions.
//
//===----------------------------------------------------------------------===//

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LDC_LLVMCOMPAT_H
#define LDC_LLVMCOMPAT_H

#if !defined(LDC_LLVM_VER)
#error "Please specify value for LDC_LLVM_VER."
#endif

#define ADDRESS_SPACE 0

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(cxx_override_control) ||                                     \
    (defined(_MSC_VER) && _MSC_VER >= 1700)
#define LLVM_OVERRIDE override
#else
#define LLVM_OVERRIDE
#endif

#if (__has_feature(cxx_rvalue_references) ||                                   \
     defined(__GXX_EXPERIMENTAL_CXX0X__) ||                                    \
     (defined(_MSC_VER) && _MSC_VER >= 1600))
#define LLVM_HAS_RVALUE_REFERENCES 1
#else
#define LLVM_HAS_RVALUE_REFERENCES 0
#endif

#if LLVM_HAS_RVALUE_REFERENCES
#define llvm_move(value) (::std::move(value))
#else
#define llvm_move(value) (value)
#endif

#endif
