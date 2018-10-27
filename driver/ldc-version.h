//===-- driver/ldc-version.h - ----------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace ldc {

extern const char *const ldc_version;
extern const char *const dmd_version;
extern const char *const llvm_version;
extern const char *const llvm_version_base; /// the base LLVM version without svn/git suffix
extern const char *const built_with_Dcompiler_version;

}
