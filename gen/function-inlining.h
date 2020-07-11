//===-- gen/function-inlining.h ---------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Determines whether a function is fit for inlining or not.
//
//===----------------------------------------------------------------------===//

#pragma once

class FuncDeclaration;

/// Returns whether `fdecl` should be emitted with externally_available
/// linkage to make it available for inlining.
///
/// If true, `semantic3` will have been run on the declaration.
bool defineAsExternallyAvailable(FuncDeclaration &fdecl);
