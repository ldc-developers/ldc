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

/// Check whether the frontend knows that the function is already defined
/// in some other module (see DMD's `FuncDeclaration_toObjFile()`).
bool skipCodegen(FuncDeclaration &fdecl);

/// Returns whether `fdecl` should be emitted with externally_available
/// linkage to make it available for inlining.
///
/// If true, `semantic3` will have been run on the declaration.
bool defineAsExternallyAvailable(FuncDeclaration &fdecl);
