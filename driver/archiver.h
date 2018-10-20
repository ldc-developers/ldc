//===-- driver/archiver.h - Creating static libraries -----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

/**
 * Create a static library from object files.
 * @return 0 on success.
 */
int createStaticLibrary();
