//===-- driver/archiver.h - Creating static libraries -----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_ARCHIVER_H
#define LDC_DRIVER_ARCHIVER_H

/**
 * Create a static library from object files.
 * @return 0 on success.
 */
int createStaticLibrary();

#endif // !LDC_DRIVER_ARCHIVER_H
