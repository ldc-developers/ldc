//===-- disassembler.h - jit support ----------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit disassembler - allow to disassemble in-memory object files
//
//===----------------------------------------------------------------------===//

#pragma once

namespace llvm {
namespace object {
class ObjectFile;
}
class TargetMachine;
class raw_ostream;
}

void disassemble(const llvm::TargetMachine &tm,
                 const llvm::object::ObjectFile &object, llvm::raw_ostream &os);
