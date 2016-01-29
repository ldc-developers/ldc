//===-- ir/irforw.h - Forward declarations used in ir/ code  ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Some common forward declarations for use in ir/ headers.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRFORW_H
#define LDC_IR_IRFORW_H

// dmd forward declarations
class Module;
class Dsymbol;
struct Declaration;
class VarDeclaration;
class FuncDeclaration;
struct AggregateDeclaration;
class StructDeclaration;
class ClassDeclaration;
struct InterfaceDeclaration;
struct Expression;
struct BaseClass;
struct Array;
struct Argument;

class Type;
class TypeStruct;
class TypeClass;
struct TypeEnum;
struct TypeArray;
class TypeFunction;

// llvm forward declarations
namespace llvm {
class Value;
class GlobalValue;
class GlobalVariable;
class Function;
class Constant;
class ConstantStruct;
class ConstantArray;
class DataLayout;
class Type;
class StructType;
class ArrayType;
class PointerType;
class BasicBlock;
class Instruction;
}

#endif
