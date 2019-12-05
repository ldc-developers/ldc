//===- Ops.td - D dialect operation definitions ----------*- tablegen -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the D Dialect.
//
//===----------------------------------------------------------------------===//

#define D_OPS

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

// Provide a definition of the 'D' dialect in the ODS framework so that we
// can define our operations.
def D_Dialect : Dialect {
  let name = "D";
  let cppNamespace = "D";
}

// Base class for D dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class D_Op<string mnemonic, list<OpTrait> traits = []> :
     Op<D_Dialect, mnemonic, traits>;

////////////////////////////////////////////////////////////////////////////////
// AddOIp IntegerLike infers the size of int

def AddIOp : D_Op<"addi"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "addi" operation performs element-wise addition between two
        integer values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs AnyType);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}



////////////////////////////////////////////////////////////////////////////////
// AddFOp FloatLike infers the size of float

def AddFOp : D_Op<"addf"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "addf" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    let results = (outs AnyType);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

////////////////////////////////////////////////////////////////////////////////

def MulOp : D_Op<"mul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
  ];
}