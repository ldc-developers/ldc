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
// AddOIp 8, 16, 32, 64, 128

def AddI8Op : D_Op<"addi8"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins I8:$lhs, I8:$rhs);
    let results = (outs I8);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

def AddI16Op : D_Op<"addi16"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins I16:$lhs, I16:$rhs);
    let results = (outs I16);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

def AddI32Op : D_Op<"addi32"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins I32:$lhs, I32:$rhs);
    let results = (outs I32);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

def AddI64Op : D_Op<"addi64"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

/*def AddI128Op : D_Op<"addi128"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins I128:$lhs, I128:$rhs);
    let results = (outs I128);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}*/


////////////////////////////////////////////////////////////////////////////////
// AddFOp 16, 32, 64

def AddF16Op : D_Op<"add16f"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins F16:$lhs, F16:$rhs);
    let results = (outs F16);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

def AddF32Op : D_Op<"add32f"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins F32:$lhs, F32:$rhs);
    let results = (outs F32);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

def AddF64Op : D_Op<"add64f"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

def MulOp : D_Op<"mul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
  ];
}