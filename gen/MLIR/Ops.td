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

include "mlir/IR/OpBase.td"

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
// AddOp IntegerLike infers the size of int

def AddOp : D_Op<"add"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two
        integer values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

////////////////////////////////////////////////////////////////////////////////
// AddFOp FloatLike infers the size of float

def AddFOp : D_Op<"fadd"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "addf" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    let results = (outs FloatLike);

    // Allow building an AddOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

////////////////////////////////////////////////////////////////////////////////
// SubOp IntegerLike infers the size of int

def SubOp : D_Op<"sub"> {
    let summary = "element-wise subtraction operation";
    let description = [{
        The "sub" operation performs element-wise subtraction between two
        integer values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

    // Allow building an SubOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

////////////////////////////////////////////////////////////////////////////////
// SubFOp FloatLike infers the size of float

def SubFOp : D_Op<"fsub"> {
    let summary = "element-wise subtraction operation";
    let description = [{
        The "fsub" operation performs element-wise subtraction between two
        float values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    let results = (outs FloatLike);

    // Allow building an SubFOp with from the two input operands.
    let builders = [
      OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
    ];
}

////////////////////////////////////////////////////////////////////////////////

def MulOp : D_Op<"mul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    operands. The type of operands are expected to match.
  }];

  let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
  let results = (outs IntegerLike);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
  ];
}

////////////////////////////////////////////////////////////////////////////////

def MulFOp : D_Op<"fmul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    operands. The type of operands are expected to match.
  }];

  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
  let results = (outs FloatLike);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
  ];
}

////////////////////////////////////////////////////////////////////////////////
def DivSOp : D_Op<"sdiv"> {
    let summary = "signed element-wise division operation";
    let description = [{
        The "sdiv" operation performs element-wise division between two signed
        operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

   // Allow building a DivSOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////
def DivUOp : D_Op<"udiv"> {
    let summary = "unsigned element-wise division operation";
    let description = [{
        The "sdiv" operation performs element-wise division between two unsigned
        operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

   // Allow building a DivUOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////
def DivFOp : D_Op<"fdiv"> {
    let summary = "element-wise division operation";
    let description = [{
        The "fdiv" operation performs element-wise division between two
        operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    let results = (outs FloatLike);

   // Allow building a DivIOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////
def ModSOp : D_Op<"srem"> {
    let summary = "Returns the signed integer remainder of a division.";
    let description = [{
        The ‘srem‘ instruction returns the remainder from the signed division of
        its two operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

   // Allow building a ModSOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////
def ModUOp : D_Op<"urem"> {
    let summary = "Returns the unsigned integer remainder of a division.";
    let description = [{
        The ‘urem‘ instruction returns the remainder from the un signed division
        of its two operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

   // Allow building a ModUOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////
def ModFOp : D_Op<"frem"> {
    let summary = "Returns the remainder of a division.";
    let description = [{
        The ‘frem‘ instruction returns the remainder from the division of its
        two operands. The remainder has the same sign as the dividend.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    let results = (outs FloatLike);

   // Allow building a ModFOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////
def AndOp : D_Op<"and"> {
    let summary = "Logical 'and' between two operands";
    let description = [{
        The ‘and‘ instruction returns the bitwise logical and of its two operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

   // Allow building a AndOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////
def OrOp : D_Op<"or"> {
    let summary = "Logical 'or' between two operands";
    let description = [{
        The ‘or‘ instruction returns the bitwise logical and of its two operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

   // Allow building a OrOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////
def XorOp : D_Op<"xor"> {
    let summary = "Logical 'xor' between two operands";
    let description = [{
        The ‘xor‘ instruction returns the bitwise logical and of its two
        operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins IntegerLike:$lhs, IntegerLike:$rhs);
    let results = (outs IntegerLike);

   // Allow building a XorOp with from the two input operands.
     let builders = [
       OpBuilder<"Builder *b, OperationState &state, Value *lhs, Value *rhs">
     ];
}

////////////////////////////////////////////////////////////////////////////////

def CallOp : D_Op<"call"> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that was
        already declared on the scope. The callee name is attached
        as a symbol reference via an attribute. The arguments list must match the
        arguments expected by the callee. For example:

        ```mlir
         %4 = "D.call"(%1, %3) {callee = @my_func}
               : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

        This is only valid if a function named "my_func" exists and takes two
        arguments.
    }];

    // The call operation takes a symbol reference attribute as the callee, and
    // inputs for the call.
    let arguments = (ins FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$inputs);

    // The generic call operation returns a single value of AnyType - The type
    // is inferred later.
    let results = (outs AnyType);

    // Add custom build methods for the call operation.
    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, "
                  "StringRef callee, ArrayRef<Type> types, "
                  "ArrayRef<Value *> arguments">
    ];
}

def IntegerOp : D_Op<"int", [NoSideEffect]> {
  let summary = "Integer";
  let description = [{
    IntegerExp operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "D.Integer"() {value = 10 : i32} : () -> i32
    ```
  }];

 /* // The integerExp operation takes an attribute as the only input.
  let arguments = (ins IntegerLike:$value);*/
  let arguments = (ins I64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs IntegerLike);

  // Add custom build methods for the integerExp operation. These method
  // populates the `state` that MLIR uses to create operations, i.e. these are
  // used when using `builder.create<IntegerOp>(...)`.
  let builders = [
    // Build a constant with a given constant integer value.
    OpBuilder<"Builder *builder, OperationState &state, int value">,

    // Build a integer with a given value.
    OpBuilder<"Builder *builder, OperationState &state, "
              "DenseIntElementsAttr value", [{
                build(builder, state, value.getType(), value);
                }]>,
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}
