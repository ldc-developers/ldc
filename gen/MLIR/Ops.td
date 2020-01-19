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

// Provide a definition for the D StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef.
def D_StructType :
    Type<CPred<"$_self.isa<StructType>()">, "D struct type">;

// Provide a definition of the types that are used within the D dialect.
def D_Type : AnyTypeOf<[AnyType, D_StructType]>;

////////////////////////////////////////////////////////////////////////////////
// Struct Operations
////////////////////////////////////////////////////////////////////////////////

def StructAccessOp : D_Op<"struct_access", [NoSideEffect]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins D_StructType:$input, I64Attr:$index);
  let results = (outs D_Type);

  // Allow building a StructAccessOp with just a struct value and an index.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value input, size_t index">
  ];

  let verifier = [{ return ::verify(*this); }];

  // Set the folder bit so that we can fold constant accesses.
 // let hasFolder = 1;
}

def StructConstantOp : D_Op<"struct_constant", [NoSideEffect]> {
  let summary = "struct constant";
  let description = [{
    Constant operation turns a literal struct value into an SSA value. The data
    is attached to the operation as an attribute. The struct constant is encoded
    as an array of other constant values. For example:

    ```mlir
      %0 = "D.struct_constant"() {
        value = [dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>]
      } : () -> !D.struct<tensor<*xf64>>
    ```
  }];

 // let hasFolder = 1;
  let arguments = (ins ArrayAttr:$value);
  let results = (outs D_StructType);
  let verifier = [{ return ::verify(*this); }];
}

////////////////////////////////////////////////////////////////////////////////
// Cast Operations
////////////////////////////////////////////////////////////////////////////////

def CastOp : D_Op<"cast"> {
  let summary = "cast type";
  let description = [{
    The "cast" operation converts a Value from one type to another type
    without changing any data elements. If both are ranked
    then the rank should be the same and static dimensions should match. The
    operation is invalid if converting to a mismatching constant dimension.
  }];

  let arguments = (ins AnyTypeOf<[AnyType, AnyMemRef]>:$input);
  let results = (outs AnyType:$output);

  // Set the folder bit so that we can fold redundant cast operations.
 // let hasFolder = 1;
}
