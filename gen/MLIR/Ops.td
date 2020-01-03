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
// Binary Operations
////////////////////////////////////////////////////////////////////////////////

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
