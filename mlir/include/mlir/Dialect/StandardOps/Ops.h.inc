/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

class AddFOp;
class AddIOp;
class AllocOp;
class AndOp;
class BranchOp;
class CallIndirectOp;
class CallOp;
class CmpFOp;
class CmpIOp;
class CondBranchOp;
class ConstantOp;
class DeallocOp;
class DimOp;
class DivFOp;
class DivISOp;
class DivIUOp;
class ExtractElementOp;
class IndexCastOp;
class LoadOp;
class MemRefCastOp;
class MulFOp;
class MulIOp;
class OrOp;
class RankOp;
class RemFOp;
class RemISOp;
class RemIUOp;
class ReturnOp;
class SIToFPOp;
class SelectOp;
class ShlISOp;
class SignExtendIOp;
class SplatOp;
class StoreOp;
class SubFOp;
class SubIOp;
class TensorCastOp;
class TensorLoadOp;
class TensorStoreOp;
class TruncateIOp;
class XOrOp;
class ZeroExtendIOp;

//===----------------------------------------------------------------------===//
// AddFOp declarations
//===----------------------------------------------------------------------===//

class AddFOpOperandAdaptor {
public:
  AddFOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class AddFOp : public Op<AddFOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = AddFOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// AddIOp declarations
//===----------------------------------------------------------------------===//

class AddIOpOperandAdaptor {
public:
  AddIOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class AddIOp : public Op<AddIOp, OpTrait::OneResult, OpTrait::IsCommutative, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = AddIOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// AllocOp declarations
//===----------------------------------------------------------------------===//

class AllocOpOperandAdaptor {
public:
  AllocOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  ArrayRef<Value *> value();

private:
  ArrayRef<Value *> tblgen_operands;
};
class AllocOp : public Op<AllocOp, OpTrait::OneResult, OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using OperandAdaptor = AllocOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range value();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, MemRefType memrefType);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, ArrayRef<Value *> value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);

    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }

    /// Returns the number of symbolic operands (the ones in square brackets),
    /// which bind to the symbols of the memref's layout map.
    unsigned getNumSymbolicOperands() {
      return getNumOperands() - getType().getNumDynamicDims();
    }

    /// Returns the symbolic operands (the ones in square brackets), which bind
    /// to the symbols of the memref's layout map.
    operand_range getSymbolicOperands() {
      return {operand_begin() + getType().getNumDynamicDims(), operand_end()};
    }
  
};

//===----------------------------------------------------------------------===//
// AndOp declarations
//===----------------------------------------------------------------------===//

class AndOpOperandAdaptor {
public:
  AndOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class AndOp : public Op<AndOp, OpTrait::OneResult, OpTrait::IsCommutative, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = AndOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// BranchOp declarations
//===----------------------------------------------------------------------===//

class BranchOpOperandAdaptor {
public:
  BranchOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  ArrayRef<Value *> operands();

private:
  ArrayRef<Value *> tblgen_operands;
};
class BranchOp : public Op<BranchOp, OpTrait::ZeroResult, OpTrait::IsTerminator, OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using OperandAdaptor = BranchOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range operands();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &result, Block *dest,ArrayRef<Value *> operands = {});
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Value *> operands);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

    Block *getDest();
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  
};

//===----------------------------------------------------------------------===//
// CallIndirectOp declarations
//===----------------------------------------------------------------------===//

class CallIndirectOpOperandAdaptor {
public:
  CallIndirectOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *callee();
  ArrayRef<Value *> operands();

private:
  ArrayRef<Value *> tblgen_operands;
};
class CallIndirectOp : public Op<CallIndirectOp, OpTrait::VariadicResults, CallOpInterface::Trait, OpTrait::AtLeastNOperands<1>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = CallIndirectOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *callee();
  Operation::operand_range operands();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &result, Value *callee,ArrayRef<Value *> operands = {});
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultType0, Value *callee, ArrayRef<Value *> operands);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);

    Value *getCallee() { return getOperand(0); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }
  
};

//===----------------------------------------------------------------------===//
// CallOp declarations
//===----------------------------------------------------------------------===//

class CallOpOperandAdaptor {
public:
  CallOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  ArrayRef<Value *> operands();

private:
  ArrayRef<Value *> tblgen_operands;
};
class CallOp : public Op<CallOp, OpTrait::VariadicResults, CallOpInterface::Trait, OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using OperandAdaptor = CallOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range operands();
  Operation::result_range getODSResults(unsigned index);
  StringRef callee();
  static void build(Builder *builder, OperationState &result, FuncOp callee,ArrayRef<Value *> operands = {});
  static void build(Builder *builder, OperationState &result, StringRef callee,ArrayRef<Type> results, ArrayRef<Value *> operands = {});
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultType0, SymbolRefAttr callee, ArrayRef<Value *> operands);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("callee");
    }
  
};

//===----------------------------------------------------------------------===//
// CmpFOp declarations
//===----------------------------------------------------------------------===//

class CmpFOpOperandAdaptor {
public:
  CmpFOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class CmpFOp : public Op<CmpFOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameTypeOperands, OpTrait::SameOperandsAndResultShape, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = CmpFOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, CmpFPredicate predicate,Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);

    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpFPredicate getPredicateByName(StringRef name);

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<IntegerAttr>(getPredicateAttrName())
          .getInt();
    }
  
};

//===----------------------------------------------------------------------===//
// CmpIOp declarations
//===----------------------------------------------------------------------===//

class CmpIOpOperandAdaptor {
public:
  CmpIOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class CmpIOp : public Op<CmpIOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameTypeOperands, OpTrait::SameOperandsAndResultShape, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = CmpIOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, CmpIPredicate predicate,Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);

    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpIPredicate getPredicateByName(StringRef name);

    CmpIPredicate getPredicate() {
      return (CmpIPredicate)getAttrOfType<IntegerAttr>(getPredicateAttrName())
          .getInt();
    }
  
};

//===----------------------------------------------------------------------===//
// CondBranchOp declarations
//===----------------------------------------------------------------------===//

class CondBranchOpOperandAdaptor {
public:
  CondBranchOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *condition();
  ArrayRef<Value *> branchOperands();

private:
  ArrayRef<Value *> tblgen_operands;
};
class CondBranchOp : public Op<CondBranchOp, OpTrait::ZeroResult, OpTrait::IsTerminator, OpTrait::AtLeastNOperands<1>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = CondBranchOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *condition();
  Operation::operand_range branchOperands();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &result, Value *condition,Block *trueDest, ArrayRef<Value *> trueOperands,Block *falseDest, ArrayRef<Value *> falseOperands);
  static void build(Builder *, OperationState &tblgen_state, Value *condition, ArrayRef<Value *> branchOperands);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);

    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    // The condition operand is the first operand in the list.
    Value *getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getOperation()->getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getOperation()->getSuccessor(falseIndex);
    }

    // Accessors for operands to the 'true' destination.
    Value *getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value *value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_iterator true_operand_begin() {
      return operand_begin() + getTrueDestOperandIndex();
    }
    operand_iterator true_operand_end() {
      return true_operand_begin() + getNumTrueOperands();
    }
    operand_range getTrueOperands() {
      return {true_operand_begin(), true_operand_end()};
    }

    unsigned getNumTrueOperands()  {
      return getOperation()->getNumSuccessorOperands(trueIndex);
    }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      getOperation()->eraseSuccessorOperand(trueIndex, index);
    }

    // Accessors for operands to the 'false' destination.
    Value *getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value *value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_iterator false_operand_begin() { return true_operand_end(); }
    operand_iterator false_operand_end() {
      return false_operand_begin() + getNumFalseOperands();
    }
    operand_range getFalseOperands() {
      return {false_operand_begin(), false_operand_end()};
    }

    unsigned getNumFalseOperands() {
      return getOperation()->getNumSuccessorOperands(falseIndex);
    }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getOperation()->eraseSuccessorOperand(falseIndex, index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  
};

//===----------------------------------------------------------------------===//
// ConstantOp declarations
//===----------------------------------------------------------------------===//

class ConstantOpOperandAdaptor {
public:
  ConstantOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);

private:
  ArrayRef<Value *> tblgen_operands;
};
class ConstantOp : public Op<ConstantOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using OperandAdaptor = ConstantOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  Attribute value();
  static void build(Builder *builder, OperationState &result, Attribute value);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Attribute value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Attribute value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);

    Attribute getValue() { return getAttr("value"); }

    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  
};

//===----------------------------------------------------------------------===//
// DeallocOp declarations
//===----------------------------------------------------------------------===//

class DeallocOpOperandAdaptor {
public:
  DeallocOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *memref();

private:
  ArrayRef<Value *> tblgen_operands;
};
class DeallocOp : public Op<DeallocOp, OpTrait::ZeroResult, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = DeallocOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *memref();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Value *memref);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
};

//===----------------------------------------------------------------------===//
// DimOp declarations
//===----------------------------------------------------------------------===//

class DimOpOperandAdaptor {
public:
  DimOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *memrefOrTensor();

private:
  ArrayRef<Value *> tblgen_operands;
};
class DimOp : public Op<DimOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = DimOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *memrefOrTensor();
  Operation::result_range getODSResults(unsigned index);
  APInt index();
  static void build(Builder *builder, OperationState &result, Value *memrefOrTensor,unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *memrefOrTensor, IntegerAttr index);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *memrefOrTensor, IntegerAttr index);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
  OpFoldResult fold(ArrayRef<Attribute> operands);

    unsigned getIndex() {
      return getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
    }
  
};

//===----------------------------------------------------------------------===//
// DivFOp declarations
//===----------------------------------------------------------------------===//

class DivFOpOperandAdaptor {
public:
  DivFOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class DivFOp : public Op<DivFOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = DivFOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// DivISOp declarations
//===----------------------------------------------------------------------===//

class DivISOpOperandAdaptor {
public:
  DivISOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class DivISOp : public Op<DivISOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = DivISOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// DivIUOp declarations
//===----------------------------------------------------------------------===//

class DivIUOpOperandAdaptor {
public:
  DivIUOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class DivIUOp : public Op<DivIUOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = DivIUOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// ExtractElementOp declarations
//===----------------------------------------------------------------------===//

class ExtractElementOpOperandAdaptor {
public:
  ExtractElementOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *aggregate();
  ArrayRef<Value *> indices();

private:
  ArrayRef<Value *> tblgen_operands;
};
class ExtractElementOp : public Op<ExtractElementOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::AtLeastNOperands<1>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = ExtractElementOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *aggregate();
  Operation::operand_range indices();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *aggregate,ArrayRef<Value *> indices = {});
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *aggregate, ArrayRef<Value *> indices);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *aggregate, ArrayRef<Value *> indices);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);

    Value *getAggregate() { return getOperand(0); }

    operand_range getIndices() {
      return {getOperation()->operand_begin() + 1,
              getOperation()->operand_end()};
    }
  
};

//===----------------------------------------------------------------------===//
// IndexCastOp declarations
//===----------------------------------------------------------------------===//

class IndexCastOpOperandAdaptor {
public:
  IndexCastOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *in();

private:
  ArrayRef<Value *> tblgen_operands;
};
class IndexCastOp : public Op<IndexCastOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = IndexCastOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *in();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *source, Type destType);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *in);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *in);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  
};

//===----------------------------------------------------------------------===//
// LoadOp declarations
//===----------------------------------------------------------------------===//

class LoadOpOperandAdaptor {
public:
  LoadOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *memref();
  ArrayRef<Value *> indices();

private:
  ArrayRef<Value *> tblgen_operands;
};
class LoadOp : public Op<LoadOp, OpTrait::OneResult, OpTrait::AtLeastNOperands<1>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = LoadOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *memref();
  Operation::operand_range indices();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &result, Value *memref,ArrayRef<Value *> indices = {});
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *memref, ArrayRef<Value *> indices);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *memref, ArrayRef<Value *> indices);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);

    Value *getMemRef() { return getOperand(0); }
    void setMemRef(Value *value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef()->getType().cast<MemRefType>();
    }

    operand_range getIndices() {
      return {getOperation()->operand_begin() + 1, getOperation()->operand_end()};
    }
  
};

//===----------------------------------------------------------------------===//
// MemRefCastOp declarations
//===----------------------------------------------------------------------===//

class MemRefCastOpOperandAdaptor {
public:
  MemRefCastOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *source();

private:
  ArrayRef<Value *> tblgen_operands;
};
class MemRefCastOp : public Op<MemRefCastOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = MemRefCastOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *source();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *source, Type destType);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *source);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *source);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);

    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a memref_cast is always a memref.
    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }
  
};

//===----------------------------------------------------------------------===//
// MulFOp declarations
//===----------------------------------------------------------------------===//

class MulFOpOperandAdaptor {
public:
  MulFOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class MulFOp : public Op<MulFOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = MulFOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// MulIOp declarations
//===----------------------------------------------------------------------===//

class MulIOpOperandAdaptor {
public:
  MulIOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class MulIOp : public Op<MulIOp, OpTrait::OneResult, OpTrait::IsCommutative, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = MulIOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// OrOp declarations
//===----------------------------------------------------------------------===//

class OrOpOperandAdaptor {
public:
  OrOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class OrOp : public Op<OrOp, OpTrait::OneResult, OpTrait::IsCommutative, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = OrOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// RankOp declarations
//===----------------------------------------------------------------------===//

class RankOpOperandAdaptor {
public:
  RankOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);

private:
  ArrayRef<Value *> tblgen_operands;
};
class RankOp : public Op<RankOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = RankOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *tensor);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *tblgen_arg_0);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *tblgen_arg_0);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// RemFOp declarations
//===----------------------------------------------------------------------===//

class RemFOpOperandAdaptor {
public:
  RemFOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class RemFOp : public Op<RemFOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = RemFOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// RemISOp declarations
//===----------------------------------------------------------------------===//

class RemISOpOperandAdaptor {
public:
  RemISOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class RemISOp : public Op<RemISOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = RemISOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// RemIUOp declarations
//===----------------------------------------------------------------------===//

class RemIUOpOperandAdaptor {
public:
  RemIUOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class RemIUOp : public Op<RemIUOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = RemIUOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// ReturnOp declarations
//===----------------------------------------------------------------------===//

class ReturnOpOperandAdaptor {
public:
  ReturnOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  ArrayRef<Value *> operands();

private:
  ArrayRef<Value *> tblgen_operands;
};
class ReturnOp : public Op<ReturnOp, OpTrait::ZeroResult, OpTrait::IsTerminator, OpTrait::HasParent<FuncOp>::Impl, OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using OperandAdaptor = ReturnOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range operands();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *b, OperationState &result);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Value *> operands);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// SIToFPOp declarations
//===----------------------------------------------------------------------===//

class SIToFPOpOperandAdaptor {
public:
  SIToFPOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *in();

private:
  ArrayRef<Value *> tblgen_operands;
};
class SIToFPOp : public Op<SIToFPOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = SIToFPOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *in();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *source, Type destType);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *in);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *in);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  
};

//===----------------------------------------------------------------------===//
// SelectOp declarations
//===----------------------------------------------------------------------===//

class SelectOpOperandAdaptor {
public:
  SelectOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *condition();
  Value *true_value();
  Value *false_value();

private:
  ArrayRef<Value *> tblgen_operands;
};
class SelectOp : public Op<SelectOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultShape, OpTrait::NOperands<3>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = SelectOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *condition();
  Value *true_value();
  Value *false_value();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *condition,Value *trueValue, Value *falseValue);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *condition, Value *true_value, Value *false_value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *condition, Value *true_value, Value *false_value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);

      Value *getCondition() { return condition(); }
      Value *getTrueValue() { return true_value(); }
      Value *getFalseValue() { return false_value(); }
  
};

//===----------------------------------------------------------------------===//
// ShlISOp declarations
//===----------------------------------------------------------------------===//

class ShlISOpOperandAdaptor {
public:
  ShlISOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class ShlISOp : public Op<ShlISOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = ShlISOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// SignExtendIOp declarations
//===----------------------------------------------------------------------===//

class SignExtendIOpOperandAdaptor {
public:
  SignExtendIOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *value();

private:
  ArrayRef<Value *> tblgen_operands;
};
class SignExtendIOp : public Op<SignExtendIOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultShape, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = SignExtendIOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *value();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *value, Type destType);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// SplatOp declarations
//===----------------------------------------------------------------------===//

class SplatOpOperandAdaptor {
public:
  SplatOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *input();

private:
  ArrayRef<Value *> tblgen_operands;
};
class SplatOp : public Op<SplatOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = SplatOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *input();
  Operation::result_range getODSResults(unsigned index);
  Value *aggregate();
  static void build(Builder *builder, OperationState &result, Value *element, Type aggregateType);
  static void build(Builder *, OperationState &tblgen_state, Type aggregate, Value *input);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *input);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// StoreOp declarations
//===----------------------------------------------------------------------===//

class StoreOpOperandAdaptor {
public:
  StoreOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *value();
  Value *memref();
  ArrayRef<Value *> indices();

private:
  ArrayRef<Value *> tblgen_operands;
};
class StoreOp : public Op<StoreOp, OpTrait::ZeroResult, OpTrait::AtLeastNOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = StoreOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *value();
  Value *memref();
  Operation::operand_range indices();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &result, Value *valueToStore, Value *memref);
  static void build(Builder *, OperationState &tblgen_state, Value *value, Value *memref, ArrayRef<Value *> indices);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);

      Value *getValueToStore() { return getOperand(0); }

      Value *getMemRef() { return getOperand(1); }
      void setMemRef(Value *value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return getMemRef()->getType().cast<MemRefType>();
      }

      operand_range getIndices() {
        return {getOperation()->operand_begin() + 2, getOperation()->operand_end()};
      }
  
};

//===----------------------------------------------------------------------===//
// SubFOp declarations
//===----------------------------------------------------------------------===//

class SubFOpOperandAdaptor {
public:
  SubFOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class SubFOp : public Op<SubFOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = SubFOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// SubIOp declarations
//===----------------------------------------------------------------------===//

class SubIOpOperandAdaptor {
public:
  SubIOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class SubIOp : public Op<SubIOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = SubIOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// TensorCastOp declarations
//===----------------------------------------------------------------------===//

class TensorCastOpOperandAdaptor {
public:
  TensorCastOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);

private:
  ArrayRef<Value *> tblgen_operands;
};
class TensorCastOp : public Op<TensorCastOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = TensorCastOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *source, Type destType);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *tblgen_arg_0);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *tblgen_arg_0);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);

    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a tensor_cast is always a tensor.
    TensorType getType() { return getResult()->getType().cast<TensorType>(); }
  
};

//===----------------------------------------------------------------------===//
// TensorLoadOp declarations
//===----------------------------------------------------------------------===//

class TensorLoadOpOperandAdaptor {
public:
  TensorLoadOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);

private:
  ArrayRef<Value *> tblgen_operands;
};
class TensorLoadOp : public Op<TensorLoadOp, OpTrait::OneResult, OpTrait::SameOperandsAndResultShape, OpTrait::SameOperandsAndResultElementType, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = TensorLoadOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *memref);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *tblgen_arg_0);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *tblgen_arg_0);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

    /// The result of a tensor_load is always a tensor.
    TensorType getType() { return getResult()->getType().cast<TensorType>(); }
  
};

//===----------------------------------------------------------------------===//
// TensorStoreOp declarations
//===----------------------------------------------------------------------===//

class TensorStoreOpOperandAdaptor {
public:
  TensorStoreOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *tensor();
  Value *memref();

private:
  ArrayRef<Value *> tblgen_operands;
};
class TensorStoreOp : public Op<TensorStoreOp, OpTrait::ZeroResult, OpTrait::SameOperandsShape, OpTrait::SameOperandsElementType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = TensorStoreOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *tensor();
  Value *memref();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Value *tensor, Value *memref);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// TruncateIOp declarations
//===----------------------------------------------------------------------===//

class TruncateIOpOperandAdaptor {
public:
  TruncateIOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *value();

private:
  ArrayRef<Value *> tblgen_operands;
};
class TruncateIOp : public Op<TruncateIOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultShape, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = TruncateIOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *value();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *value, Type destType);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// XOrOp declarations
//===----------------------------------------------------------------------===//

class XOrOpOperandAdaptor {
public:
  XOrOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();

private:
  ArrayRef<Value *> tblgen_operands;
};
class XOrOp : public Op<XOrOp, OpTrait::OneResult, OpTrait::IsCommutative, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultType, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = XOrOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *lhs();
  Value *rhs();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *lhs, Value *rhs);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  OpFoldResult fold(ArrayRef<Attribute> operands);
};

//===----------------------------------------------------------------------===//
// ZeroExtendIOp declarations
//===----------------------------------------------------------------------===//

class ZeroExtendIOpOperandAdaptor {
public:
  ZeroExtendIOpOperandAdaptor(ArrayRef<Value *> values);
  ArrayRef<Value *> getODSOperands(unsigned index);
  Value *value();

private:
  ArrayRef<Value *> tblgen_operands;
};
class ZeroExtendIOp : public Op<ZeroExtendIOp, OpTrait::OneResult, OpTrait::HasNoSideEffect, OpTrait::SameOperandsAndResultShape, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = ZeroExtendIOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value *value();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result, Value *value, Type destType);
  static void build(Builder *, OperationState &tblgen_state, Type resultType0, Value *value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, Value *value);
  static void build(Builder *, OperationState &tblgen_state, ArrayRef<Type> resultTypes, ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

#endif  // GET_OP_CLASSES

