// Converted to the D programming language by Tomas Lindquist Olsen 2008
//                                        and Frits van Bommel 2008
// Original file header:
/*===-- llvm-c/Core.h - Core Library C Interface ------------------*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMCore.a, which implements    *|
|* the LLVM intermediate representation.                                      *|
|*                                                                            *|
|* LLVM uses a polymorphic type hierarchy which C cannot represent, therefore *|
|* parameters must be passed as base types. Despite the declared types, most  *|
|* of the functions provided operate only on branches of the type hierarchy.  *|
|* The declared parameter names are descriptive and specify which type is     *|
|* required. Additionally, each type hierarchy is documented along with the   *|
|* functions that operate upon it. For more detail, refer to LLVM's C++ code. *|
|* If in doubt, refer to Core.cpp, which performs paramter downcasts in the   *|
|* form unwrap<RequiredType>(Param).                                          *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
|* When included into a C++ source file, also declares 'wrap' and 'unwrap'    *|
|* helpers to perform opaque reference<-->pointer conversions. These helpers  *|
|* are shorter and more tightly typed than writing the casts by hand when     *|
|* authoring bindings. In assert builds, they will do runtime type checking.  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
module llvm.c.Core;

extern(C):

/**
 * The top-level container for all other LLVM Intermediate Representation (IR)
 * objects. See the llvm::Module class.
 */
typedef void* LLVMModuleRef;

/**
 * Each value in the LLVM IR has a type, an instance of [lltype]. See the
 * llvm::Type class.
 */
typedef void* LLVMTypeRef;

/**
 * When building recursive types using [refine_type], [lltype] values may become
 * invalid; use [lltypehandle] to resolve this problem. See the
 * llvm::AbstractTypeHolder] class.
 */
typedef void* LLVMTypeHandleRef;

typedef void* LLVMValueRef;
typedef void* LLVMBasicBlockRef;
typedef void* LLVMBuilderRef;

/* Used to provide a module to JIT or interpreter.
 * See the llvm::ModuleProvider class.
 */
typedef void* LLVMModuleProviderRef;

/* Used to provide a module to JIT or interpreter.
 * See the llvm::MemoryBuffer class.
 */
typedef void* LLVMMemoryBufferRef;

/** See the llvm::PassManagerBase class. */
typedef void* LLVMPassManagerRef;

enum LLVMAttribute {
    ZExt       = 1<<0,
    SExt       = 1<<1,
    NoReturn   = 1<<2,
    InReg      = 1<<3,
    StructRet  = 1<<4,
    NoUnwind   = 1<<5,
    NoAlias    = 1<<6,
    ByVal      = 1<<7,
    Nest       = 1<<8,
    ReadNone   = 1<<9,
    ReadOnly   = 1<<10,
    NoInline   = 1<<11,
    AlwaysInline    = 1<<12,
    OptimizeForSize = 1<<13,
    StackProtect    = 1<<14,
    StackProtectReq = 1<<15,
    NoCapture  = 1<<21,
    NoRedZone  = 1<<22,
    NoImplicitFloat = 1<<23,
    Naked      = 1<<24
}

enum LLVMTypeKind {
    Void,        /**< type with no size */
    Float,       /**< 32 bit floating point type */
    Double,      /**< 64 bit floating point type */
    X86_FP80,    /**< 80 bit floating point type (X87) */
    FP128,       /**< 128 bit floating point type (112-bit mantissa)*/
    PPC_FP128,   /**< 128 bit floating point type (two 64-bits) */
    Label,       /**< Labels */
    Integer,     /**< Arbitrary bit width integers */
    Function,    /**< Functions */
    Struct,      /**< Structures */
    Array,       /**< Arrays */
    Pointer,     /**< Pointers */
    Opaque,      /**< Opaque: type with unknown structure */
    Vector,      /**< SIMD 'packed' format, or other vector type */
    Metadata     /**< Metadata */
}

enum LLVMLinkage {
    External,    /**< Externally visible function */
    AvailableExternally,
    LinkOnceAny, /**< Keep one copy of function when linking (inline)*/
    LinkOnceODR, /**< Same, but only replaced by something
                                equivalent. */
    WeakAny,     /**< Keep one copy of function when linking (weak) */
    WeakODR,     /**< Same, but only replaced by something
                                equivalent. */
    Appending,   /**< Special purpose, only applies to global arrays */
    Internal,    /**< Rename collisions when linking (static
                                functions) */
    Private,     /**< Like Internal, but omit from symbol table */
    DLLImport,   /**< Function to be imported from DLL */
    DLLExport,   /**< Function to be accessible from DLL */
    ExternalWeak,/**< ExternalWeak linkage description */
    Ghost,       /**< Stand-in functions for streaming fns from
                                bitcode */
    Common,      /**< Tentative definitions */
    LinkerPrivate /**< Like Private, but linker removes. */
}

enum LLVMVisibility {
  Default,  /**< The GV is visible */
  Hidden,   /**< The GV is hidden */
  Protected/**< The GV is protected */
}

enum LLVMCallConv {
  C          = 0,
  Fast       = 8,
  Cold       = 9,
  X86Stdcall = 64,
  X86Fastcall= 65
}

enum LLVMIntPredicate {
  EQ = 32, /**< equal */
  NE,      /**< not equal */
  UGT,     /**< uint greater than */
  UGE,     /**< uint greater or equal */
  ULT,     /**< uint less than */
  ULE,     /**< uint less or equal */
  SGT,     /**< signed greater than */
  SGE,     /**< signed greater or equal */
  SLT,     /**< signed less than */
  SLE      /**< signed less or equal */
}

enum LLVMRealPredicate {
  False,          /**< Always false (always folded) */
  OEQ,            /**< True if ordered and equal */
  OGT,            /**< True if ordered and greater than */
  OGE,            /**< True if ordered and greater than or equal */
  OLT,            /**< True if ordered and less than */
  OLE,            /**< True if ordered and less than or equal */
  ONE,            /**< True if ordered and operands are unequal */
  ORD,            /**< True if ordered (no nans) */
  UNO,            /**< True if unordered: isnan(X) | isnan(Y) */
  UEQ,            /**< True if unordered or equal */
  UGT,            /**< True if unordered or greater than */
  UGE,            /**< True if unordered, greater than, or equal */
  ULT,            /**< True if unordered or less than */
  ULE,            /**< True if unordered, less than, or equal */
  UNE,            /**< True if unordered or not equal */
  True            /**< Always true (always folded) */
}

/*===-- Error handling ----------------------------------------------------===*/

void LLVMDisposeMessage(char *Message);


/*===-- Modules -----------------------------------------------------------===*/

/* Create and destroy modules. */
/** See llvm::Module::Module. */
LLVMModuleRef LLVMModuleCreateWithName(/*const*/ char *ModuleID);

/** See llvm::Module::~Module. */
void LLVMDisposeModule(LLVMModuleRef M);

/** Data layout. See Module::getDataLayout. */
/*const*/ char *LLVMGetDataLayout(LLVMModuleRef M);
void LLVMSetDataLayout(LLVMModuleRef M, /*const*/ char *DataLayout);

/** Target triple. See Module::getTargetTriple. */
/*const*/ char *LLVMGetTarget(LLVMModuleRef M);
void LLVMSetTarget(LLVMModuleRef M, /*const*/ char *Triple);

/** See Module::addTypeName. */
int LLVMAddTypeName(LLVMModuleRef M, /*const*/ char *Name, LLVMTypeRef Ty);
void LLVMDeleteTypeName(LLVMModuleRef M, /*const*/ char *Name);

/** See Module::dump. */
void LLVMDumpModule(LLVMModuleRef M);

/*===-- Types -------------------------------------------------------------===*/

/* LLVM types conform to the following hierarchy:
 *
 *   types:
 *     integer type
 *     real type
 *     function type
 *     sequence types:
 *       array type
 *       pointer type
 *       vector type
 *     void type
 *     label type
 *     opaque type
 */

LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef Ty);

/* Operations on integer types */
LLVMTypeRef LLVMInt1Type();
LLVMTypeRef LLVMInt8Type();
LLVMTypeRef LLVMInt16Type();
LLVMTypeRef LLVMInt32Type();
LLVMTypeRef LLVMInt64Type();
LLVMTypeRef LLVMIntType(uint NumBits);
uint LLVMGetIntTypeWidth(LLVMTypeRef IntegerTy);

/* Operations on real types */
LLVMTypeRef LLVMFloatType();
LLVMTypeRef LLVMDoubleType();
LLVMTypeRef LLVMX86FP80Type();
LLVMTypeRef LLVMFP128Type();
LLVMTypeRef LLVMPPCFP128Type();

/* Operations on function types */
LLVMTypeRef LLVMFunctionType(LLVMTypeRef ReturnType,
                             LLVMTypeRef *ParamTypes, uint ParamCount,
                             int IsVarArg);
int LLVMIsFunctionVarArg(LLVMTypeRef FunctionTy);
LLVMTypeRef LLVMGetReturnType(LLVMTypeRef FunctionTy);
uint LLVMCountParamTypes(LLVMTypeRef FunctionTy);
void LLVMGetParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef *Dest);

/* Operations on struct types */
LLVMTypeRef LLVMStructType(LLVMTypeRef *ElementTypes, uint ElementCount,
                           int Packed);
uint LLVMCountStructElementTypes(LLVMTypeRef StructTy);
void LLVMGetStructElementTypes(LLVMTypeRef StructTy, LLVMTypeRef *Dest);
int LLVMIsPackedStruct(LLVMTypeRef StructTy);

/* Operations on array, pointer, and vector types (sequence types) */
LLVMTypeRef LLVMArrayType(LLVMTypeRef ElementType, uint ElementCount);
LLVMTypeRef LLVMPointerType(LLVMTypeRef ElementType, uint AddressSpace);
LLVMTypeRef LLVMVectorType(LLVMTypeRef ElementType, uint ElementCount);

LLVMTypeRef LLVMGetElementType(LLVMTypeRef Ty);
uint LLVMGetArrayLength(LLVMTypeRef ArrayTy);
uint LLVMGetPointerAddressSpace(LLVMTypeRef PointerTy);
uint LLVMGetVectorSize(LLVMTypeRef VectorTy);

/* Operations on other types */
LLVMTypeRef LLVMVoidType();
LLVMTypeRef LLVMLabelType();
LLVMTypeRef LLVMOpaqueType();

/* Operations on type handles */
LLVMTypeHandleRef LLVMCreateTypeHandle(LLVMTypeRef PotentiallyAbstractTy);
void LLVMRefineType(LLVMTypeRef AbstractTy, LLVMTypeRef ConcreteTy);
LLVMTypeRef LLVMResolveTypeHandle(LLVMTypeHandleRef TypeHandle);
void LLVMDisposeTypeHandle(LLVMTypeHandleRef TypeHandle);


/*===-- Values ------------------------------------------------------------===*/

/* The bulk of LLVM's object model consists of values, which comprise a very
 * rich type hierarchy.
 *
 *   values:
 *     constants:
 *       scalar constants
 *       composite contants
 *       globals:
 *         global variable
 *         function
 *         alias
 *       basic blocks
 */

/* Operations on all values */
LLVMTypeRef LLVMTypeOf(LLVMValueRef Val);
/*const*/ char *LLVMGetValueName(LLVMValueRef Val);
void LLVMSetValueName(LLVMValueRef Val, /*const*/ char *Name);
void LLVMDumpValue(LLVMValueRef Val);

/* Operations on constants of any type */
LLVMValueRef LLVMConstNull(LLVMTypeRef Ty); /* all zeroes */
LLVMValueRef LLVMConstAllOnes(LLVMTypeRef Ty); /* only for int/vector */
LLVMValueRef LLVMGetUndef(LLVMTypeRef Ty);
int LLVMIsConstant(LLVMValueRef Val);
int LLVMIsNull(LLVMValueRef Val);
int LLVMIsUndef(LLVMValueRef Val);

/* Operations on scalar constants */
LLVMValueRef LLVMConstInt(LLVMTypeRef IntTy, ulong N,
                          int SignExtend);
LLVMValueRef LLVMConstIntOfString(LLVMTypeRef IntTy, /*const*/ char *Text,
                                  ubyte Radix);
LLVMValueRef LLVMConstIntOfStringAndSize(LLVMTypeRef IntTy, /*const*/ char *Text,
                                         uint SLen, ubyte Radix);
LLVMValueRef LLVMConstReal(LLVMTypeRef RealTy, double N);
LLVMValueRef LLVMConstRealOfString(LLVMTypeRef RealTy, /*const*/ char *Text);
LLVMValueRef LLVMConstRealOfStringAndSize(LLVMTypeRef RealTy, /*const*/ char *Text,
                                          uint SLen);

/* Operations on composite constants */
LLVMValueRef LLVMConstString(/*const*/ char *Str, uint Length,
                             int DontNullTerminate);
LLVMValueRef LLVMConstArray(LLVMTypeRef ElementTy,
                            LLVMValueRef *ConstantVals, uint Length);
LLVMValueRef LLVMConstStruct(LLVMValueRef *ConstantVals, uint Count,
                             int packed);
LLVMValueRef LLVMConstVector(LLVMValueRef *ScalarConstantVals, uint Size);

/* Constant expressions */
LLVMValueRef LLVMSizeOf(LLVMTypeRef Ty);
LLVMValueRef LLVMConstNeg(LLVMValueRef ConstantVal);
LLVMValueRef LLVMConstNot(LLVMValueRef ConstantVal);
LLVMValueRef LLVMConstAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstUDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstSDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstURem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstSRem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFRem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstAnd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstOr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstXor(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstICmp(LLVMIntPredicate Predicate,
                           LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFCmp(LLVMRealPredicate Predicate,
                           LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstShl(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstLShr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstAShr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstGEP(LLVMValueRef ConstantVal,
                          LLVMValueRef *ConstantIndices, uint NumIndices);
LLVMValueRef LLVMConstTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstSExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstZExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstFPTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstFPExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstUIToFP(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstSIToFP(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstFPToUI(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstFPToSI(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstPtrToInt(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstIntToPtr(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstBitCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstSelect(LLVMValueRef ConstantCondition,
                             LLVMValueRef ConstantIfTrue,
                             LLVMValueRef ConstantIfFalse);
LLVMValueRef LLVMConstExtractElement(LLVMValueRef VectorConstant,
                                     LLVMValueRef IndexConstant);
LLVMValueRef LLVMConstInsertElement(LLVMValueRef VectorConstant,
                                    LLVMValueRef ElementValueConstant,
                                    LLVMValueRef IndexConstant);
LLVMValueRef LLVMConstShuffleVector(LLVMValueRef VectorAConstant,
                                    LLVMValueRef VectorBConstant,
                                    LLVMValueRef MaskConstant);
LLVMValueRef LLVMConstExtractValue(LLVMValueRef AggConstant, uint *IdxList,
                                   uint NumIdx);
LLVMValueRef LLVMConstInsertValue(LLVMValueRef AggConstant,
                                  LLVMValueRef ElementValueConstant,
                                  uint *IdxList, uint NumIdx);

/* Operations on global variables, functions, and aliases (globals) */
LLVMModuleRef LLVMGetGlobalParent(LLVMValueRef Global);
int LLVMIsDeclaration(LLVMValueRef Global);
LLVMLinkage LLVMGetLinkage(LLVMValueRef Global);
void LLVMSetLinkage(LLVMValueRef Global, LLVMLinkage Linkage);
/*const*/ char *LLVMGetSection(LLVMValueRef Global);
void LLVMSetSection(LLVMValueRef Global, /*const*/ char *Section);
LLVMVisibility LLVMGetVisibility(LLVMValueRef Global);
void LLVMSetVisibility(LLVMValueRef Global, LLVMVisibility Viz);
uint LLVMGetAlignment(LLVMValueRef Global);
void LLVMSetAlignment(LLVMValueRef Global, uint Bytes);

/* Operations on global variables */
LLVMValueRef LLVMAddGlobal(LLVMModuleRef M, LLVMTypeRef Ty, /*const*/ char *Name);
LLVMValueRef LLVMGetNamedGlobal(LLVMModuleRef M, /*const*/ char *Name);
LLVMValueRef LLVMGetFirstGlobal(LLVMModuleRef M);
LLVMValueRef LLVMGetLastGlobal(LLVMModuleRef M);
LLVMValueRef LLVMGetNextGlobal(LLVMValueRef GlobalVar);
LLVMValueRef LLVMGetPreviousGlobal(LLVMValueRef GlobalVar);
void LLVMDeleteGlobal(LLVMValueRef GlobalVar);
LLVMValueRef LLVMGetInitializer(LLVMValueRef GlobalVar);
void LLVMSetInitializer(LLVMValueRef GlobalVar, LLVMValueRef ConstantVal);
int LLVMIsThreadLocal(LLVMValueRef GlobalVar);
void LLVMSetThreadLocal(LLVMValueRef GlobalVar, int IsThreadLocal);
int LLVMIsGlobalConstant(LLVMValueRef GlobalVar);
void LLVMSetGlobalConstant(LLVMValueRef GlobalVar, int IsConstant);

/* Operations on functions */
LLVMValueRef LLVMAddFunction(LLVMModuleRef M, /*const*/ char *Name,
                             LLVMTypeRef FunctionTy);
LLVMValueRef LLVMGetNamedFunction(LLVMModuleRef M, /*const*/ char *Name);
LLVMValueRef LLVMGetFirstFunction(LLVMModuleRef M);
LLVMValueRef LLVMGetLastFunction(LLVMModuleRef M);
LLVMValueRef LLVMGetNextFunction(LLVMValueRef Fn);
LLVMValueRef LLVMGetPreviousFunction(LLVMValueRef Fn);
void LLVMDeleteFunction(LLVMValueRef Fn);
uint LLVMGetIntrinsicID(LLVMValueRef Fn);
uint LLVMGetFunctionCallConv(LLVMValueRef Fn);
void LLVMSetFunctionCallConv(LLVMValueRef Fn, uint CC);
/*const*/ char *LLVMGetGC(LLVMValueRef Fn);
void LLVMSetGC(LLVMValueRef Fn, /*const*/ char *Name);
void LLVMAddFunctionAttr(LLVMValueRef Fn, LLVMAttribute PA);
void LLVMRemoveFunctionAttr(LLVMValueRef Fn, LLVMAttribute PA);

/* Operations on parameters */
uint LLVMCountParams(LLVMValueRef Fn);
void LLVMGetParams(LLVMValueRef Fn, LLVMValueRef *Params);
LLVMValueRef LLVMGetParam(LLVMValueRef Fn, uint Index);
LLVMValueRef LLVMGetParamParent(LLVMValueRef Inst);
LLVMValueRef LLVMGetFirstParam(LLVMValueRef Fn);
LLVMValueRef LLVMGetLastParam(LLVMValueRef Fn);
LLVMValueRef LLVMGetNextParam(LLVMValueRef Arg);
LLVMValueRef LLVMGetPreviousParam(LLVMValueRef Arg);
void LLVMAddAttribute(LLVMValueRef Arg, LLVMAttribute PA);
void LLVMRemoveAttribute(LLVMValueRef Arg, LLVMAttribute PA);
void LLVMSetParamAlignment(LLVMValueRef Arg, uint alignm);


/* Operations on basic blocks */
LLVMValueRef LLVMBasicBlockAsValue(LLVMBasicBlockRef Bb);
int LLVMValueIsBasicBlock(LLVMValueRef Val);
LLVMBasicBlockRef LLVMValueAsBasicBlock(LLVMValueRef Val);
LLVMValueRef LLVMGetBasicBlockParent(LLVMBasicBlockRef BB);
uint LLVMCountBasicBlocks(LLVMValueRef Fn);
void LLVMGetBasicBlocks(LLVMValueRef Fn, LLVMBasicBlockRef *BasicBlocks);
LLVMBasicBlockRef LLVMGetFirstBasicBlock(LLVMValueRef Fn);
LLVMBasicBlockRef LLVMGetLastBasicBlock(LLVMValueRef Fn);
LLVMBasicBlockRef LLVMGetNextBasicBlock(LLVMBasicBlockRef BB);
LLVMBasicBlockRef LLVMGetPreviousBasicBlock(LLVMBasicBlockRef BB);
LLVMBasicBlockRef LLVMGetEntryBasicBlock(LLVMValueRef Fn);
LLVMBasicBlockRef LLVMAppendBasicBlock(LLVMValueRef Fn, /*const*/ char *Name);
LLVMBasicBlockRef LLVMInsertBasicBlock(LLVMBasicBlockRef InsertBeforeBB,
                                       /*const*/ char *Name);
void LLVMDeleteBasicBlock(LLVMBasicBlockRef BB);

/* Operations on instructions */
LLVMBasicBlockRef LLVMGetInstructionParent(LLVMValueRef Inst);
LLVMValueRef LLVMGetFirstInstruction(LLVMBasicBlockRef BB);
LLVMValueRef LLVMGetLastInstruction(LLVMBasicBlockRef BB);
LLVMValueRef LLVMGetNextInstruction(LLVMValueRef Inst);
LLVMValueRef LLVMGetPreviousInstruction(LLVMValueRef Inst);

/* Operations on call sites */
void LLVMSetInstructionCallConv(LLVMValueRef Instr, uint CC);
uint LLVMGetInstructionCallConv(LLVMValueRef Instr);
void LLVMAddInstrAttribute(LLVMValueRef Instr, uint index, LLVMAttribute);
void LLVMRemoveInstrAttribute(LLVMValueRef Instr, uint index, LLVMAttribute);
void LLVMSetInstrParamAlignment(LLVMValueRef Instr, uint index, uint alignm);

/* Operations on call instructions (only) */
int LLVMIsTailCall(LLVMValueRef CallInst);
void LLVMSetTailCall(LLVMValueRef CallInst, int IsTailCall);

/* Operations on phi nodes */
void LLVMAddIncoming(LLVMValueRef PhiNode, LLVMValueRef *IncomingValues,
                     LLVMBasicBlockRef *IncomingBlocks, uint Count);
uint LLVMCountIncoming(LLVMValueRef PhiNode);
LLVMValueRef LLVMGetIncomingValue(LLVMValueRef PhiNode, uint Index);
LLVMBasicBlockRef LLVMGetIncomingBlock(LLVMValueRef PhiNode, uint Index);

/*===-- Instruction builders ----------------------------------------------===*/

/* An instruction builder represents a point within a basic block, and is the
 * exclusive means of building instructions using the C interface.
 */

LLVMBuilderRef LLVMCreateBuilder();
void LLVMPositionBuilder(LLVMBuilderRef Builder, LLVMBasicBlockRef Block,
                         LLVMValueRef Instr);
void LLVMPositionBuilderBefore(LLVMBuilderRef Builder, LLVMValueRef Instr);
void LLVMPositionBuilderAtEnd(LLVMBuilderRef Builder, LLVMBasicBlockRef Block);
LLVMBasicBlockRef LLVMGetInsertBlock(LLVMBuilderRef Builder);
void LLVMDisposeBuilder(LLVMBuilderRef Builder);

/* Terminators */
LLVMValueRef LLVMBuildRetVoid(LLVMBuilderRef);
LLVMValueRef LLVMBuildRet(LLVMBuilderRef, LLVMValueRef V);
LLVMValueRef LLVMBuildBr(LLVMBuilderRef, LLVMBasicBlockRef Dest);
LLVMValueRef LLVMBuildCondBr(LLVMBuilderRef, LLVMValueRef If,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Else);
LLVMValueRef LLVMBuildSwitch(LLVMBuilderRef, LLVMValueRef V,
                             LLVMBasicBlockRef Else, uint NumCases);
LLVMValueRef LLVMBuildInvoke(LLVMBuilderRef, LLVMValueRef Fn,
                             LLVMValueRef *Args, uint NumArgs,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch,
                             /*const*/ char *Name);
LLVMValueRef LLVMBuildUnwind(LLVMBuilderRef);
LLVMValueRef LLVMBuildUnreachable(LLVMBuilderRef);

/* Add a case to the switch instruction */
void LLVMAddCase(LLVMValueRef Switch, LLVMValueRef OnVal,
                 LLVMBasicBlockRef Dest);

/* Arithmetic */
LLVMValueRef LLVMBuildAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          /*const*/ char *Name);
LLVMValueRef LLVMBuildSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          /*const*/ char *Name);
LLVMValueRef LLVMBuildMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          /*const*/ char *Name);
LLVMValueRef LLVMBuildUDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildSDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildFDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildURem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildSRem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildFRem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildShl(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildLShr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildAShr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildAnd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          /*const*/ char *Name);
LLVMValueRef LLVMBuildOr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          /*const*/ char *Name);
LLVMValueRef LLVMBuildXor(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          /*const*/ char *Name);
LLVMValueRef LLVMBuildNeg(LLVMBuilderRef, LLVMValueRef V, /*const*/ char *Name);
LLVMValueRef LLVMBuildNot(LLVMBuilderRef, LLVMValueRef V, /*const*/ char *Name);

/* Memory */
LLVMValueRef LLVMBuildMalloc(LLVMBuilderRef, LLVMTypeRef Ty, /*const*/ char *Name);
LLVMValueRef LLVMBuildArrayMalloc(LLVMBuilderRef, LLVMTypeRef Ty,
                                  LLVMValueRef Val, /*const*/ char *Name);
LLVMValueRef LLVMBuildAlloca(LLVMBuilderRef, LLVMTypeRef Ty, /*const*/ char *Name);
LLVMValueRef LLVMBuildArrayAlloca(LLVMBuilderRef, LLVMTypeRef Ty,
                                  LLVMValueRef Val, /*const*/ char *Name);
LLVMValueRef LLVMBuildFree(LLVMBuilderRef, LLVMValueRef PointerVal);
LLVMValueRef LLVMBuildLoad(LLVMBuilderRef, LLVMValueRef PointerVal,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildStore(LLVMBuilderRef, LLVMValueRef Val, LLVMValueRef Ptr);
LLVMValueRef LLVMBuildGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                          LLVMValueRef *Indices, uint NumIndices,
                          /*const*/ char *Name);
LLVMValueRef LLVMBuildInBoundsGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                                  LLVMValueRef *Indices, uint NumIndices,
                                  /*const*/ char *Name);

/* Casts */
LLVMValueRef LLVMBuildTrunc(LLVMBuilderRef, LLVMValueRef Val,
                            LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildZExt(LLVMBuilderRef, LLVMValueRef Val,
                           LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildSExt(LLVMBuilderRef, LLVMValueRef Val,
                           LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildFPToUI(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildFPToSI(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildUIToFP(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildSIToFP(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildFPTrunc(LLVMBuilderRef, LLVMValueRef Val,
                              LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildFPExt(LLVMBuilderRef, LLVMValueRef Val,
                            LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildPtrToInt(LLVMBuilderRef, LLVMValueRef Val,
                               LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildIntToPtr(LLVMBuilderRef, LLVMValueRef Val,
                               LLVMTypeRef DestTy, /*const*/ char *Name);
LLVMValueRef LLVMBuildBitCast(LLVMBuilderRef, LLVMValueRef Val,
                              LLVMTypeRef DestTy, /*const*/ char *Name);

/* Comparisons */
LLVMValueRef LLVMBuildICmp(LLVMBuilderRef, LLVMIntPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildFCmp(LLVMBuilderRef, LLVMRealPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           /*const*/ char *Name);

/* Miscellaneous instructions */
LLVMValueRef LLVMBuildPhi(LLVMBuilderRef, LLVMTypeRef Ty, /*const*/ char *Name);
LLVMValueRef LLVMBuildCall(LLVMBuilderRef, LLVMValueRef Fn,
                           LLVMValueRef *Args, uint NumArgs,
                           /*const*/ char *Name);
LLVMValueRef LLVMBuildSelect(LLVMBuilderRef, LLVMValueRef If,
                             LLVMValueRef Then, LLVMValueRef Else,
                             /*const*/ char *Name);
LLVMValueRef LLVMBuildVAArg(LLVMBuilderRef, LLVMValueRef List, LLVMTypeRef Ty,
                            /*const*/ char *Name);
LLVMValueRef LLVMBuildExtractElement(LLVMBuilderRef, LLVMValueRef VecVal,
                                     LLVMValueRef Index, /*const*/ char *Name);
LLVMValueRef LLVMBuildInsertElement(LLVMBuilderRef, LLVMValueRef VecVal,
                                    LLVMValueRef EltVal, LLVMValueRef Index,
                                    /*const*/ char *Name);
LLVMValueRef LLVMBuildShuffleVector(LLVMBuilderRef, LLVMValueRef V1,
                                    LLVMValueRef V2, LLVMValueRef Mask,
                                    /*const*/ char *Name);
LLVMValueRef LLVMBuildExtractValue(LLVMBuilderRef, LLVMValueRef AggVal,
                                   uint Index, /*const*/ char *Name);
LLVMValueRef LLVMBuildInsertValue(LLVMBuilderRef, LLVMValueRef AggVal,
                                  LLVMValueRef EltVal, uint Index,
                                  /*const*/ char *Name);


/*===-- Module providers --------------------------------------------------===*/

/* Encapsulates the module M in a module provider, taking ownership of the
 * module.
 * See the constructor llvm::ExistingModuleProvider::ExistingModuleProvider.
 */
LLVMModuleProviderRef
LLVMCreateModuleProviderForExistingModule(LLVMModuleRef M);

/* Destroys the module provider MP as well as the contained module.
 * See the destructor llvm::ModuleProvider::~ModuleProvider.
 */
void LLVMDisposeModuleProvider(LLVMModuleProviderRef MP);


/*===-- Memory buffers ----------------------------------------------------===*/

int LLVMCreateMemoryBufferWithContentsOfFile(/*const*/ char *Path,
                                             LLVMMemoryBufferRef *OutMemBuf,
                                             char **OutMessage);
int LLVMCreateMemoryBufferWithSTDIN(LLVMMemoryBufferRef *OutMemBuf,
                                    char **OutMessage);
void LLVMDisposeMemoryBuffer(LLVMMemoryBufferRef MemBuf);

/*===-- Pass Managers -----------------------------------------------------===*/

/** Constructs a new whole-module pass pipeline. This type of pipeline is
    suitable for link-time optimization and whole-module transformations.
    See llvm::PassManager::PassManager. */
LLVMPassManagerRef LLVMCreatePassManager();

/** Constructs a new function-by-function pass pipeline over the module
    provider. It does not take ownership of the module provider. This type of
    pipeline is suitable for code generation and JIT compilation tasks.
    See llvm::FunctionPassManager::FunctionPassManager. */
LLVMPassManagerRef LLVMCreateFunctionPassManager(LLVMModuleProviderRef MP);

/** Initializes, executes on the provided module, and finalizes all of the
    passes scheduled in the pass manager. Returns 1 if any of the passes
    modified the module, 0 otherwise. See llvm::PassManager::run(Module&). */
int LLVMRunPassManager(LLVMPassManagerRef PM, LLVMModuleRef M);

/** Initializes all of the function passes scheduled in the function pass
    manager. Returns 1 if any of the passes modified the module, 0 otherwise.
    See llvm::FunctionPassManager::doInitialization. */
int LLVMInitializeFunctionPassManager(LLVMPassManagerRef FPM);

/** Executes all of the function passes scheduled in the function pass manager
    on the provided function. Returns 1 if any of the passes modified the
    function, false otherwise.
    See llvm::FunctionPassManager::run(Function&). */
int LLVMRunFunctionPassManager(LLVMPassManagerRef FPM, LLVMValueRef F);

/** Finalizes all of the function passes scheduled in in the function pass
    manager. Returns 1 if any of the passes modified the module, 0 otherwise.
    See llvm::FunctionPassManager::doFinalization. */
int LLVMFinalizeFunctionPassManager(LLVMPassManagerRef FPM);

/** Frees the memory of a pass pipeline. For function pipelines, does not free
    the module provider.
    See llvm::PassManagerBase::~PassManagerBase. */
void LLVMDisposePassManager(LLVMPassManagerRef PM);
