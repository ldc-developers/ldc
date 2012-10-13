#ifndef LDC_IR_IRFORW_H
#define LDC_IR_IRFORW_H

// dmd forward declarations
struct Module;
struct Dsymbol;
struct Declaration;
struct VarDeclaration;
struct FuncDeclaration;
struct AggregateDeclaration;
struct StructDeclaration;
struct ClassDeclaration;
struct InterfaceDeclaration;
struct Expression;
struct BaseClass;
struct Array;
struct Argument;

struct Type;
struct TypeStruct;
struct TypeClass;
struct TypeEnum;
struct TypeArray;
struct TypeFunction;

// llvm forward declarations
namespace llvm
{
    class Value;
    class GlobalValue;
    class GlobalVariable;
    class Function;
    class Constant;
    class ConstantStruct;
    class ConstantArray;
#if LDC_LLVM_VER >= 302
    class DataLayout;
#else
    class TargetData;
#endif
    class Type;
    class StructType;
    class ArrayType;
    class PointerType;
    class BasicBlock;
    class Instruction;
}

#endif
