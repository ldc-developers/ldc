#ifndef LDC_IR_IRFUNCTION_H
#define LDC_IR_IRFUNCTION_H

#include "gen/llvm.h"
#include "ir/ir.h"
#include "ir/irlandingpad.h"

#include <vector>
#include <stack>
#include <map>

struct ABIRewrite;

// represents a function type argument
// both explicit and implicit as well as return values
struct IrFuncTyArg : IrBase
{
    /** This is the original D type as the frontend knows it
     *  May NOT be rewritten!!! */
    Type* type;

    /// This is the final LLVM Type used for the parameter/return value type
    const llvm::Type* ltype;

    /** These are the final LLVM attributes used for the function.
     *  Must be valid for the LLVM Type and byref setting */
    unsigned attrs;

    /** 'true' if the final LLVM argument is a LLVM reference type.
     *  Must be true when the D Type is a value type, but the final
     *  LLVM Type is a reference type! */
    bool byref;

    /** Pointer to the ABIRewrite structure needed to rewrite LLVM ValueS
     *  to match the final LLVM Type when passing arguments and getting
     *  return values */
    ABIRewrite* rewrite;

    /// Helper to check if the 'inreg' attribute is set
    bool isInReg() const { return (attrs & llvm::Attribute::InReg) != 0; }
    /// Helper to check if the 'sret' attribute is set
    bool isSRet() const  { return (attrs & llvm::Attribute::StructRet) != 0; }
    /// Helper to check if the 'byval' attribute is set
    bool isByVal() const { return (attrs & llvm::Attribute::ByVal) != 0; }

    /** @param t D type of argument/return value as known by the frontend
     *  @param byref Initial value for the 'byref' field. If true the initial
     *               LLVM Type will be of DtoType(type->pointerTo()), instead
     *               of just DtoType(type) */
    IrFuncTyArg(Type* t, bool byref, unsigned a = 0);
};

// represents a function type
struct IrFuncTy : IrBase
{
    // return value
    IrFuncTyArg* ret;

    // null if not applicable
    IrFuncTyArg* arg_sret;
    IrFuncTyArg* arg_this;
    IrFuncTyArg* arg_nest;
    IrFuncTyArg* arg_arguments;
    IrFuncTyArg* arg_argptr;

    // normal explicit arguments
    typedef LLSmallVector<IrFuncTyArg*, 4> ArgList;
    typedef ArgList::iterator ArgIter;
    ArgList args;

    // C varargs
    bool c_vararg;

    // range of normal parameters to reverse
    bool reverseParams;

    IrFuncTy()
    :   ret(NULL),
        arg_sret(NULL),
        arg_this(NULL),
        arg_nest(NULL),
        arg_arguments(NULL),
        arg_argptr(NULL),
        c_vararg(false),
        reverseParams(false)
    {}

    llvm::Value* putRet(Type* dty, DValue* dval);
    llvm::Value* getRet(Type* dty, DValue* dval);

    llvm::Value* putParam(Type* dty, int idx, DValue* dval);
    llvm::Value* getParam(Type* dty, int idx, DValue* dval);
    void getParam(Type* dty, int idx, DValue* dval, llvm::Value* lval);
};

// represents a function
struct IrFunction : IrBase
{
    llvm::Function* func;
    llvm::Instruction* allocapoint;
    FuncDeclaration* decl;
    TypeFunction* type;

    bool queued;
    bool defined;
    
    llvm::Value* retArg; // return in ptr arg
    llvm::Value* thisArg; // class/struct 'this' arg
    llvm::Value* nestArg; // nested function 'this' arg
    
    llvm::Value* nestedVar; // nested var alloca
    
    llvm::Value* _arguments;
    llvm::Value* _argptr;
    
    llvm::DISubprogram diSubprogram;

    // pushes a unique label scope of the given name
    void pushUniqueLabelScope(const char* name);
    // pops a label scope
    void popLabelScope();

    // gets the string under which the label's BB
    // is stored in the labelToBB map.
    // essentially prefixes ident by the strings in labelScopes
    std::string getScopedLabelName(const char* ident);

    // label to basic block lookup
    typedef std::map<std::string, llvm::BasicBlock*> LabelToBBMap;
    LabelToBBMap labelToBB;

    // landing pads for try statements
    IRLandingPad landingPad;

    IrFunction(FuncDeclaration* fd);

    // annotations
    void setNeverInline();
    void setAlwaysInline();

private:
    // prefix for labels and gotos
    // used for allowing labels to be emitted twice
    std::vector<std::string> labelScopes;

    // next unique id stack
    std::stack<int> nextUnique;
};

#endif
