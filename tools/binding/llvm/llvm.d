// Written in the D programming language by Tomas Lindquist Olsen 2008
// Binding of llvm.c.Core values for D.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
module llvm.llvm;

import llvm.c.Core;
import llvm.c.Ext;
import llvm.c.BitWriter;
import llvm.c.BitReader;
import llvm.c.Analysis;
import llvm.c.Target;

public import llvm.type;
public import llvm.builder;

import llvm.util;

///
class LLVMException : Exception
{
    this(char[] msg) {
        super(msg);
    }
}

version(Tango) {
    import tango.stdc.stdlib;
}
else {
    import std.c.stdlib;
}
///
alias LLVMLinkage Linkage;
///
alias LLVMIntPredicate IntPredicate;
///
alias LLVMRealPredicate RealPredicate;
///
alias LLVMCallConv CallConv;
///
alias LLVMVisibility Visibility;
///
alias LLVMValueKind ValueKind;

///
class Module
{
    /// global registry for 1:1 mapping of ModuleRef's -> Module's
    private static Module[LLVMModuleRef] registry;
    ///
    private LLVMModuleRef mod;
    const char[] name;
    
    // Make all methods final to enable linking with just needed libs.
    // To make use of this if compiling with GDC: use -ffunction-sections when
    // compiling and --gc-sections when linking.
    // (Final avoids references in the vtable)
    final:
    
    ///
    this(char[] nam)
    {
        name = nam;
        mod = LLVMModuleCreateWithName(to_stringz(nam));
        registry[mod] = this;
    }
    ///
    private this(LLVMModuleRef m)
    {
        name = null;
        mod = m;
        registry[m] = this;
    }
    ///
    static package Module GetExisting(LLVMModuleRef m)
    {
        if (auto p = m in registry)
        {
            return *p;
        }
        return new Module(m);
    }
    /// Create a module from bitcode. Returns the Module on success, null on failure.
    static Module GetFromBitcode(char[] bitcodepath, ref char[] errmsg)
    {
        LLVMModuleRef mref;
        LLVMMemoryBufferRef bref;
        char* msg;
        if (LLVMCreateMemoryBufferWithContentsOfFile(to_stringz(bitcodepath), &bref, &msg))
        {
            errmsg = from_stringz(msg).dup;
            LLVMDisposeMessage(msg);
            if (errmsg.length == 0)
                errmsg = "Error reading bitcode file";
            throw new LLVMException(errmsg);
        }
        scope(exit)
            LLVMDisposeMemoryBuffer(bref);
        
        if (LLVMParseBitcode(bref, &mref, &msg))
        {
            errmsg = from_stringz(msg).dup;
            LLVMDisposeMessage(msg);
            if (errmsg.length == 0)
                errmsg = "Error parsing bitcode";
            LLVMDisposeMemoryBuffer(bref);
            throw new LLVMException(errmsg);
        }
        return new Module(mref);
    }
    /// important to call this when done
    void dispose()
    {
        if (mod)
        {
            registry.remove(mod);
            LLVMDisposeModule(mod);
            mod = null;
        }
    }
    ///
    char[] dataLayout()
    {
        assert(mod !is null);
        return from_stringz(LLVMGetDataLayout(mod));
    }
    ///
    void dataLayout(char[] dl)
    {
        assert(mod !is null);
        LLVMSetDataLayout(mod, to_stringz(dl));
    }
    ///
    char[] target()
    {
        assert(mod !is null);
        return from_stringz(LLVMGetTarget(mod));
    }
    ///
    void target(char[] dl)
    {
        assert(mod !is null);
        LLVMSetTarget(mod, to_stringz(dl));
    }
    ///
    bool addTypeName(char[] nam, Type t)
    {
        assert(mod !is null);
        return LLVMAddTypeName(mod, to_stringz(nam), t.ll) != 0;
    }
    ///
    Type getTypeByName(char[] name) {
        return getTypeOf(LLVMGetTypeByName(mod, to_stringz(name)));
    }
    ///
    void deleteTypeName(char[] nam)
    {
        assert(mod !is null);
        LLVMDeleteTypeName(mod, to_stringz(nam));
    }
    ///
    GlobalVariable addGlobal(Type t, char[] nam)
    {
        assert(mod !is null);
        auto c = LLVMAddGlobal(mod, t.ll, to_stringz(nam));
        assert(c !is null);
        return new GlobalVariable(c, getTypeOf(c));
    }
    /// Convenience method, type is taken to be that of the initializer
    GlobalVariable addGlobal(Constant initializer, char[] name)
    {
        auto global = addGlobal(initializer.type, name);
        global.initializer = initializer;
        return global;
    }
    ///
    GlobalValue getNamedGlobal(char[] nam)
    {
        assert(mod !is null);
        auto c = LLVMGetNamedGlobal(mod, to_stringz(nam));
        if (c is null) return null;
        return cast(GlobalValue)getValueOf(c);
    }
    ///
    Function addFunction(Type t, char[] nam)
    {
        assert(mod !is null);
        auto c = LLVMAddFunction(mod, to_stringz(nam), t.ll);
        assert(c !is null);
        return new Function(c, getTypeOf(c));
    }
    ///
    Function getNamedFunction(char[] nam)
    {
        assert(mod !is null);
        auto c = LLVMGetNamedFunction(mod, to_stringz(nam));
        if (c is null) return null;
        return cast(Function)getValueOf(c);
    }
    ///
    Function getOrInsertFunction(Type t, char[] nam)
    {
        assert(mod !is null);
        auto c = LLVMGetOrInsertFunction(mod, to_stringz(nam), t.ll);
        auto val = getValueOf(c);
        auto fn = cast(Function) val;
        // Can happen if 'nam' names a function of a different type:
        assert(fn !is null, "Not a function of type " ~ t.toString() ~ ": " ~ val.toString());
        return fn;
    }
    /// Performs the same optimizations as `opt -std-compile-opts ...' would on the module.
    /// If inline is true, function inlining will be performed.
    void optimize(bool inline)
    {
        LLVMOptimizeModule(mod, inline);
    }
    /// Writes the module to an open file descriptor. Returns true on success.
    bool writeBitcodeToFileHandle(int handle)
    {
        return (LLVMWriteBitcodeToFileHandle(mod, handle) == 0);
    }
    /// Writes the module to the specified path. Returns 0 on success.
    bool writeBitcodeToFile(char[] path)
    {
        return (LLVMWriteBitcodeToFile(mod, to_stringz(path)) == 0);
    }
    /// Throws an exception if the module doesn't pass the LLVM verifier.
    void verify()
    {
        char* msg;
        if (LLVMVerifyModule(mod, LLVMVerifierFailureAction.ReturnStatus, &msg))
        {
            auto errmsg = from_stringz(msg).dup;
            LLVMDisposeMessage(msg);
            if (errmsg.length == 0)
                errmsg = "Module verification failed";
            throw new LLVMException(errmsg);
        }
    }
}

class ModuleProvider
{
    ///
    private LLVMModuleProviderRef mp;
    ///
    private this(LLVMModuleProviderRef mp)
    {
        this.mp = mp;
    }
    /// Takes ownership of module, returns a ModuleProvider for it.
    static ModuleProvider GetForModule(Module m)
    {
        auto mp = LLVMCreateModuleProviderForExistingModule(m.mod);
        return new ModuleProvider(mp);
    }
    /// Destroys the provided module, unless this MP was passed to an ExecutionEngine.
    void dispose()
    {
        LLVMDisposeModuleProvider(mp);
        mp = null;
    }
    /// Returns a lazily-deserializing ModuleProvider
    static ModuleProvider GetFromBitcode(char[] filename)
    {
        LLVMMemoryBufferRef buf;
        char* msg;
        if (LLVMCreateMemoryBufferWithContentsOfFile(to_stringz(filename), &buf, &msg))
        {
            auto errmsg = from_stringz(msg).dup;
            LLVMDisposeMessage(msg);
            if (errmsg.length == 0)
                errmsg = "ModuleProvider: Error reading bitcode file";
            throw new LLVMException(errmsg);
        }
        
        LLVMModuleProviderRef mp;
        // Takes ownership of buffer ...
        if (LLVMGetBitcodeModuleProvider(buf, &mp, &msg))
        {
            // ... unless it fails, in which case we need to clean it up ourselves
            LLVMDisposeMemoryBuffer(buf);
            
            auto errmsg = from_stringz(msg).dup;
            LLVMDisposeMessage(msg);
            if (errmsg.length == 0)
                errmsg = "Error creating ModuleProvider for bitcode file";
            throw new LLVMException(errmsg);
        }
        return new ModuleProvider(mp);
    }
    ///
    package LLVMModuleProviderRef ll()
    {
        return mp;
    }
}

///
class Value
{
    ///
    const LLVMValueRef value;
    ///
    const Type type;
    ///
    this(LLVMValueRef v, Type t=null) {
        value = v;
        if (t is null) t = getTypeOf(v);
        type = t;
    }
    ///
    char[] toString() {
        auto cstr = LLVMValueToString(value);
        auto result = from_stringz(cstr).dup;
        free(cstr);
        return result;
    }
    ///
    ValueKind kind()
    {
        return LLVMGetValueKind(value);
    }
    ///
    char[] name()
    {
        return from_stringz(LLVMGetValueName(value));
    }
    ///
    void name(char[] s)
    {
        LLVMSetValueName(value, to_stringz(s));
    }
    ///
    void dump() {
        LLVMDumpValue(value);
    }
    ///
    bool isConstant()
    {
        return LLVMIsConstant(value) != 0;
    }
    ///
    int opEquals(Object o)
    {
        auto v = cast(Value)o;
        if (v is null) return 0;
        if (value is v.value)
            return 1;
        return 0;
    }
    /// invalidates object
    void eraseFromParent()
    {
        LLVMEraseFromParent(value);
    }
    /// ditto
    void replaceAllUsesWith(Value newval)
    {
        LLVMReplaceAllUsesWith(value, newval.value);
    }

    /// only for call's
    void callConv(uint CC)
    {
        LLVMSetInstructionCallConv(value, CC);
    }
    /// ditto
    uint callConv()
    {
        return LLVMGetInstructionCallConv(value);
    }

    /// only for phi's
    void addIncoming(Value[] inValues, BasicBlock[] inBlocks)
    {
        auto n = inValues.length;
        assert(n == inBlocks.length);
        auto v = new LLVMValueRef[n];
        auto b = new LLVMBasicBlockRef[n];
        for (size_t i=0; i<n; i++) {
            v[i] = inValues[i].value;
            b[i] = inBlocks[i].bb;
        }
        LLVMAddIncoming(value, v.ptr, b.ptr, n);
    }
    /// ditto
    uint numIncoming()
    {
        return LLVMCountIncoming(value);
    }
    /// ditto
    Value getIncomingValue(uint index)
    {
        return getValueOf(LLVMGetIncomingValue(value, index));
    }
    /// ditto
    BasicBlock getIncomingBlock(uint index)
    {
        // TODO bb's should be unique as well
        return new BasicBlock(LLVMGetIncomingBlock(value, index));
    }

    /// only for switch's
    void addCase(Value onval, BasicBlock b)
    {
        LLVMAddCase(value, onval.value, b.bb);
    }
}

///
Value getValueOf(LLVMValueRef v)
{
    auto kind = LLVMGetValueKind(v);
    switch(kind)
    {
    case ValueKind.Argument:
    case ValueKind.InlineAsm:
    case ValueKind.Instruction:
        return new Value(v);

    case ValueKind.Function:
        return new Function(v, getTypeOf(v));

    case ValueKind.GlobalVariable:
        return new GlobalVariable(v, getTypeOf(v));

    case ValueKind.GlobalAlias:
    case ValueKind.UndefValue:
    case ValueKind.ConstantExpr:
    case ValueKind.ConstantAggregateZero:
    case ValueKind.ConstantPointerNull:
        return new Constant(v, getTypeOf(v));

    case ValueKind.ConstantInt:
        return new ConstantInt(v, getTypeOf(v));

    case ValueKind.ConstantFP:
        return new ConstantReal(v, getTypeOf(v));

    case ValueKind.ConstantArray:
        return new ConstantArray(v, getTypeOf(v));

    case ValueKind.ConstantStruct:
        return new ConstantStruct(v, getTypeOf(v));

    case ValueKind.ConstantVector:
        return new ConstantVector(v, getTypeOf(v));

    case ValueKind.BasicBlock:
    default:
        assert(0);
    }
}

private
{
    template GenericConstUnaOp(char[] N)
    {
        const GenericConstUnaOp =
        "Constant Get"~N~"(Constant v) {
            auto c = LLVMConst"~N~"(v.value);
            return cast(Constant)getValueOf(c);
        }";
    }

    template GenericConstBinOp(char[] N)
    {
        const GenericConstBinOp =
        "Constant Get"~N~"(Constant l, Constant r) {
            auto c = LLVMConst"~N~"(l.value, r.value);
            return cast(Constant)getValueOf(c);
        }";
    }

    template GenericConstTriOp(char[] N)
    {
        const GenericConstTriOp =
        "Constant Get"~N~"(Constant s, Constant t, Constant u) {
            auto c = LLVMConst"~N~"(s.value, t.value, u.value);
            return cast(Constant)getValueOf(c);
        }";
    }

    template GenericConstCast(char[] N)
    {
        const GenericConstCast =
        "Constant Get"~N~"(Constant v, Type t) {
            auto c = LLVMConst"~N~"(v.value, t.ll);
            return cast(Constant)getValueOf(c);
        }";
    }

    template GenericConstCmp(char[] PRED, char[] N)
    {
        const GenericConstCmp =
        "Constant Get"~N~"("~PRED~"Predicate p, Constant l, Constant r) {
            auto c = LLVMConst"~N~"(p, l.value, r.value);
            return cast(Constant)getValueOf(c);
        }";
    }

    template StringDistribute(alias T, U...)
    {
        static if (!U.length)
            const char[] StringDistribute="";
        else
            const char[] StringDistribute = T!(U[0]) ~ StringDistribute!(T, U[1..$]);
    }
}

///
class Constant : Value
{
    ///
    protected this(LLVMValueRef v, Type t)
    {
        super(v,t);
    }

    ///
    static Constant GetNull(Type t)
    {
        return cast(Constant)getValueOf(LLVMConstNull(t.ll));
    }
    /// only for int/vector
    static Constant GetAllOnes(Type t)
    {
        return cast(Constant)getValueOf(LLVMConstAllOnes(t.ll));
    }
    ///
    static Constant GetUndef(Type t)
    {
        return cast(Constant)getValueOf(LLVMGetUndef(t.ll));
    }
    ///
    static ConstantInt GetTrue()
    {
        return ConstantInt.GetU(Type.Int1, 1);
    }
    ///
    static ConstantInt GetFalse()
    {
        return ConstantInt.GetU(Type.Int1, 0);
    }

    ///
    bool isNull()
    {
        return LLVMIsNull(value) != 0;
    }
    ///
    bool isUndef()
    {
        return LLVMIsUndef(value) != 0;
    }

    static
    {
        ///
        mixin(StringDistribute!(GenericConstUnaOp,
            "Neg","Not"
        ));
        ///
        mixin(StringDistribute!(GenericConstBinOp,
            "Add","Sub","Mul","UDiv","SDiv","FDiv","URem","SRem","FRem",
            "And","Or","Xor","Shl","LShr","AShr",
            "ExtractElement"
        ));
        ///
        mixin(StringDistribute!(GenericConstCast,
            "Trunc","SExt","ZExt","FPTrunc","FPExt",
            "UIToFP","SIToFP","FPToUI","FPToSI",
            "PtrToInt","IntToPtr","BitCast"
        ));
        ///
        mixin(StringDistribute!(GenericConstTriOp,
            "Select",
            "InsertElement",
            "ShuffleVector"
        ));
        ///
        mixin(GenericConstCmp!("Int","ICmp"));
        ///
        mixin(GenericConstCmp!("Real","FCmp"));
        ///
        Constant GetGEP(Constant ptr, Constant[] idxs...)
        {
            static if (size_t.max > uint.max) {
                assert(idxs.length <= uint.max, "Ridiculous number of indexes to GEP");
            }
            auto ar = new LLVMValueRef[idxs.length];
            foreach(i,v; idxs) ar[i] = v.value;
            auto c = LLVMConstGEP(ptr.value, ar.ptr, ar.length);
            return cast(Constant)getValueOf(c);
        }
        ///
        Constant GetExtractValue(Constant agg, uint[] idxs...) {
            static if (size_t.max > uint.max) {
                assert(idxs.length <= uint.max, "Ridiculous number of indexes to ExtractValue");
            }
            auto c = LLVMConstExtractValue(agg.value, idxs.ptr, idxs.length);
            return cast(Constant)getValueOf(c);
        }
        ///
        Constant GetInsertValue(Constant agg, Constant elt, uint[] idxs...) {
            static if (size_t.max > uint.max) {
                assert(idxs.length <= uint.max, "Ridiculous number of indexes to InsertValue");
            }
            auto c = LLVMConstInsertValue(agg.value, elt.value, idxs.ptr, idxs.length);
            return cast(Constant)getValueOf(c);
        }
        ///
        Constant GetSizeOf(Type t)
        {
            return cast(Constant)getValueOf(LLVMSizeOf(t.ll));
        }
    }
}

///
abstract class ScalarConstant : Constant
{
    ///
    protected this(LLVMValueRef v, Type t)
    {
        super(v, t);
    }
}

///
class ConstantInt : ScalarConstant
{
    ///
    private this(LLVMValueRef v, Type t)
    {
        super(v, t);
    }
    ///
    static ConstantInt Get(Type t, ulong N, bool signExt)
    {
        auto c = LLVMConstInt(t.ll, N, signExt);
        return new ConstantInt(c, t);
    }
    ///
    static ConstantInt GetS(Type t, long N)
    {
        return Get(t, cast(ulong)N, true);
    }
    ///
    static ConstantInt GetU(Type t, ulong N)
    {
        return Get(t, N, false);
    }
}

///
class ConstantReal : ScalarConstant
{
    ///
    private this(LLVMValueRef v, Type t)
    {
        super(v, t);
    }
    ///
    static ConstantReal Get(Type t, real N)
    {
        auto c = LLVMConstReal(t.ll, N);
        return new ConstantReal(c, t);
    }
}

///
abstract class CompositeConstant : Constant
{
    ///
    protected this(LLVMValueRef v, Type t)
    {
        super(v, t);
    }
}

///
class ConstantArray : CompositeConstant
{
    ///
    private this(LLVMValueRef v, Type t)
    {
        super(v, t);
    }
    ///
    static ConstantArray Get(Type eltty, Constant[] vals)
    {
        auto p = new LLVMValueRef[vals.length];
        foreach(i,v; vals) p[i] = v.value;
        auto c = LLVMConstArray(eltty.ll, p.ptr, p.length);
        return new ConstantArray(c, getTypeOf(c));
    }
    ///
    static ConstantArray GetString(char[] str, bool nullterm)
    {
        auto len = str.length + nullterm;
        auto c = LLVMConstString(str.ptr, str.length, !nullterm);
        return new ConstantArray(c, getTypeOf(c));
    }
}

///
class ConstantStruct : CompositeConstant
{
    ///
    private this(LLVMValueRef v, Type t)
    {
        super(v, t);
    }
    ///
    static ConstantStruct Get(Constant[] vals, bool packed=false)
    {
        auto p = new LLVMValueRef[vals.length];
        foreach(i,v; vals) p[i] = v.value;
        auto c = LLVMConstStruct(p.ptr, p.length, packed);
        return new ConstantStruct(c, getTypeOf(c));
    }
}

///
class ConstantVector : CompositeConstant
{
    ///
    private this(LLVMValueRef v, Type t)
    {
        super(v, t);
    }
    ///
    static ConstantVector Get(ScalarConstant[] vals)
    {
        auto p = new LLVMValueRef[vals.length];
        foreach(i,v; vals) p[i] = v.value;
        auto c = LLVMConstVector(p.ptr, p.length);
        return new ConstantVector(c, getTypeOf(c));
    }
}

///
abstract class GlobalValue : Constant
{
    ///
    private this(LLVMValueRef v, Type t) {
        super(v, t);
    }
    ///
    bool isDeclaration()
    {
        return LLVMIsDeclaration(value) != 0;
    }
    ///
    Linkage linkage()
    {
        return LLVMGetLinkage(value);
    }
    ///
    void linkage(Linkage l)
    {
        LLVMSetLinkage(value, l);
    }
    ///
    char[] section()
    {
        return from_stringz(LLVMGetSection(value));
    }
    ///
    void section(char[] s)
    {
        LLVMSetSection(value, to_stringz(s));
    }
    ///
    Visibility visibility()
    {
        return LLVMGetVisibility(value);
    }
    ///
    void visibility(Visibility v)
    {
        LLVMSetVisibility(value, v);
    }
    ///
    uint alignment()
    {
        return LLVMGetAlignment(value);
    }
    ///
    void alignment(uint bytes)
    {
        LLVMSetAlignment(value, bytes);
    }
}

///
class GlobalVariable : GlobalValue
{
    /// TODO: void DeleteGlobal(ValueRef GlobalVar);

    ///
    private this(LLVMValueRef v, Type t) {
        super(v, t);
    }
    ///
    bool hasInitializer()
    {
        return isDeclaration() == 0;
    }
    ///
    Constant initializer()
    {
        auto c = LLVMGetInitializer(value);
        if (c is null) return null;
        return cast(Constant)getValueOf(c);
    }
    ///
    void initializer(Constant c)
    {
        LLVMSetInitializer(value, c.value);
    }
    ///
    bool threadLocal()
    {
        return LLVMIsThreadLocal(value) != 0;
    }
    ///
    void threadLocal(bool b)
    {
        LLVMSetThreadLocal(value, b);
    }
    ///
    bool globalConstant()
    {
        return LLVMIsGlobalConstant(value) != 0;
    }
    ///
    void globalConstant(bool b)
    {
        LLVMSetGlobalConstant(value, b);
    }
}

///
class Function : GlobalValue
{
    /// TODO: void GetParams(ValueRef Fn, ValueRef *Params);
    /// TODO: void GetBasicBlocks(ValueRef Fn, BasicBlockRef *BasicBlocks);

    ///
    package this(LLVMValueRef v, Type t) {
        super(v, t);
    }
    ///
    void eraseFromParent()
    {
        LLVMDeleteFunction(value);
    }
    ///
    uint numParams()
    {
        return LLVMCountParams(value);
    }
    ///
    Value getParam(uint idx)
    {
        auto v = LLVMGetParam(value, idx);
        assert(v !is null);
        return getValueOf(v);
    }
    ///
    uint intrinsicID()
    {
        return LLVMGetIntrinsicID(value);
    }
    ///
    uint callConv()
    {
        return LLVMGetFunctionCallConv(value);
    }
    ///
    void callConv(uint cc)
    {
        LLVMSetFunctionCallConv(value, cc);
    }
    ///
    char[] gc()
    {
        return from_stringz(LLVMGetGC(value));
    }
    ///
    void gc(char[] name)
    {
        LLVMSetGC(value, to_stringz(name));
    }
    ///
    uint numBasicBlocks()
    {
        return LLVMCountBasicBlocks(value);
    }
    ///
    static BasicBlock InsertBasicBlock(BasicBlock bb, char[] name)
    {
        auto b = LLVMInsertBasicBlock(bb.bb, to_stringz(name));
        assert(b !is null);
        return new BasicBlock(b);
    }
    ///
    BasicBlock appendBasicBlock(char[] name)
    {
        auto b = LLVMAppendBasicBlock(value, to_stringz(name));
        assert(b !is null);
        return new BasicBlock(b);
    }
    ///
    BasicBlock getEntryBasicBlock()
    {
        auto b = LLVMGetEntryBasicBlock(value);
        if (b is null) return null;
        return new BasicBlock(b);
    }
    /// Throws an exception if the function doesn't pass the LLVM verifier.
    void verify()
    {
        if (LLVMVerifyFunction(value, LLVMVerifierFailureAction.ReturnStatus))
        {
            auto exceptionmsg = "Function failed to verify (" ~ name ~ ")";
            throw new LLVMException(exceptionmsg);
        }
    }
}

///
class BasicBlock
{
    ///
    LLVMBasicBlockRef bb;
    ///
    this(LLVMBasicBlockRef b)
    {
        assert(b !is null);
        bb = b;
    }
    ///
    this(Value v)
    {
        assert(LLVMValueIsBasicBlock(v.value));
        bb = LLVMValueAsBasicBlock(v.value);
    }
    ///
    override int opEquals(Object o) {
        auto block = cast(BasicBlock) o;
        if (!block)
            return false;
        return bb == block.bb;
    }
    ///
    void dispose()
    {
        LLVMDeleteBasicBlock(bb);
        bb = null;
    }
    ///
    Function getParent() {
        assert(bb !is null);
        auto func = LLVMGetBasicBlockParent(bb);
        if (!func) return null;
        return new Function(func, getTypeOf(func));
    }
    ///
    Value asValue()
    {
        assert(bb !is null);
        auto v = LLVMBasicBlockAsValue(bb);
        return new Value(v, Type.Label);
    }
    ///
    bool terminated()
    {
        assert(bb !is null);
        return (LLVMIsTerminated(bb) != 0);
    }
    ///
    bool hasPredecessors()
    {
        assert(bb !is null);
        return (LLVMHasPredecessors(bb) != 0);
    }
    ///
    bool empty()
    {
        assert(bb !is null);
        return (LLVMIsBasicBlockEmpty(bb) != 0);
    }
}

///
class TargetData
{
    ///
    private LLVMTargetDataRef target;
    ///
    private this(LLVMTargetDataRef td)
    {
        target = td;
    }
    ///
    static TargetData Get(char[] str)
    {
        return new TargetData(LLVMCreateTargetData(to_stringz(str)));
    }
    ///
    static TargetData Get(Module M)
    {
        return new TargetData(LLVMCreateTargetData(to_stringz(M.dataLayout)));
    }
    /// invalidates object
    void dispose()
    {
        LLVMDisposeTargetData(target);
        target = null;
    }
    ///
    size_t getABITypeSize(Type T)
    {
        return LLVMABISizeOfType(target, T.ll);
    }
}
