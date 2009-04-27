// Written in the D programming language by Tomas Lindquist Olsen 2008
// Binding of llvm.c.Core builder for D.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
module llvm.builder;

import llvm.c.Core;
import llvm.c.Ext;

import llvm.llvm;
import llvm.util;

private
{
    template Build_NoArgs_Mixin(char[] N) {
       const Build_NoArgs_Mixin = "
       Value build"~N~"() {
           return new Value(LLVMBuild"~N~"(builder));
       }
       ";
    }

    template Build_Mixin(char[] NAME, char[] ARGS, char[] VALUES) {
       const Build_Mixin = "
       Value build"~NAME~"("~ARGS~") {
           return new Value(LLVMBuild"~NAME~"(builder, "~VALUES~"));
       }
       ";
       static assert(ARGS != "");
       static assert(VALUES != "");
    }

    // unnamed
    template Build_Value(char[] N) {
       const Build_Value = Build_Mixin!(N, "Value v", `v.value`);
    }
    template Build_Value_Value(char[] N) {
       const Build_Value_Value = Build_Mixin!(N, "Value v, Value w", `v.value, w.value`);
    }
    template Build_BB(char[] N) {
       const Build_BB = Build_Mixin!(N, "BasicBlock b", `b.bb`);
    }
    template Build_Value_BB_BB(char[] N) {
       const Build_Value_BB_BB = Build_Mixin!(N, "Value v, BasicBlock b1, BasicBlock b2", `v.value, b1.bb, b2.bb`);
    }
    template Build_Value_BB_uint(char[] N) {
       const Build_Value_BB_uint = Build_Mixin!(N, "Value v, BasicBlock b, uint n", `v.value, b.bb, n`);
    }

    // named
    template Build_Named_Mixin(char[] NAME, char[] ARGS, char[] VALUES) {
       const Build_Named_Mixin = Build_Mixin!(NAME, ARGS~", char[] name", VALUES~`, to_stringz(name)`);
    }

    template Build_Type_Name(char[] N) {
       const Build_Type_Name = Build_Named_Mixin!(N, "Type t", `t.ll`);
    }
    template Build_Value_Name(char[] N) {
       const Build_Value_Name = Build_Named_Mixin!(N, "Value v", `v.value`);
    }
    template Build_Type_Value_Name(char[] N) {
       const Build_Type_Value_Name = Build_Named_Mixin!(N, "Type t, Value v", `t.ll, v.value`);
    }
    template Build_Value_Type_Name(char[] N) {
       const Build_Value_Type_Name = Build_Named_Mixin!(N, "Value v, Type t", `v.value, t.ll`);
    }
    template Build_Value_Value_Name(char[] N) {
       const Build_Value_Value_Name = Build_Named_Mixin!(N, "Value a, Value b", `a.value, b.value`);
    }
    template Build_Value_Value_Value_Name(char[] N) {
       const Build_Value_Value_Value_Name = Build_Named_Mixin!(N, "Value a, Value b, Value c", `a.value, b.value, c.value`);
    }
    template Build_Value_uint_Name(char[] N) {
        const Build_Value_uint_Name = Build_Named_Mixin!(N, "Value a, uint n", `a.value, n`);
    }
    template Build_Value_Value_uint_Name(char[] N) {
       const Build_Value_Value_uint_Name = Build_Named_Mixin!(N, "Value a, Value b, uint n", `a.value, b.value, n`);
    }
    template Build_Cmp(char[] PRED, char[] N) {
       const Build_Cmp = Build_Named_Mixin!(N, ""~PRED~"Predicate p, Value l, Value r", `p, l.value, r.value`);
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
class Builder
{
    ///
    private LLVMBuilderRef builder;
    ///
    this()
    {
        builder = LLVMCreateBuilder();
    }
    ///
    void dispose()
    {
        LLVMDisposeBuilder(builder);
        builder = null;
    }
    ///
    ~this()
    {
        // safe because builder isn't on the GC heap and isn't exposed.
        dispose();
    }
    ///
    void positionBefore(Value v)
    {
        assert(builder !is null);
        LLVMPositionBuilderBefore(builder, v.value);
    }
    ///
    void positionAtEnd(BasicBlock bb)
    {
        assert(builder !is null);
        LLVMPositionBuilderAtEnd(builder, bb.bb);
    }
    ///
    void positionAtStart(BasicBlock bb)
    {
        assert(builder !is null);
        LLVMPositionBuilderBefore(builder, LLVMGetFirstInstruction(bb.bb));
    }
    ///
    BasicBlock getInsertBlock()
    {
        return new BasicBlock(LLVMGetInsertBlock(builder));
    }

    ///
    mixin(StringDistribute!(Build_NoArgs_Mixin, "RetVoid", "Unwind", "Unreachable"));
    mixin(Build_BB!("Br"));
    mixin(Build_Value_BB_BB!("CondBr"));
    mixin(Build_Value_BB_uint!("Switch"));
    ///
    mixin(StringDistribute!(Build_Value, "Ret", "Free"));
    ///
    mixin(Build_Value_Value!("Store"));
    ///
    mixin(StringDistribute!(Build_Value_Value_Name,
        "Add","Sub","Mul","UDiv","SDiv","FDiv","URem","SRem","FRem",
        "Shl","LShr","AShr","And","Or","Xor",
        "ExtractElement"
    ));
    ///
    mixin(StringDistribute!(Build_Value_Name, "Neg","Not", "Load"));
    ///
    mixin(StringDistribute!(Build_Value_Type_Name,
        "Trunc","SExt","ZExt","FPTrunc","FPExt",
        "UIToFP","SIToFP","FPToUI","FPToSI",
        "PtrToInt","IntToPtr","BitCast",
        "VAArg"
    ));
    ///
    mixin(Build_Cmp!("Int","ICmp"));
    ///
    mixin(Build_Cmp!("Real","FCmp"));
    ///
    mixin(StringDistribute!(Build_Type_Name,
        "Phi", "Malloc", "Alloca"
    ));
    ///
    mixin(StringDistribute!(Build_Type_Value_Name,
        "ArrayMalloc", "ArrayAlloca"
    ));
    ///
    mixin(StringDistribute!(Build_Value_Value_Value_Name,
        "Select", "InsertElement", "ShuffleVector"
    ));
    ///
    mixin(Build_Value_uint_Name!("ExtractValue"));
    ///
    mixin(Build_Value_Value_uint_Name!("InsertValue"));
    ///
    Value buildCall(Value fn, Value[] args, char[] name) {
        auto llargs = new LLVMValueRef[args.length];
        foreach(i,a; args) llargs[i] = a.value;
        return new Value(LLVMBuildCall(builder, fn.value, llargs.ptr, llargs.length, to_stringz(name)));
    }
    ///
    Value buildGEP(Value ptr, Value[] indices, char[] name) {
        auto llindices = new LLVMValueRef[indices.length];
        foreach(i,idx; indices) llindices[i] = idx.value;
        return new Value(LLVMBuildGEP(builder, ptr.value, llindices.ptr, llindices.length, to_stringz(name)));
    }
    ///
    Value buildInvoke(Value fn, Value[] args, BasicBlock thenbb, BasicBlock catchbb, char[] name) {
        auto llargs = new LLVMValueRef[args.length];
        foreach(i,a; args) llargs[i] = a.value;
        return new Value(LLVMBuildInvoke(builder, fn.value, llargs.ptr, llargs.length, thenbb.bb, catchbb.bb, to_stringz(name)));
    }
}
