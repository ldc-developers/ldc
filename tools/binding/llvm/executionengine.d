// Written in the D programming language by Frits van Bommel 2008
// Binding of llvm.c.ExecutionEngine for D.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
module llvm.executionengine;

import llvm.c.Core;
import llvm.c.ExecutionEngine;

import llvm.llvm;
import llvm.util;

///
class GenericValue
{
    ///
    private LLVMGenericValueRef value;
    ///
    private this(LLVMGenericValueRef v)
    {
        value = v;
    }
    ///
    void dispose()
    {
        LLVMDisposeGenericValue(value);
        value = null;
    }
    ///
    ~this()
    {
        dispose();  // safe because value isn't on the GC heap and isn't exposed.
    }
    ///
    static GenericValue GetS(IntegerType ty, long N)
    {
        return new GenericValue(LLVMCreateGenericValueOfInt(ty.ll, N, true));
    }
    ///
    static GenericValue GetU(IntegerType ty, ulong N)
    {
        return new GenericValue(LLVMCreateGenericValueOfInt(ty.ll, N, false));
    }
    ///
    static GenericValue GetP(void* P)
    {
        return new GenericValue(LLVMCreateGenericValueOfPointer(P));
    }
    ///
    static GenericValue GetF(RealType ty, double N)
    {
        return new GenericValue(LLVMCreateGenericValueOfFloat(ty.ll, N));
    }
    ///
    uint intWidth()
    {
        return LLVMGenericValueIntWidth(value);
    }
    ///
    ulong toUInt()
    {
        return LLVMGenericValueToInt(value, false);
    }
    ///
    long toSInt()
    {
        return LLVMGenericValueToInt(value, true);
    }
    ///
    void* toPointer()
    {
        return LLVMGenericValueToPointer(value);
    }
    ///
    double toFloat(RealType ty)
    {
        return LLVMGenericValueToFloat(ty.ll, value);
    }
}


///
class ExecutionEngine
{
    ///
    private LLVMExecutionEngineRef ee;
    ///
    private this(LLVMExecutionEngineRef ee)
    {
        this.ee = ee;
    }
    ///
    static ExecutionEngine Create(ModuleProvider mp)
    {
        LLVMExecutionEngineRef ee;
        char* err;
        if (LLVMCreateExecutionEngine(&ee, mp.ll, &err))
        {
            auto errmsg = from_stringz(err).dup;
            LLVMDisposeMessage(err);
            if (errmsg.length == 0)
                errmsg = "Error creating execution engine";
            throw new LLVMException(errmsg);
        }
        return new ExecutionEngine(ee);
    }
    ///
    static ExecutionEngine CreateInterpreter(ModuleProvider mp)
    {
        LLVMExecutionEngineRef ee;
        char* err;
        if (LLVMCreateInterpreter(&ee, mp.ll, &err))
        {
            auto errmsg = from_stringz(err).dup;
            LLVMDisposeMessage(err);
            if (errmsg.length == 0)
                errmsg = "Error creating interpreter";
            throw new LLVMException(errmsg);
        }
        return new ExecutionEngine(ee);
    }
    ///
    static ExecutionEngine CreateJIT(ModuleProvider mp)
    {
        LLVMExecutionEngineRef ee;
        char* err;
        if (LLVMCreateJITCompiler(&ee, mp.ll, &err))
        {
            auto errmsg = from_stringz(err).dup;
            LLVMDisposeMessage(err);
            if (errmsg.length == 0)
                errmsg = "Error creating JIT";
            throw new LLVMException(errmsg);
        }
        return new ExecutionEngine(ee);
    }
    ///
    void dispose()
    {
        LLVMDisposeExecutionEngine(ee);
        ee = null;
    }
    ///
    ~this()
    {
        dispose(); // safe because ee isn't on the GC heap and isn't exposed.
    }
    ///
    void runStaticConstructors()
    {
        LLVMRunStaticConstructors(ee);
    }
    ///
    void runStaticDestructors()
    {
        LLVMRunStaticDestructors(ee);
    }
    ///
    int runAsMain(Function f, char[][] args = null, char[][] env = null) {
        auto argv = new char*[args.length];
        foreach (size_t idx, ref arg; args)
        {
            argv[idx] = to_stringz(arg);
        }

        auto envp = new char*[env.length + 1];
        foreach (size_t idx, ref envvar ; env)
        {
            envp[idx] = to_stringz(envvar);
        }
        envp[$-1] = null;

        return LLVMRunFunctionAsMain(ee, f.value, argv.length, argv.ptr, envp.ptr);
    }
    ///
    GenericValue run(Function f, GenericValue[] args = null)
    {
        auto cargs = new LLVMGenericValueRef[args.length];
        foreach (size_t idx, ref arg ; args)
        {
            cargs[idx] = arg.value;
        }

        auto result = LLVMRunFunction(ee, f.value, cargs.length, cargs.ptr);
        return new GenericValue(result);
    }
    ///
    void freeMachineCodeForFunction(Function f)
    {
        LLVMFreeMachineCodeForFunction(ee, f.value);
    }
    ///
    void addModuleProvider(ModuleProvider mp)
    {
        LLVMAddModuleProvider(ee, mp.ll);
    }
    ///
    Module removeModuleProvider(ModuleProvider mp)
    {
        LLVMModuleRef mod;
        char* err;
        if (LLVMRemoveModuleProvider(ee, mp.ll, &mod, &err))
        {
            auto errmsg = from_stringz(err).dup;
            LLVMDisposeMessage(err);
            if (errmsg.length == 0)
                errmsg = "Error removing ModuleProvider from ExecutionEngine";
            throw new LLVMException(errmsg);
        }
        return Module.GetExisting(mod);
    }
    ///
    Function findFunction(char[] name)
    {
        LLVMValueRef fn;
        if (LLVMFindFunction(ee, to_stringz(name), &fn))
        {
            return null;
        }
        return new Function(fn, getTypeOf(fn));
    }
}
