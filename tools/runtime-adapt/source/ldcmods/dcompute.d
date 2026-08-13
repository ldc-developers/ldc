//===-- tools/runtime-adapt/source/ldcmods/dcompute.d -------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/dcompute.d ← gen/dcompute / driver/main.cpp LDC_DCompute_* versions.
// Typical LDC commit: addPredefinedGlobalIdent("LDC_DCompute_*") then types here.
//
//===----------------------------------------------------------------------===//

module ldcmods.dcompute;

import compilerparse;
import ldcmods.common;

import std.algorithm : canFind;
import std.array : appender;

bool wantDcompute(const CompilerModel m)
{
    return m.dcomputeNames.length > 0 || hasLdcModule(m, "dcompute");
}

string renderDcompute(const CompilerModel model, string tag)
{
    auto buf = appender!string();
    buf.put(banner(tag));
    buf.put("module ldc.dcompute;\n\n");
    buf.put("enum ReflectTarget : uint { Host = 0, OpenCL = 1, CUDA = 2 }\n\n");
    buf.put("pure nothrow @nogc\nextern (C) bool __dcompute_reflect(ReflectTarget t, uint _version = 0);\n\n");
    buf.put("enum CompileFor : int { deviceOnly = 0, hostAndDevice = 1 }\n\n");
    buf.put("struct compute { CompileFor codeProduction = CompileFor.deviceOnly; }\n\n");
    buf.put("private struct _kernel { size_t[3] bounds; }\n");
    buf.put("_kernel kernel(size_t[3] a = [1, 1, 1]) => _kernel(a);\n\n");
    buf.put("enum AddrSpace : uint\n{\n");
    buf.put("    Private = 0,\n    Global = 1,\n    Shared = 2,\n    Constant = 3,\n    Generic = 4,\n}\n\n");
    buf.put("struct Pointer(AddrSpace as, T) { T* ptr; alias ptr this; }\n");
    buf.put("struct Variable(AddrSpace as, T) { T val; alias val this; }\n");
    buf.put("alias PrivatePointer(T) = Pointer!(AddrSpace.Private, T);\n");
    buf.put("alias GlobalPointer(T) = Pointer!(AddrSpace.Global, T);\n");
    buf.put("alias SharedPointer(T) = Pointer!(AddrSpace.Shared, T);\n");
    buf.put("alias ConstantPointer(T) = Pointer!(AddrSpace.Constant, immutable(T));\n");
    buf.put("alias GenericPointer(T) = Pointer!(AddrSpace.Generic, T);\n");
    buf.put("alias Global(T) = Variable!(AddrSpace.Global, T);\n");
    buf.put("alias Shared(T) = shared Variable!(AddrSpace.Shared, T);\n");
    buf.put("alias Constant(T) = immutable Variable!(AddrSpace.Constant, T);\n");
    foreach (n; model.dcomputeNames)
    {
        if (n == "compute" || n == "_kernel" || n == "kernel" || n == "Pointer"
            || n == "__dcompute_reflect" || buf.data.canFind(n))
            continue;
        buf.put("\n// compiler id: ");
        buf.put(n);
        buf.put("\n");
    }
    return buf.data;
}

void correctDcompute(ref GeneratedFile f, const CompilerModel model)
{
    foreach (n; model.dcomputeNames)
    {
        if (!n.length || f.body.canFind(n))
            continue;
        f.body ~= "\nstruct " ~ (n[0] == '_' ? n[1 .. $] : n) ~ " { }\n";
        f.corrections ~= "add-dcompute:" ~ n;
    }
}
