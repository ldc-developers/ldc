/**
 * Contains compiler-recognized special symbols for dcompute.
 *
 * Copyright: Authors 2017
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   Nicholas Wilson
 */

module ldc.dcompute;

///Readability aliases for compute
enum CompileFor : int
{
    deviceOnly = 0,
    hostAndDevice = 1
}

/++
+ When applied to a module, specifies that the module should be compiled for
+ dcompute (-mdcompute-targets=<...>) using the NVPTX and/or SPIRV backends of
+ LLVM.
+
+ Examples:
+ ---
+ @compute(CompileFor.deviceOnly) module foo;
+ import ldc.attributes;
+ ---
+/
struct compute
{
    CompileFor codeProduction = CompileFor.deviceOnly;
}

/++
+ Mark a function as a 'kernel', a compute API (CUDA, OpenCL) entry point.
+ Equivalent to __kernel__ in OpenCL and __global__ in CUDA.
+
+ Examples:
+ ---
+ @compute(CompileFor.deviceOnly) module foo;
+ import ldc.attributes;
+
+ @kernel void bar()
+ {
+     //...
+ }
+ ---
+/
private struct _kernel {}
immutable kernel = _kernel();

/**
 * DCompute has the notion of adress spaces, provide by the magic struct below.
 * The numbers are for the DCompute virtual addess space and are translated into
 * the correct address space for each DCompute backend (SPIRV, NVPTX).
 * The table below shows the equivalent annotation between DCompute OpenCL and CUDA
 *
 *   DCompute   OpenCL      Cuda
 *   Global     __global    __device__
 *   Shared     __local     __shared__
 *   Constant   __constant  __constant__
 *   Private    __private   __local__
 *   Generic    __generic   (no qualifier)
 */
struct Pointer(AddrSpace as, T)
{
    T* ptr;
    alias ptr this;
}

enum AddrSpace : uint
{
    Private  = 0,
    Global   = 1,
    Shared   = 2,
    Constant = 3,
    Generic  = 4,
}

alias PrivatePointer(T)  = Pointer!(AddrSpace.Private,  T);
alias GlobalPointer(T)   = Pointer!(AddrSpace.Global,   T);
alias SharedPointer(T)   = Pointer!(AddrSpace.Shared,   T);
alias ConstantPointer(T) = Pointer!(AddrSpace.Constant, immutable(T));
alias GenericPointer(T)  = Pointer!(AddrSpace.Generic,  T);


