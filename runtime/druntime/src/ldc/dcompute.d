/**
 * Contains compiler-recognized special symbols for dcompute.
 *
 * Copyright: Authors 2017
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   Nicholas Wilson
 */

module ldc.dcompute;

enum ReflectTarget : uint
{
    // These numbers MUST match DcomputeTarget::target in LDC.
    Host = 0,
    OpenCL = 1,
    CUDA = 2,
}
/**
 * The pseudo conditional compilation function.
 * returns true if t and _version match the target and it's version.
 * think __ctfe but per target, codegen time conditional compilation.
 * as oppsed to compiletime/runtime.
 * arguments MUST be compiletime constants
 * valid values of _version are for OpenCL 100 110 120 200 210
 * and for CUDA are x*100 + y*10 for x any valid values of sm x.y
 * use 0 as a wildcard to match any version.
 
 * This is mostly used for selecting the correct intrinsic for the
 * given target and version, but could also be used to tailor for
 * performance characteristics. See dcompute.std.index for an example
 * of how to select intrinsics.
 */
pure nothrow @nogc
extern(C) bool __dcompute_reflect(ReflectTarget t, uint _version = 0);

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
+ import ldc.dcompute;
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
+ import ldc.dcompute;
+
+ @kernel void bar()
+ {
+     //...
+ }
+ ---
+/
private struct _kernel {}
enum kernel = _kernel();

/++
 + DCompute has the notion of adress spaces, provide by the magic structs below.
 + The numbers are for the DCompute virtual addess space and are translated into
 + the correct address space for each DCompute backend (SPIRV, NVPTX).
 + The table below shows the equivalent annotation between DCompute OpenCL and CUDA
 +
 +   DCompute   OpenCL      Cuda
 +   Global     __global    __device__
 +   Shared     __local     __shared__
 *   Constant   __constant  __constant__
 +   Private    __private   __local__
 +   Generic    __generic   (no qualifier)
 +/
struct Pointer(AddrSpace as, T)
{
    T* ptr;
    alias ptr this;
}

struct Variable(AddrSpace as, T)
{
    T val;
    alias val this;
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

// N.B private variables are declared on the stack and so cannot be declared
// at module scope.
// No variables exist in the generic address space, it is purely for generalised
// pointers to point to.
//
// The __gshared below does not work. It is kludged into place in `DtoResolveVariable`

alias Global(T)   = /*__gshared*/ Variable!(AddrSpace.Global,   T);
alias Shared(T)   = shared        Variable!(AddrSpace.Shared,   T);
alias Constant(T) = immutable     Variable!(AddrSpace.Constant, T);
