/**
 * Contains compiler-recognized special types for dcompute.
 *
 * Copyright: Authors 2017
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   Nicholas Wilson
 */

module ldc.dcomputetypes;

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

enum Private = 0;
enum Global = 1;
enum Shared = 2;
enum Constant = 3;
enum Generic = 4;

alias PrivatePointer(T)     = Pointer!(0,T);
alias GlobalPointer(T)      = Pointer!(1,T);
alias SharedPointer(T)      = Pointer!(2,T);
alias ConstantPointer(T)    = Pointer!(3,immutable(T));
alias GenericPointer(T)     = Pointer!(4,T);

struct Pointer(uint p, T) if(p <= Generic)
{
    T* ptr;
    alias ptr this;
}
