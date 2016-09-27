/*
 Variable declaration naming
 
 DCompute   OpenCL      Cuda            Metal
 Global     __global    __device__      device
 Shared     __local     __shared__      threadgroup
 Constant   __constant  __constant__    constant
 Private    __private   __local__       thread
 Generic
 
 */
module dcompute.types.pointer;


enum Private = 0;
enum Global = 1;
enum Shared = 2;
enum Constant = 3;
enum Generic = 4;

alias PrivatePointer(T)     = Pointer!(0,T);
alias GlobalPointer(T)      = Pointer!(1,T);
alias SharedPointer(T)      = Pointer!(2,T);
alias ConstantPointer(T)    = Pointer!(3,T);
alias GenericPointer(T)     = Pointer!(4,T);

//This is a Magic compiler type. DO NOT CHANGE
struct Pointer(uint p, T) if(p <= Generic)
{
    T* ptr;
    alias ptr this;
}

