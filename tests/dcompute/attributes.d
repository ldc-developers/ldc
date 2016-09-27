module dcompute.attributes;


///Mark a module to be compiled for {ptx, spv}
private struct _compute {}
immutable compute = _compute();

///Mark a function as a compute API entry point
private struct _kernel {}
immutable kernel = _kernel();



//TODO: support the following in the compiler

///OpenCL only
///hint that the consumer should vectorise the function with type T
//e.g. @kernel @vec_hint_type!int4 void foo(){}
// the extra i32 0 or 1 that should end up in the metadata
//is an indication of T or the elements of T, signedness
struct vec_hint_type(T){}

///OpenCL only
//require the execution of the kernel to have groups of this size
struct reqd_work_group_size
{
    uint x,y,z;
}

// as above but only a hint
struct work_group_size_hint
{
    uint x,y,z;
}

///Cuda only
///similar to above but not quite the same
//TODO: merge with *work_group_size*
struct launch_bounds
{
    uint max_threads_per_block;
    uint min_threads_per_multiporcessor;
}

/*
 As yet unsupported metal attributes
 struct vertex {}
 struct fragment {}
 enum patch_type
 {
    quad,
 }
 struct patch
 {
    patch_type type;
    uint n;
 }
 struct early_fragment_tests{}
 */

