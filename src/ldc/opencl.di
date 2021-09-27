@compute(CompileFor.deviceOnly) module ldc.opencl;

import ldc.dcompute;

/*
 See https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_C.html#other-built-in-data-types
 and https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_C.html#pipe-functions
 for the meaning of these types. Types with access qualifiers ('read_only', 'write_only' and 'read_write')
 in OpenCL are actually separate types are are represented as '*_ro_t', '*_wo_t' and '*rw_t' types.
 
 All these opaque types must be accessed through pointers declared in the aliases below.
 Unlike in OpenCL, raw pipes are untyped (like image types), and the desired type must be provided
 to the pipe read/write functions.
 */

alias Sampler = SharedPointer!sampler_t;

template Image(int dim)
{
    static if (dim == 1)
        alias Image = GlobalPointer!image1d_rw_t;
    else static if (dim == 2)
        alias Image = GlobalPointer!image2d_rw_t;
    else static if (dim == 3)
        alias Image = GlobalPointer!image3d_rw_t;
}
template ReadOnlyImage(int dim)
{
    static if (dim == 1)
        alias ReadOnlyImage = GlobalPointer!image1d_ro_t;
    else static if (dim == 2)
        alias ReadOnlyImage = GlobalPointer!image2d_ro_t;
    else static if (dim == 3)
        alias ReadOnlyImage = GlobalPointer!image3d_ro_t;
}
template WriteOnlyImage(int dim)
{
    static if (dim == 1)
        alias WriteOnlyImage = GlobalPointer!image1d_wo_t;
    else static if (dim == 2)
        alias WriteOnlyImage = GlobalPointer!image2d_wo_t;
    else static if (dim == 3)
        alias WriteOnlyImage = GlobalPointer!image3d_wo_t;
}

// N.B: 1 & 2 dimensions only
template ImageArray(int dim)
{
    static if (dim == 1)
        alias ImageArray = GlobalPointer!image1d_array_rw_t;
    else static if (dim == 2)
        alias ImageArray = GlobalPointer!image2d_array_rw_t;
}
template ReadOnlyImageArray(int dim)
{
    static if (dim == 1)
        alias ReadOnlyImageArray = GlobalPointer!image1d_array_ro_t;
    else static if (dim == 2)
        alias ReadOnlyImageArray = GlobalPointer!image2d_array_ro_t;
}
template WriteOnlyImageArray(int dim)
{
    static if (dim == 1)
        alias WriteOnlyImageArray = GlobalPointer!image2d_array_wo_t;
    else static if (dim == 2)
        alias WriteOnlyImageArray = GlobalPointer!image2d_array_wo_t;
}

alias Image1dBuffer          = GlobalPointer!image1d_buffer_rw_t;
alias ReadOnlyImage1dBuffer  = GlobalPointer!image1d_buffer_ro_t;
alias WriteOnlyImage1dBuffer = GlobalPointer!image1d_buffer_wo_t;

alias Image2dDepth           = GlobalPointer!image2d_depth_rw_t;
alias ReadOnlyImage2dDepth   = GlobalPointer!image2d_depth_ro_t;
alias WriteOnlyImage2dDepth  = GlobalPointer!image2d_depth_wo_t;

alias Image2dArrayDepth          = GlobalPointer!image2d_array_depth_rw_t;
alias ReadOnlyImage2dArrayDepth  = GlobalPointer!image2d_array_depth_ro_t;
alias WriteOnlyImage2dArrayDepth = GlobalPointer!image2d_array_depth_wo_t;

alias ReserveId     = PrivatePointer!reserve_id_t;
alias ReadOnlyPipe  = GlobalPointer!pipe_ro_t;
alias WriteOnlyPipe = GlobalPointer!pipe_wo_t;

// ConstantPointer
struct sampler_t;

/// Image types, GlobalPointer
struct image1d_ro_t;
struct image1d_wo_t;
struct image1d_rw_t;

struct image1d_array_ro_t;
struct image1d_array_wo_t;
struct image1d_array_rw_t;

struct image1d_buffer_ro_t;
struct image1d_buffer_wo_t;
struct image1d_buffer_rw_t;

struct image2d_ro_t;
struct image2d_wo_t;
struct image2d_rw_t;

struct image2d_array_ro_t;
struct image2d_array_wo_t;
struct image2d_array_rw_t;

struct image2d_depth_ro_t;
struct image2d_depth_wo_t;
struct image2d_depth_rw_t;

struct image2d_array_depth_ro_t;
struct image2d_array_depth_wo_t;
struct image2d_array_depth_rw_t;

struct image3d_ro_t;
struct image3d_wo_t;
struct image3d_rw_t;

struct reserve_id_t;
struct pipe_ro_t;
struct pipe_wo_t;
// N.B there is no struct pipe_rw_t;

/* Requires support for device side enqueue : unsupported.
struct event_t;
struct clk_event_t;
struct queue_t;
 */
/* Requires cl_khr_gl_msaa_sharing : unsupported
 struct image2d_msaa_t;
 struct image2d_array_msaa_t;
 struct image2d_msaa_depth_t;
 struct image2d_array_msaa_depth_t;
 */
