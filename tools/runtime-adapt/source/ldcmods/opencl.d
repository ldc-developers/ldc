//===-- tools/runtime-adapt/source/ldcmods/opencl.d ---------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/opencl.di ← gen/dcompute + magic module "opencl".
//
//===----------------------------------------------------------------------===//

module ldcmods.opencl;

import compilerparse;
import ldcmods.common;

import std.algorithm : canFind;

bool wantOpencl(const CompilerModel m)
{
    return hasLdcModule(m, "opencl") || m.magicModules.canFind("opencl");
}

string renderOpencl(const CompilerModel, string tag)
{
    return banner(tag) ~ q"D
@compute(CompileFor.deviceOnly) module ldc.opencl;

import ldc.dcompute;

alias Sampler = SharedPointer!sampler_t;
struct sampler_t;
struct image1d_ro_t; struct image1d_wo_t; struct image1d_rw_t;
struct image2d_ro_t; struct image2d_wo_t; struct image2d_rw_t;
struct image3d_ro_t; struct image3d_wo_t; struct image3d_rw_t;
struct image1d_array_ro_t; struct image1d_array_wo_t; struct image1d_array_rw_t;
struct image2d_array_ro_t; struct image2d_array_wo_t; struct image2d_array_rw_t;
struct image1d_buffer_ro_t; struct image1d_buffer_wo_t; struct image1d_buffer_rw_t;
struct image2d_depth_ro_t; struct image2d_depth_wo_t; struct image2d_depth_rw_t;
struct image2d_array_depth_ro_t; struct image2d_array_depth_wo_t; struct image2d_array_depth_rw_t;
struct reserve_id_t; struct pipe_ro_t; struct pipe_wo_t;
struct event_t; struct clk_event_t; struct queue_t;

template Image(int dim)
{
    static if (dim == 1) alias Image = GlobalPointer!image1d_rw_t;
    else static if (dim == 2) alias Image = GlobalPointer!image2d_rw_t;
    else static if (dim == 3) alias Image = GlobalPointer!image3d_rw_t;
}
template ReadOnlyImage(int dim)
{
    static if (dim == 1) alias ReadOnlyImage = GlobalPointer!image1d_ro_t;
    else static if (dim == 2) alias ReadOnlyImage = GlobalPointer!image2d_ro_t;
    else static if (dim == 3) alias ReadOnlyImage = GlobalPointer!image3d_ro_t;
}
template WriteOnlyImage(int dim)
{
    static if (dim == 1) alias WriteOnlyImage = GlobalPointer!image1d_wo_t;
    else static if (dim == 2) alias WriteOnlyImage = GlobalPointer!image2d_wo_t;
    else static if (dim == 3) alias WriteOnlyImage = GlobalPointer!image3d_wo_t;
}
template ImageArray(int dim)
{
    static if (dim == 1) alias ImageArray = GlobalPointer!image1d_array_rw_t;
    else alias ImageArray = GlobalPointer!image2d_array_rw_t;
}
template ReadOnlyImageArray(int dim)
{
    static if (dim == 1) alias ReadOnlyImageArray = GlobalPointer!image1d_array_ro_t;
    else alias ReadOnlyImageArray = GlobalPointer!image2d_array_ro_t;
}
template WriteOnlyImageArray(int dim)
{
    alias WriteOnlyImageArray = GlobalPointer!image2d_array_wo_t;
}

alias Image1dBuffer = GlobalPointer!image1d_buffer_rw_t;
alias ReadOnlyImage1dBuffer = GlobalPointer!image1d_buffer_ro_t;
alias WriteOnlyImage1dBuffer = GlobalPointer!image1d_buffer_wo_t;
alias Image2dDepth = GlobalPointer!image2d_depth_rw_t;
alias ReadOnlyImage2dDepth = GlobalPointer!image2d_depth_ro_t;
alias WriteOnlyImage2dDepth = GlobalPointer!image2d_depth_wo_t;
alias Image2dArrayDepth = GlobalPointer!image2d_array_depth_rw_t;
alias ReadOnlyImage2dArrayDepth = GlobalPointer!image2d_array_depth_ro_t;
alias WriteOnlyImage2dArrayDepth = GlobalPointer!image2d_array_depth_wo_t;
alias ReserveId = PrivatePointer!reserve_id_t;
alias ReadOnlyPipe = GlobalPointer!pipe_ro_t;
alias WriteOnlyPipe = GlobalPointer!pipe_wo_t;
D";
}
