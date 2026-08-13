//===-- tools/runtime-adapt/source/ldcmods/profile.d --------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/profile.di ← gen/pragma.cpp LDC_profile_instr + compiler-rt profile-rt.
//
//===----------------------------------------------------------------------===//

module ldcmods.profile;

import compilerparse;
import ldcmods.common;

bool wantProfile(const CompilerModel m)
{
    return hasLdcModule(m, "profile") || hasPragma(m, "LDC_profile_instr");
}

string renderProfile(const CompilerModel, string tag)
{
    return banner(tag) ~ q"D
module ldc.profile;

import ldc.intrinsics : LLVM_atleast;

@nogc:
nothrow:

extern (C++) struct ProfileData
{
    ulong NameRef;
    ulong FuncHash;
    static if (LLVM_atleast!14)
    {
        private void* RelativeCounters;
        inout(ulong)* Counters()() inout @property @trusted
        {
            return cast(inout(ulong)*)((cast(size_t)&this) + cast(size_t)RelativeCounters);
        }
    }
    else
    {
        ulong* Counters;
    }
    void* FunctionPointer;
    void* Values;
    uint NumCounters;
    ushort NumValueSites;
}

extern (C)
{
    alias uint64_t = ulong;
    alias __llvm_profile_data = ProfileData;
    const(__llvm_profile_data)* __llvm_profile_begin_data();
    const(__llvm_profile_data)* __llvm_profile_end_data();
    immutable(char)* __llvm_profile_begin_names();
    immutable(char)* __llvm_profile_end_names();
    uint64_t* __llvm_profile_begin_counters();
    uint64_t* __llvm_profile_end_counters();
    void __llvm_profile_reset_counters();
    uint64_t __llvm_profile_get_magic();
    uint64_t __llvm_profile_get_version();
}

alias resetAll = __llvm_profile_reset_counters;

const(ProfileData)* getData(alias F)()
{
    return null;
}

void resetCounts(alias F)()
{
    auto data = getData!F;
    if (data && (*data).NumCounters)
        cast(ulong[])(*data).Counters[0 .. (*data).NumCounters] = 0;
}

ulong getCallCount(alias F)()
{
    auto data = getData!F;
    return data ? (*data).NumCounters : ulong.max;
}
D";
}
