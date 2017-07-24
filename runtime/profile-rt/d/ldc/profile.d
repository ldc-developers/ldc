/**
 * Contains funtions to control and query profiling information for profile
 * instrumented programs (compiled with -fprofile-instr-generate).
 * It provides an interface to the profile-rt runtime library.
 *
 * Note that this only works for instrumented binaries.
 *
 * Copyright: Authors 2016-2016
 * License: University of Illinois Open Source License and MIT License. See LDC's LICENSE for details.
 * Authors:   Johan B C Engelen
 */
module ldc.profile;

version(LDC_LLVM_309) version = HASHED_FUNC_NAMES;
version(LDC_LLVM_400) version = HASHED_FUNC_NAMES;
version(LDC_LLVM_500) version = HASHED_FUNC_NAMES;
version(LDC_LLVM_600) version = HASHED_FUNC_NAMES;

@nogc:
nothrow:

/**
 * Data structure for profile data per instrumented function
 */
extern(C++) struct ProfileData {
    // This has to match INSTR_PROF_DATA in profile-rt/InstrProfData.inc

    version(LDC_LLVM_307)
    {
        uint NameSize;
        uint NumCounters;
        ulong FuncHash;
        immutable(char)* Name;
        ulong* Counters;
    }
    else version(LDC_LLVM_308)
    {
        uint NameSize;
        uint NumCounters;
        ulong FuncHash;
        immutable(char)* Name;
        ulong* Counters;
        void* FunctionPointer;
        void* Values;
        ushort NumValueSites;
    }
    else version(LDC_LLVM_309)
    {
        ulong NameRef;
        ulong FuncHash;
        ulong* Counters;
        void* FunctionPointer;
        void* Values;
        uint NumCounters;
        ushort NumValueSites;
    }
    else version(LDC_LLVM_400)
    {
        ulong NameRef;
        ulong FuncHash;
        ulong* Counters;
        void* FunctionPointer;
        void* Values;
        uint NumCounters;
        ushort NumValueSites;
    }
    else version(LDC_LLVM_500)
    {
        ulong NameRef;
        ulong FuncHash;
        ulong* Counters;
        void* FunctionPointer;
        void* Values;
        uint NumCounters;
        ushort NumValueSites;
    }
    else version(LDC_LLVM_600)
    {
        ulong NameRef;
        ulong FuncHash;
        ulong* Counters;
        void* FunctionPointer;
        void* Values;
        uint NumCounters;
        ushort NumValueSites;
    }
    else
    {
        static assert(0, "unsupported LLVM version");
    }
}

// Symbols provided by profile-rt lib
private {
extern(C) {
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
}}

/**
 * Reset all profiling information of the whole program.
 * This can be used for example to remove transient start-up behavior from the
 * profile.
 */
void resetAll() {
    __llvm_profile_reset_counters();
}

/**
 * Reset profile counter values for a function.
 *
 * The function does nothing if no profile data is found for ($D F).
 *
 * Params:
 *  F = The function to set the profile data of.
 */
void resetCounts(alias F)()
    // TODO: add constraint on F
{
    auto data = getData!F;
    if (data && ((*data).NumCounters > 0))
    {
        cast(ulong[])(*data).Counters[0..(*data).NumCounters] = 0;
    }
}

/**
 * Get profile data struct for a given (mangled) function name.
 *
 * Params:
 *  funcname = Mangled function name.
 * Returns:
 *  Pointer to the profile data for ($D funcname), or null if no profile data
 *  was found for ($D funcname).
 */
const(ProfileData)* getData(string funcname)
{
    version(HASHED_FUNC_NAMES)
    {
        import std.digest.md;
        import std.bitmanip;
        auto md5hash = md5Of(funcname);
        auto nameref = peek!(ulong, Endian.littleEndian)(md5hash[0..8]);
    }

    for (auto data = __llvm_profile_begin_data(),
              e = __llvm_profile_end_data();
        data < e; ++data)
    {
        version(HASHED_FUNC_NAMES)
        {
            if (nameref == (*data).NameRef)
                return data;
        }
        else
        {
            if (funcname == (*data).Name[0..(*data).NameSize])
                return data;
        }
    }
    return null;
}

/**
 * Get profile data struct for a given function.
 *
 * Params:
 *  F = The function to get the profile data of.
 * Returns:
 *  Pointer to the profile data for ($D F), or null if no profile data was found
 *  for ($D F).
 */
const(ProfileData)* getData(alias F)()
    // TODO: add constraint on F
{
    enum mangledName = F.mangleof;
    version(Win32)
    {
        import std.traits : functionLinkage;
        static if (functionLinkage!F == "D")
            return getData("_" ~ mangledName);
        else
            return getData(mangledName);
    }
    else
    {
        return getData(mangledName);
    }
}

/**
 * Get the current number of times function ($D F) has been called.
 *
 * Params:
 *  F = The function to get the call count of.
 *
 * Returns:
 *  The call count of function ($D F). If no profile data
 *  is found for ($D F), ulong.max is returned.
 */
ulong getCallCount(alias F)()
    // TODO: add constraint on F
{
    auto data = getData!F;
    if (data && ((*data).NumCounters > 0))
    {
        return (*data).Counters[0];
    }
    else
    {
        return ulong.max;
    }
}

/**
 * Get current counter value.
 *
 * Params:
 *  F = The function to get the profile data of.
 *  idx = Counter index.
 *
 * Returns:
 *  Value of the counter with index ($D idx) for function ($D F). If no profile data
 *  is found for ($D F), or if ($D idx) is out of range for ($D F), ulong.max is returned.
 */
ulong getCount(alias F)(uint idx)
    // TODO: add constraint on F
{
    auto data = getData!F;
    if (data && (idx < (*data).NumCounters))
    {
        return (*data).Counters[idx];
    }
    else
    {
        return ulong.max;
    }
}

/**
 * Set profile counter value.
 *
 * The function does nothing if no profile data is found for ($D F), or if ($D
 * idx) is out of range for ($D F).
 *
 * Params:
 *  F = The function to set the profile data of.
 *  idx = Counter index.
 *  count = The new counter value.
 *
 */
void setCount(alias F)(uint idx, ulong count)
    // TODO: add constraint on F
{
    auto data = getData!F;
    if (data && (idx < (*data).NumCounters))
    {
        cast(ulong)(*data).Counters[idx] = count;
    }
}
