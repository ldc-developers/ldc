/**
 * This module contains functions and structures required for
 * exception handling.
 */
module ldc.eh;

private import core.stdc.stdio;
private import core.stdc.stdlib;
private import core.stdc.stdarg;

// debug = EH_personality;
// debug = EH_personality_verbose;

version (PPC)   version = PPC_Any;
version (PPC64) version = PPC_Any;

// current EH implementation works on x86
// if it has a working unwind runtime
version(X86)
{
    version(linux) version=GCC_UNWIND;
    version(darwin) version=GCC_UNWIND;
    version(solaris) version=GCC_UNWIND;
    version(FreeBSD) version=GCC_UNWIND;
    version(MinGW) version=GCC_UNWIND;
}
version(X86_64)
{
    version(linux) version=GCC_UNWIND;
    version(darwin) version=GCC_UNWIND;
    version(solaris) version=GCC_UNWIND;
    version(FreeBSD) version=GCC_UNWIND;
    version(MinGW) version=GCC_UNWIND;
}
version (ARM)
{
    // FIXME: Almost certainly wrong.
    version (linux) version = GCC_UNWIND;
    version (FreeBSD) version = GCC_UNWIND;
}
version (AArch64)
{
    version (linux) version = GCC_UNWIND;
}
version (PPC_Any)
{
    version (linux) version = GCC_UNWIND;
}
version (MIPS)
{
    version (linux) version = GCC_UNWIND;
}
version (MIPS64)
{
    version (linux) version = GCC_UNWIND;
}

//version = HP_LIBUNWIND;

// D runtime functions
extern(C)
{
    int _d_isbaseof(ClassInfo oc, ClassInfo c);
}

// libunwind headers
extern(C)
{
    // FIXME: Some of these do not actually exist on ARM.
    enum _Unwind_Reason_Code : int
    {
        NO_REASON = 0, // "OK" on ARM
        FOREIGN_EXCEPTION_CAUGHT = 1,
        FATAL_PHASE2_ERROR = 2,
        FATAL_PHASE1_ERROR = 3,
        NORMAL_STOP = 4,
        END_OF_STACK = 5,
        HANDLER_FOUND = 6,
        INSTALL_CONTEXT = 7,
        CONTINUE_UNWIND = 8,
        FAILURE = 9 // ARM only
    }

    enum _Unwind_Action : int
    {
        SEARCH_PHASE = 1,
        CLEANUP_PHASE = 2,
        HANDLER_FRAME = 4,
        FORCE_UNWIND = 8
    }

    enum _DW_EH_Format : int
    {
        DW_EH_PE_absptr  = 0x00,  // The Value is a literal pointer whose size is determined by the architecture.
        DW_EH_PE_uleb128 = 0x01,  // Unsigned value is encoded using the Little Endian Base 128 (LEB128)
        DW_EH_PE_udata2  = 0x02,  // A 2 bytes unsigned value.
        DW_EH_PE_udata4  = 0x03,  // A 4 bytes unsigned value.
        DW_EH_PE_udata8  = 0x04,  // An 8 bytes unsigned value.
        DW_EH_PE_sleb128 = 0x09,  // Signed value is encoded using the Little Endian Base 128 (LEB128)
        DW_EH_PE_sdata2  = 0x0A,  // A 2 bytes signed value.
        DW_EH_PE_sdata4  = 0x0B,  // A 4 bytes signed value.
        DW_EH_PE_sdata8  = 0x0C,  // An 8 bytes signed value.

        DW_EH_PE_pcrel   = 0x10,  // Value is relative to the current program counter.
        DW_EH_PE_textrel = 0x20,  // Value is relative to the beginning of the .text section.
        DW_EH_PE_datarel = 0x30,  // Value is relative to the beginning of the .got or .eh_frame_hdr section.
        DW_EH_PE_funcrel = 0x40,  // Value is relative to the beginning of the function.
        DW_EH_PE_aligned = 0x50,  // Value is aligned to an address unit sized boundary.

        DW_EH_PE_indirect = 0x80,

        DW_EH_PE_omit    = 0xff   // Indicates that no value is present.
    }

    alias void* _Unwind_Context_Ptr;

    alias void function(_Unwind_Reason_Code, _Unwind_Exception*) _Unwind_Exception_Cleanup_Fn;

    struct _Unwind_Exception
    {
        ulong exception_class;
        _Unwind_Exception_Cleanup_Fn exception_cleanup;
        ptrdiff_t private_1;
        ptrdiff_t private_2;
    }

// interface to HP's libunwind from http://www.nongnu.org/libunwind/
version(HP_LIBUNWIND)
{
    void __libunwind_Unwind_Resume(_Unwind_Exception *);
    _Unwind_Reason_Code __libunwind_Unwind_RaiseException(_Unwind_Exception *);
    ptrdiff_t __libunwind_Unwind_GetLanguageSpecificData(_Unwind_Context_Ptr
            context);
    ptrdiff_t __libunwind_Unwind_GetIP(_Unwind_Context_Ptr context);
    ptrdiff_t __libunwind_Unwind_SetIP(_Unwind_Context_Ptr context,
            ptrdiff_t new_value);
    ptrdiff_t __libunwind_Unwind_SetGR(_Unwind_Context_Ptr context, int index,
            ptrdiff_t new_value);
    ptrdiff_t __libunwind_Unwind_GetRegionStart(_Unwind_Context_Ptr context);
    ptrdiff_t __libunwind_Unwind_GetTextRelBase(_Unwind_Context_Ptr context);
    ptrdiff_t __libunwind_Unwind_GetDataRelBase(_Unwind_Context_Ptr context);

    alias __libunwind_Unwind_Resume _Unwind_Resume;
    alias __libunwind_Unwind_RaiseException _Unwind_RaiseException;
    alias __libunwind_Unwind_GetLanguageSpecificData
        _Unwind_GetLanguageSpecificData;
    alias __libunwind_Unwind_GetIP _Unwind_GetIP;
    alias __libunwind_Unwind_SetIP _Unwind_SetIP;
    alias __libunwind_Unwind_SetGR _Unwind_SetGR;
    alias __libunwind_Unwind_GetRegionStart _Unwind_GetRegionStart;
    alias __libunwind_Unwind_GetTextRelBase _Unwind_GetTextRelBase;
    alias __libunwind_Unwind_GetDataRelBase _Unwind_GetDataRelBase;
}
else version(GCC_UNWIND)
{
    ptrdiff_t _Unwind_GetLanguageSpecificData(_Unwind_Context_Ptr context);
    version (ARM)
    {
        _Unwind_Reason_Code _Unwind_RaiseException(_Unwind_Control_Block*);
        void _Unwind_Resume(_Unwind_Control_Block*);

        // On ARM, these are macros resp. not visible (static inline). To avoid
        // an unmaintainable amount of dependencies on implementation details,
        // just use a C shim.
        ptrdiff_t _d_eh_GetIP(_Unwind_Context_Ptr context);
        alias _Unwind_GetIP = _d_eh_GetIP;

        void _d_eh_SetIP(_Unwind_Context_Ptr context, ptrdiff_t new_value);
        alias _Unwind_SetIP = _d_eh_SetIP;

        ptrdiff_t _d_eh_GetGR(_Unwind_Context_Ptr context, int index);
        alias _Unwind_GetGR = _d_eh_GetGR;

        void _d_eh_SetGR(_Unwind_Context_Ptr context, int index, ptrdiff_t new_value);
        alias _Unwind_SetGR = _d_eh_SetGR;
    }
    else
    {
        _Unwind_Reason_Code _Unwind_RaiseException(_Unwind_Exception*);
        void _Unwind_Resume(_Unwind_Exception*);
        ptrdiff_t _Unwind_GetIP(_Unwind_Context_Ptr context);
        void _Unwind_SetIP(_Unwind_Context_Ptr context, ptrdiff_t new_value);
        void _Unwind_SetGR(_Unwind_Context_Ptr context, int index,
                ptrdiff_t new_value);
    }
    ptrdiff_t _Unwind_GetRegionStart(_Unwind_Context_Ptr context);
    ptrdiff_t _Unwind_GetTextRelBase(_Unwind_Context_Ptr context);
    ptrdiff_t _Unwind_GetDataRelBase(_Unwind_Context_Ptr context);
}
else
{
    // runtime calls these directly
    void _Unwind_Resume(_Unwind_Exception*)
    {
        fprintf(stderr, "_Unwind_Resume is not implemented on this platform.\n");
    }
    _Unwind_Reason_Code _Unwind_RaiseException(_Unwind_Exception*)
    {
        fprintf(stderr, "_Unwind_RaiseException is not implemented on this platform.\n");
        return _Unwind_Reason_Code.FATAL_PHASE1_ERROR;
    }
}

}

// error and exit
extern(C) private void fatalerror(in char* format, ...)
{
    va_list args;
    va_start(args, format);
    printf("Fatal error in EH code: ");
    vprintf(format, args);
    printf("\n");
    abort();
}


// helpers for reading certain DWARF data
private ubyte* get_uleb128(ubyte* addr, ref size_t res)
{
    res = 0;
    size_t bitsize = 0;

    // read as long as high bit is set
    while(*addr & 0x80)
    {
        res |= (*addr & 0x7f) << bitsize;
        bitsize += 7;
        addr += 1;
        if (bitsize >= size_t.sizeof*8)
            fatalerror("tried to read uleb128 that exceeded size of size_t");
    }
    // read last
    if (bitsize != 0 && *addr >= 1L << size_t.sizeof*8 - bitsize)
        fatalerror("tried to read uleb128 that exceeded size of size_t");
    res |= (*addr) << bitsize;

    return addr + 1;
}

private ubyte* get_sleb128(ubyte* addr, ref ptrdiff_t res)
{
    res = 0;
    size_t bitsize = 0;

    // read as long as high bit is set
    while (*addr & 0x80)
    {
        res |= (*addr & 0x7f) << bitsize;
        bitsize += 7;
        addr += 1;
        if (bitsize >= size_t.sizeof*8)
            fatalerror("tried to read sleb128 that exceeded size of size_t");
    }
    // read last
    if (bitsize != 0 && *addr >= 1L << size_t.sizeof*8 - bitsize)
        fatalerror("tried to read sleb128 that exceeded size of size_t");
    res |= (*addr) << bitsize;

    // take care of sign
    if (bitsize < size_t.sizeof*8 && ((*addr) & 0x40))
        res |= cast(ptrdiff_t)(-1) ^ ((1 << (bitsize+7)) - 1);

    return addr + 1;
}

private size_t get_size_of_encoded_value(ubyte encoding)
{
    if (encoding == _DW_EH_Format.DW_EH_PE_omit)
        return 0;

    switch (encoding & 0x07)
    {
        case _DW_EH_Format.DW_EH_PE_absptr:
            return size_t.sizeof;
        case _DW_EH_Format.DW_EH_PE_udata2:
            return 2;
        case _DW_EH_Format.DW_EH_PE_udata4:
            return 4;
        case _DW_EH_Format.DW_EH_PE_udata8:
            return 8;
        default:
            fatalerror("Unsupported DWARF Exception Header value format: unknown encoding");
    }
    assert(0);
}

private ubyte* get_encoded_value(ubyte* addr, ref size_t res, ubyte encoding, _Unwind_Context_Ptr context)
{
    ubyte *old_addr = addr;

    if (encoding == _DW_EH_Format.DW_EH_PE_aligned)
        goto Lerr;

    switch (encoding & 0x0f)
    {
      case _DW_EH_Format.DW_EH_PE_absptr:
          res = cast(size_t)*cast(ubyte**)addr;
          addr += size_t.sizeof;
          break;

      case _DW_EH_Format.DW_EH_PE_uleb128:
          addr = get_uleb128(addr, res);
          break;

      case _DW_EH_Format.DW_EH_PE_sleb128:
          ptrdiff_t r;
          addr = get_sleb128(addr, r);
          r = cast(size_t)res;
          break;

      case _DW_EH_Format.DW_EH_PE_udata2:
          res = *cast(ushort*)addr;
          addr += 2;
          break;
      case _DW_EH_Format.DW_EH_PE_udata4:
          res = *cast(uint*)addr;
          addr += 4;
          break;
      case _DW_EH_Format.DW_EH_PE_udata8:
          res = cast(size_t)*cast(ulong*)addr;
          addr += 8;
          break;

      case _DW_EH_Format.DW_EH_PE_sdata2:
          res = *cast(short*)addr;
          addr += 2;
          break;
      case _DW_EH_Format.DW_EH_PE_sdata4:
          res = *cast(int*)addr;
          addr += 4;
          break;
      case _DW_EH_Format.DW_EH_PE_sdata8:
          res = cast(size_t)*cast(long*)addr;
          addr += 8;
          break;

      default:
          goto Lerr;
    }

    switch (encoding & 0x70)
    {
        case _DW_EH_Format.DW_EH_PE_absptr:
            break;
        case _DW_EH_Format.DW_EH_PE_pcrel:
            res += cast(size_t)old_addr;
            break;
        case _DW_EH_Format.DW_EH_PE_funcrel:
            res += cast(size_t)_Unwind_GetRegionStart(context);
            break;
        case _DW_EH_Format.DW_EH_PE_textrel:
            res += cast(size_t)_Unwind_GetTextRelBase(context);
            break;
        case _DW_EH_Format.DW_EH_PE_datarel:
            res += cast(size_t)_Unwind_GetDataRelBase(context);
            break;
        default:
            goto Lerr;
    }

    if (encoding & _DW_EH_Format.DW_EH_PE_indirect)
        res = cast(size_t)*cast(void**)res;

    return addr;

Lerr:
    fatalerror("Unsupported DWARF Exception Header value format");
    return addr;
}

ptrdiff_t get_base_of_encoded_value(ubyte encoding, _Unwind_Context_Ptr context)
{
    if (encoding == _DW_EH_Format.DW_EH_PE_omit)
        return 0;

    with (_DW_EH_Format) switch (encoding & 0x70) {
        case DW_EH_PE_absptr:
        case DW_EH_PE_pcrel:
        case DW_EH_PE_aligned:
            return 0;

        case DW_EH_PE_textrel:
            return _Unwind_GetTextRelBase (context);
        case DW_EH_PE_datarel:
            return _Unwind_GetDataRelBase (context);
        case DW_EH_PE_funcrel:
            return _Unwind_GetRegionStart (context);

        default:
            fatalerror("Unsupported encoding type to get base from.");
            assert(0);
    }
}

// exception struct used by the runtime.
// _d_throw allocates a new instance and passes the address of its
// _Unwind_Exception member to the unwind call. The personality
// routine is then able to get the whole struct by looking at the data
// surrounding the unwind info.
//
// Note that the compiler-generated landing pad code also relies on the
// exception object reference being stored at offset 0.
struct _d_exception
{
    Object exception_object;
    version (ARM)
    {
        _Unwind_Control_Block unwind_info;
    }
    else
    {
        _Unwind_Exception unwind_info;
    }
}

// the 8-byte string identifying the type of exception
// the first 4 are for vendor, the second 4 for language
//TODO: This may be the wrong way around
__gshared char[8] _d_exception_class = "LLDCD2\0\0";


//
// x86 unwind specific implementation of personality function
// and helpers
//
version(GCC_UNWIND)
{

version(ARM)
{

enum _Unwind_State
{
    VIRTUAL_UNWIND_FRAME = 0,
    UNWIND_FRAME_STARTING = 1,
    UNWIND_FRAME_RESUME = 2,
    ACTION_MASK = 3,
    FORCE_UNWIND = 8,
    END_OF_STACK = 16
}

alias _uw = ptrdiff_t;

struct _Unwind_Control_Block
{
    char[8] exception_class;
    void function(_Unwind_Reason_Code, _Unwind_Control_Block *) exception_cleanup;

    struct unwinder_cache_t
    {
        _uw reserved1;
        _uw reserved2;
        _uw reserved3;
        _uw reserved4;
        _uw reserved5;
    }
    unwinder_cache_t unwinder_cache;

    struct barrier_cache_t
    {
        _uw sp;
        _uw[5] bitpattern;
    }
    barrier_cache_t barrier_cache;

    struct cleanup_cache_t
    {
        _uw[4] bitpattern;
    }
    cleanup_cache_t cleanup_cache;

    struct pr_cache_t
    {
      _uw fnstart;
      _uw *ehtp;
      _uw additional;
      _uw reserved1;
    }
    pr_cache_t pr_cache;
}

extern(C) _Unwind_Reason_Code __gnu_unwind_frame(_Unwind_Control_Block *, _Unwind_Context_Ptr);

private _Unwind_Reason_Code continueUnwind(_Unwind_Control_Block* ucb, _Unwind_Context_Ptr context) {
    if (__gnu_unwind_frame(ucb, context) != _Unwind_Reason_Code.NO_REASON)
        return _Unwind_Reason_Code.FAILURE;
    return _Unwind_Reason_Code.CONTINUE_UNWIND;
}

// Defined in unwind-arm.h.
enum UNWIND_STACK_REG = 13;
enum UNWIND_POINTER_REG = 12;

auto toDException(_Unwind_Control_Block* ucb) {
    return cast(_d_exception*)(cast(ubyte*)ucb - _d_exception.unwind_info.offsetof);
}

// the personality routine gets called by the unwind handler and is responsible for
// reading the EH tables and deciding what to do
extern(C) _Unwind_Reason_Code _d_eh_personality(_Unwind_State state, _Unwind_Control_Block* ucb, _Unwind_Context_Ptr context)
{
    debug(EH_personality_verbose) printf(" - entering personality function. state: %d; ucb: %p, context: %p\n", state, ucb, context);

    _Unwind_Action actions;
    with (_Unwind_State) with (_Unwind_Action) {
        switch (state & _Unwind_State.ACTION_MASK) {
            case _Unwind_State.VIRTUAL_UNWIND_FRAME:
                actions = _Unwind_Action.SEARCH_PHASE;
                break;
            case _Unwind_State.UNWIND_FRAME_STARTING:
                actions = _Unwind_Action.CLEANUP_PHASE;
                if (!(state & _Unwind_State.FORCE_UNWIND) &&
                    ucb.barrier_cache.sp == _Unwind_GetGR(context, UNWIND_STACK_REG)) {
                    actions |= _Unwind_Action.HANDLER_FRAME;
                }
                break;
            case _Unwind_State.UNWIND_FRAME_RESUME:
                return continueUnwind(ucb, context);
            default:
                fatalerror("Unhandled ARM EABI unwind state.");
          }
        actions |= state & _Unwind_State.FORCE_UNWIND;
    }

    // The dwarf unwinder assumes the context structure holds things like the
    // function and LSDA pointers.  The ARM implementation caches these in
    // the exception header (UCB).  To avoid rewriting everything we make a
    // virtual scratch register point at the UCB.
    _Unwind_SetGR(context, UNWIND_POINTER_REG, cast(ptrdiff_t)ucb);

    // check exceptionClass
    //TODO: Treat foreign exceptions with more respect
    if (ucb.exception_class != _d_exception_class)
        return _Unwind_Reason_Code.FATAL_PHASE1_ERROR;
    
    _Unwind_Reason_Code rc = eh_personality_common(actions, ucb.toDException(), context);
    if (rc == _Unwind_Reason_Code.CONTINUE_UNWIND)
        return continueUnwind(ucb, context);
    return rc;
}
}
else
{

// the personality routine gets called by the unwind handler and is responsible for
// reading the EH tables and deciding what to do
extern(C) _Unwind_Reason_Code _d_eh_personality(int ver, _Unwind_Action actions, ulong exception_class, _Unwind_Exception* exception_info, _Unwind_Context_Ptr context)
{
    debug(EH_personality_verbose) printf("entering personality function. context: %p\n", context);
    // check ver: the C++ Itanium ABI only allows ver == 1
    if (ver != 1)
        return _Unwind_Reason_Code.FATAL_PHASE1_ERROR;

    // check exceptionClass
    //TODO: Treat foreign exceptions with more respect
    if ((cast(char*)&exception_class)[0..8] != _d_exception_class)
        return _Unwind_Reason_Code.FATAL_PHASE1_ERROR;

    _d_exception* exception_struct = cast(_d_exception*)(cast(ubyte*)exception_info - _d_exception.unwind_info.offsetof);
    return eh_personality_common(actions, exception_struct, context);
}
}

// the personality routine gets called by the unwind handler and is responsible for
// reading the EH tables and deciding what to do
extern(C) _Unwind_Reason_Code eh_personality_common(_Unwind_Action actions, _d_exception* exception_struct, _Unwind_Context_Ptr context)
{
    // find call site table, action table and classinfo table
    // Note: callsite and action tables do not contain static-length
    // data and will be parsed as needed
    // Note: classinfo_table points past the end of the table
    ubyte* callsite_table;
    ubyte* action_table;
    ubyte* classinfo_table;
    ubyte classinfo_table_encoding;
    _d_getLanguageSpecificTables(context, callsite_table, action_table, classinfo_table, classinfo_table_encoding);
    if (callsite_table is null)
        return _Unwind_Reason_Code.CONTINUE_UNWIND;

    /*
      find landing pad and action table index belonging to ip by walking
      the callsite_table
    */
    ubyte* callsite_walker = callsite_table;

    // get the instruction pointer
    // will be used to find the right entry in the callsite_table
    // -1 because it will point past the last instruction
    ptrdiff_t ip = _Unwind_GetIP(context) - 1;

    // address block_start is relative to
    ptrdiff_t region_start = _Unwind_GetRegionStart(context);

    // table entries
    uint block_start_offset, block_size;
    ptrdiff_t landing_pad;
    size_t action_offset;

    while (true)
    {
        // if we've gone through the list and found nothing...
        if (callsite_walker >= action_table)
            return _Unwind_Reason_Code.CONTINUE_UNWIND;

        block_start_offset = *cast(uint*)callsite_walker;
        block_size = *(cast(uint*)callsite_walker + 1);
        landing_pad = *(cast(uint*)callsite_walker + 2);
        if (landing_pad)
            landing_pad += region_start;
        callsite_walker = get_uleb128(callsite_walker + 3*uint.sizeof, action_offset);

        debug(EH_personality_verbose) printf("ip=%llx %d %d %llx\n", ip, block_start_offset, block_size, landing_pad);

        // since the list is sorted, as soon as we're past the ip
        // there's no handler to be found
        if (ip < region_start + block_start_offset)
            return _Unwind_Reason_Code.CONTINUE_UNWIND;

        // if we've found our block, exit
        if (ip < region_start + block_start_offset + block_size)
            break;
    }

    debug(EH_personality) printf("Found correct landing pad and actionOffset %d\n", action_offset);

    // if there's no action offset and no landing pad, continue unwinding
    if (!action_offset && !landing_pad)
        return _Unwind_Reason_Code.CONTINUE_UNWIND;

    // if there's no action offset but a landing pad, this is a cleanup handler
    else if(!action_offset && landing_pad)
        return _d_eh_install_finally_context(actions, landing_pad, exception_struct, context);

    /*
     walk action table chain, comparing classinfos using _d_isbaseof
    */
    ubyte* action_walker = action_table + action_offset - 1;

    size_t ci_size = get_size_of_encoded_value(classinfo_table_encoding);
    debug(EH_personality) printf(" -- ci_size: %td, ci_encoding: %d\n", ci_size, classinfo_table_encoding);

    size_t ttype_base = get_base_of_encoded_value(classinfo_table_encoding, context);
    debug(EH_personality) printf(" -- ttype_base: 0x%tx\n", ttype_base);

    ptrdiff_t ti_offset, next_action_offset;
    while (true)
    {
        action_walker = get_sleb128(action_walker, ti_offset);
        debug(EH_personality) printf(" -- ti_offset: %tx\n", ti_offset);
        // it is intentional that we not modify action_walker here
        // next_action_offset is from current action_walker position
        get_sleb128(action_walker, next_action_offset);

        // negative are 'filters' which we don't use
        if (!(ti_offset >= 0))
            fatalerror("Filter actions are unsupported");

        // zero means cleanup, which we require to be the last action
        if (ti_offset == 0)
        {
            if (!(next_action_offset == 0))
                fatalerror("Cleanup action must be last in chain");
            return _d_eh_install_finally_context(actions, landing_pad, exception_struct, context);
        }

        // get classinfo for action and check if the one in the
        // exception structure is a base
        size_t catch_ci_ptr;
        get_encoded_value(classinfo_table - ti_offset * ci_size, catch_ci_ptr, classinfo_table_encoding, context);
        debug(EH_personality) printf(" -- catch_ci_ptr: %p\n", catch_ci_ptr);
        ClassInfo catch_ci = cast(ClassInfo)cast(void*)catch_ci_ptr;
        debug(EH_personality) printf("Comparing catch %s to exception %s\n", catch_ci.name.ptr, exception_struct.exception_object.classinfo.name.ptr);
        if (_d_isbaseof(exception_struct.exception_object.classinfo, catch_ci))
            return _d_eh_install_catch_context(actions, ti_offset, landing_pad, exception_struct, context);

        debug(EH_personality) printf(" -- Type mismatch, next action offset: %tx\n", next_action_offset);
        // we've walked through all actions and found nothing...
        if (next_action_offset == 0)
            return _Unwind_Reason_Code.CONTINUE_UNWIND;
        else
            action_walker += next_action_offset;
    }
}

// These are the register numbers for SetGR that
// llvm's eh.exception and eh.selector intrinsics
// will pick up.
// Hints for these can be found by looking at the
// EH_RETURN_DATA_REGNO macro in GCC, careful testing
// is required though.
//
// If you have a native gcc you can try the following:
// #include <stdio.h>
//
// int main(int argc, char *argv[])
// {
//     printf("EH_RETURN_DATA_REGNO(0) = %d\n", __builtin_eh_return_data_regno(0));
//     printf("EH_RETURN_DATA_REGNO(1) = %d\n", __builtin_eh_return_data_regno(1));
//     return 0;
// }
version (X86_64)
{
    private enum eh_exception_regno = 0;
    private enum eh_selector_regno = 1;
}
else version (PPC64)
{
    private enum eh_exception_regno = 3;
    private enum eh_selector_regno = 4;
}
else version (PPC)
{
    private enum eh_exception_regno = 3;
    private enum eh_selector_regno = 4;
}
else version (MIPS64)
{
    private enum eh_exception_regno = 4;
    private enum eh_selector_regno = 5;
}
else version (ARM)
{
    private enum eh_exception_regno = 0;
    private enum eh_selector_regno = 1;
}
else version (AArch64)
{
    private enum eh_exception_regno = 0;
    private enum eh_selector_regno = 1;
}
else
{
    private enum eh_exception_regno = 0;
    private enum eh_selector_regno = 2;
}

private _Unwind_Reason_Code _d_eh_install_catch_context(_Unwind_Action actions, ptrdiff_t switchval, ptrdiff_t landing_pad, _d_exception* exception_struct, _Unwind_Context_Ptr context)
{
    debug(EH_personality) printf("Found catch clause!\n");

    if (actions & _Unwind_Action.SEARCH_PHASE)
        return _Unwind_Reason_Code.HANDLER_FOUND;

    else if (actions & _Unwind_Action.CLEANUP_PHASE)
    {
        debug(EH_personality) printf("Setting switch value to: %d!\n", switchval);
        _Unwind_SetGR(context, eh_exception_regno, cast(ptrdiff_t)exception_struct);
        _Unwind_SetGR(context, eh_selector_regno, cast(ptrdiff_t)switchval);
        _Unwind_SetIP(context, landing_pad);
        return _Unwind_Reason_Code.INSTALL_CONTEXT;
    }

    fatalerror("reached unreachable");
    return _Unwind_Reason_Code.FATAL_PHASE2_ERROR;
}

private _Unwind_Reason_Code _d_eh_install_finally_context(_Unwind_Action actions, ptrdiff_t landing_pad, _d_exception* exception_struct, _Unwind_Context_Ptr context)
{
    // if we're merely in search phase, continue
    if (actions & _Unwind_Action.SEARCH_PHASE)
        return _Unwind_Reason_Code.CONTINUE_UNWIND;

    debug(EH_personality) printf("Calling cleanup routine...\n");

    _Unwind_SetGR(context, eh_exception_regno, cast(ptrdiff_t)exception_struct);
    _Unwind_SetGR(context, eh_selector_regno, 0);
    _Unwind_SetIP(context, landing_pad);
    return _Unwind_Reason_Code.INSTALL_CONTEXT;
}

private void _d_getLanguageSpecificTables(_Unwind_Context_Ptr context, ref ubyte* callsite, ref ubyte* action, ref ubyte* classinfo_table, ref ubyte ciEncoding)
{
    ubyte* data = cast(ubyte*)_Unwind_GetLanguageSpecificData(context);
    if (data is null)
    {
        debug(EH_personality) printf("language specific data was null\n");
        callsite = null;
        action = null;
        classinfo_table = null;
        return;
    }
    debug(EH_personality) printf(" -- LSDA: %p\n", data);

    //TODO: Do proper DWARF reading here
    if (*data++ != _DW_EH_Format.DW_EH_PE_omit)
        fatalerror("DWARF header has unexpected format 1");

    ciEncoding = *data++;
    if (ciEncoding == _DW_EH_Format.DW_EH_PE_omit)
        fatalerror("Language Specific Data does not contain Types Table");
    version (ARM) version (linux) {
        with (_DW_EH_Format) {
            ciEncoding = DW_EH_PE_pcrel | DW_EH_PE_indirect;
        }
    }

    size_t cioffset;
    data = get_uleb128(data, cioffset);
    classinfo_table = data + cioffset;

    if (*data++ != _DW_EH_Format.DW_EH_PE_udata4)
        fatalerror("DWARF header has unexpected format 2");
    size_t callsitelength;
    data = get_uleb128(data, callsitelength);
    action = data + callsitelength;

    callsite = data;

    debug(EH_personality) printf(" -- callsite: %p, action: %p, classinfo_table: %p, ciEncoding: %d\n", callsite, action, classinfo_table, ciEncoding);
}

} // end of x86 Linux specific implementation


extern(C) void _d_throw_exception(Object e)
{
    if (e !is null)
    {
        _d_exception* exc_struct = new _d_exception;
        version (ARM)
        {
            exc_struct.unwind_info.exception_class = _d_exception_class;
        }
        else
        {
            exc_struct.unwind_info.exception_class = *cast(ulong*)_d_exception_class.ptr;
        }
        exc_struct.exception_object = e;
        debug(EH_personality) printf("throw exception %p\n", e);
        _Unwind_Reason_Code ret = _Unwind_RaiseException(&exc_struct.unwind_info);
        fprintf(stderr, "_Unwind_RaiseException failed with reason code: %d\n", ret);
    }
    abort();
}

extern(C) void _d_eh_resume_unwind(_d_exception* exception_struct)
{
    _Unwind_Resume(&exception_struct.unwind_info);
}

extern(C) void _d_eh_handle_collision(_d_exception* exception_struct, _d_exception* inflight_exception_struct)
{
    Throwable h = cast(Throwable)exception_struct.exception_object;
    Throwable inflight = cast(Throwable)inflight_exception_struct.exception_object;

    auto e = cast(Error)h;
    if (e !is null && (cast(Error)inflight) is null)
    {
        debug(EH_personality) printf("new error %p bypassing inflight %p\n", h, inflight);
        e.bypassedException = inflight;
    }
    else if (inflight != h)
    {
        debug(EH_personality) printf("replacing thrown %p with inflight %p\n", h, inflight);
        auto n = inflight;
        while (n.next)
            n = n.next;
        n.next = h;
        exception_struct = inflight_exception_struct;
    }

    _d_eh_resume_unwind(exception_struct);
}
