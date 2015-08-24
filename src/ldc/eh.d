/**
 * This module contains functions and structures required for
 * exception handling.
 */
module ldc.eh;

private:

import core.memory : GC;
import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.stdarg;

// debug = EH_personality;
// debug = EH_personality_verbose;

version (PPC)   version = PPC_Any;
version (PPC64) version = PPC_Any;

// current EH implementation works on x86
// if it has a working unwind runtime
version (X86)
{
    version(linux) version=GCC_UNWIND;
    version(darwin) version=GCC_UNWIND;
    version(Solaris) version=GCC_UNWIND;
    version(FreeBSD) version=GCC_UNWIND;
    version(MinGW) version=GCC_UNWIND;
    enum stackGrowsDown = true;
}
version (X86_64)
{
    version(linux) version=GCC_UNWIND;
    version(darwin) version=GCC_UNWIND;
    version(Solaris) version=GCC_UNWIND;
    version(FreeBSD) version=GCC_UNWIND;
    version(MinGW) version=GCC_UNWIND;
    enum stackGrowsDown = true;
}
version (ARM)
{
    // FIXME: Almost certainly wrong.
    version (linux) version = GCC_UNWIND;
    version (FreeBSD) version = GCC_UNWIND;
    enum stackGrowsDown = true;
}
version (AArch64)
{
    version (linux) version = GCC_UNWIND;
    enum stackGrowsDown = true;
}
version (PPC_Any)
{
    version (linux) version = GCC_UNWIND;
    enum stackGrowsDown = true;
}
version (MIPS)
{
    version (linux) version = GCC_UNWIND;
    enum stackGrowsDown = true;
}
version (MIPS64)
{
    version (linux) version = GCC_UNWIND;
    enum stackGrowsDown = true;
}

// D runtime functions
extern(C) int _d_isbaseof(ClassInfo oc, ClassInfo c);

// platform-specific C headers
extern(C)
{

version (GCC_UNWIND)
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

    alias void* _Unwind_Context_Ptr;

    alias void function(_Unwind_Reason_Code, _Unwind_Exception*) _Unwind_Exception_Cleanup_Fn;

    struct _Unwind_Exception
    {
        ulong exception_class;
        _Unwind_Exception_Cleanup_Fn exception_cleanup;
        ptrdiff_t private_1;
        ptrdiff_t private_2;
    }

    ptrdiff_t _Unwind_GetLanguageSpecificData(_Unwind_Context_Ptr context);
    ptrdiff_t _Unwind_GetCFA(_Unwind_Context_Ptr context);
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
else version (Win64)
{
    import core.sys.windows.windows;

    // Missing in core.sys.windows.windows
    alias ulong ULONG64;

    extern(Windows) void RaiseException(DWORD dwExceptionCode,
                                        DWORD dwExceptionFlags,
                                        DWORD nNumberOfArguments,
                                        ULONG_PTR *lpArguments);

    // Exception disposition return values
    enum EXCEPTION_DISPOSITION
    {
        ExceptionContinueExecution,
        ExceptionContinueSearch,
        ExceptionNestedException,
        ExceptionCollidedUnwind
    }

    enum : DWORD
    {
        STATUS_WAIT_0                      = 0,
        STATUS_ABANDONED_WAIT_0            = 0x00000080,
        STATUS_USER_APC                    = 0x000000C0,
        STATUS_TIMEOUT                     = 0x00000102,
        STATUS_PENDING                     = 0x00000103,

        STATUS_SEGMENT_NOTIFICATION        = 0x40000005,
        STATUS_GUARD_PAGE_VIOLATION        = 0x80000001,
        STATUS_DATATYPE_MISALIGNMENT       = 0x80000002,
        STATUS_BREAKPOINT                  = 0x80000003,
        STATUS_SINGLE_STEP                 = 0x80000004,

        STATUS_ACCESS_VIOLATION            = 0xC0000005,
        STATUS_IN_PAGE_ERROR               = 0xC0000006,
        STATUS_INVALID_HANDLE              = 0xC0000008,

        STATUS_NO_MEMORY                   = 0xC0000017,
        STATUS_ILLEGAL_INSTRUCTION         = 0xC000001D,
        STATUS_NONCONTINUABLE_EXCEPTION    = 0xC0000025,
        STATUS_INVALID_DISPOSITION         = 0xC0000026,
        STATUS_ARRAY_BOUNDS_EXCEEDED       = 0xC000008C,
        STATUS_FLOAT_DENORMAL_OPERAND      = 0xC000008D,
        STATUS_FLOAT_DIVIDE_BY_ZERO        = 0xC000008E,
        STATUS_FLOAT_INEXACT_RESULT        = 0xC000008F,
        STATUS_FLOAT_INVALID_OPERATION     = 0xC0000090,
        STATUS_FLOAT_OVERFLOW              = 0xC0000091,
        STATUS_FLOAT_STACK_CHECK           = 0xC0000092,
        STATUS_FLOAT_UNDERFLOW             = 0xC0000093,
        STATUS_INTEGER_DIVIDE_BY_ZERO      = 0xC0000094,
        STATUS_INTEGER_OVERFLOW            = 0xC0000095,
        STATUS_PRIVILEGED_INSTRUCTION      = 0xC0000096,
        STATUS_STACK_OVERFLOW              = 0xC00000FD,
        STATUS_CONTROL_C_EXIT              = 0xC000013A,
        STATUS_DLL_INIT_FAILED             = 0xC0000142,
        STATUS_DLL_INIT_FAILED_LOGOFF      = 0xC000026B,

        CONTROL_C_EXIT                     = STATUS_CONTROL_C_EXIT,

        EXCEPTION_ACCESS_VIOLATION         = STATUS_ACCESS_VIOLATION,
        EXCEPTION_DATATYPE_MISALIGNMENT    = STATUS_DATATYPE_MISALIGNMENT,
        EXCEPTION_BREAKPOINT               = STATUS_BREAKPOINT,
        EXCEPTION_SINGLE_STEP              = STATUS_SINGLE_STEP,
        EXCEPTION_ARRAY_BOUNDS_EXCEEDED    = STATUS_ARRAY_BOUNDS_EXCEEDED,
        EXCEPTION_FLT_DENORMAL_OPERAND     = STATUS_FLOAT_DENORMAL_OPERAND,
        EXCEPTION_FLT_DIVIDE_BY_ZERO       = STATUS_FLOAT_DIVIDE_BY_ZERO,
        EXCEPTION_FLT_INEXACT_RESULT       = STATUS_FLOAT_INEXACT_RESULT,
        EXCEPTION_FLT_INVALID_OPERATION    = STATUS_FLOAT_INVALID_OPERATION,
        EXCEPTION_FLT_OVERFLOW             = STATUS_FLOAT_OVERFLOW,
        EXCEPTION_FLT_STACK_CHECK          = STATUS_FLOAT_STACK_CHECK,
        EXCEPTION_FLT_UNDERFLOW            = STATUS_FLOAT_UNDERFLOW,
        EXCEPTION_INT_DIVIDE_BY_ZERO       = STATUS_INTEGER_DIVIDE_BY_ZERO,
        EXCEPTION_INT_OVERFLOW             = STATUS_INTEGER_OVERFLOW,
        EXCEPTION_PRIV_INSTRUCTION         = STATUS_PRIVILEGED_INSTRUCTION,
        EXCEPTION_IN_PAGE_ERROR            = STATUS_IN_PAGE_ERROR,
        EXCEPTION_ILLEGAL_INSTRUCTION      = STATUS_ILLEGAL_INSTRUCTION,
        EXCEPTION_NONCONTINUABLE_EXCEPTION = STATUS_NONCONTINUABLE_EXCEPTION,
        EXCEPTION_STACK_OVERFLOW           = STATUS_STACK_OVERFLOW,
        EXCEPTION_INVALID_DISPOSITION      = STATUS_INVALID_DISPOSITION,
        EXCEPTION_GUARD_PAGE               = STATUS_GUARD_PAGE_VIOLATION,
        EXCEPTION_INVALID_HANDLE           = STATUS_INVALID_HANDLE
    }

    // Exception record flag
    enum : DWORD
    {
        EXCEPTION_NONCONTINUABLE           = 0x01,
        EXCEPTION_UNWINDING                = 0x02,
        EXCEPTION_EXIT_UNWIND              = 0x04,
        EXCEPTION_STACK_INVALID            = 0x08,
        EXCEPTION_NESTED_CALL              = 0x10,
        EXCEPTION_TARGET_UNWIND            = 0x20,
        EXCEPTION_COLLIDED_UNWIND          = 0x40,

        EXCEPTION_UNWIND                   = EXCEPTION_UNWINDING
                                             | EXCEPTION_EXIT_UNWIND
                                             | EXCEPTION_TARGET_UNWIND
                                             | EXCEPTION_COLLIDED_UNWIND
    }

    // Maximum number of exception parameters
    enum size_t EXCEPTION_MAXIMUM_PARAMETERS = 15;

    struct EXCEPTION_RECORD
    {
        DWORD ExceptionCode;
        DWORD ExceptionFlags;
        EXCEPTION_RECORD* ExceptionRecord;
        PVOID ExceptionAddress;
        DWORD NumberParameters;
        version(Win64)
        {
            DWORD __unusedAlignment;
        }
        ULONG_PTR[EXCEPTION_MAXIMUM_PARAMETERS] ExceptionInformation;
    }

    struct DISPATCHER_CONTEXT
    {
        PVOID ControlPc;
        PVOID ImageBase;
        RUNTIME_FUNCTION *FunctionEntry;
        PVOID EstablisherFrame;
        PVOID TargetIp;
        CONTEXT *ContextRecord;
        EXCEPTION_ROUTINE *LanguageHandler;
        PVOID HandlerData;
        UNWIND_HISTORY_TABLE *HistoryTable;
    }

    struct RUNTIME_FUNCTION
    {
        DWORD BeginAddress;
        DWORD EndAddress;
        DWORD UnwindData;
    }

    alias extern(C) EXCEPTION_DISPOSITION function(EXCEPTION_RECORD *ExceptionRecord,
                                                   void *EstablisherFrame,
                                                   CONTEXT *ContextRecord,
                                                   DISPATCHER_CONTEXT *DispatcherContext) EXCEPTION_ROUTINE;

    // Make our own exception code
    enum int STATUS_LDC_D_EXCEPTION = (3 << 30) // Severity = error
                                      | (1 << 29) // User defined exception
                                      | (0 << 28) // Reserved
                                      | ('L' << 16)
                                      | ('D' << 8)
                                      | ('C' << 0);

    struct UNWIND_HISTORY_TABLE_ENTRY
    {
        ULONG64 ImageBase;
        RUNTIME_FUNCTION *FunctionEntry;
    }

    enum size_t UNWIND_HISTORY_TABLE_SIZE = 12;

    struct UNWIND_HISTORY_TABLE
    {
        ULONG Count;
        UCHAR Search;
        ULONG64 LowAddress;
        ULONG64 HighAddress;
        UNWIND_HISTORY_TABLE_ENTRY[UNWIND_HISTORY_TABLE_SIZE] Entry;
    }

    extern(Windows) void RtlUnwindEx(PVOID TargetFrame,
                                     PVOID TargetIp,
                                     EXCEPTION_RECORD *ExceptionRecord,
                                     PVOID ReturnValue,
                                     CONTEXT *OriginalContext,
                                     UNWIND_HISTORY_TABLE *HistoryTable);
    extern(Windows) void RtlRaiseException(EXCEPTION_RECORD *ExceptionRecord);
}
else // !GCC_UNWIND && !Win64
{
    static assert(0, "Not implemented on this platform");
}

} // extern(C)

// error and exit
extern(C) void fatalerror(in char* format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "Fatal error in EH code: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    abort();
}


// helpers for reading certain DWARF data
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

ubyte* get_uleb128(ubyte* addr, ref size_t res)
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

ubyte* get_sleb128(ubyte* addr, ref ptrdiff_t res)
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

size_t get_size_of_encoded_value(ubyte encoding)
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

ubyte* get_encoded_value(ubyte* addr, ref size_t res, ubyte encoding, void* context)
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
            version(Win64) fatalerror("Not yet implemented."); else
            res += cast(size_t)_Unwind_GetRegionStart(context);
            break;
        case _DW_EH_Format.DW_EH_PE_textrel:
            version(Win64) fatalerror("Not yet implemented."); else
            res += cast(size_t)_Unwind_GetTextRelBase(context);
            break;
        case _DW_EH_Format.DW_EH_PE_datarel:
            version(Win64) fatalerror("Not yet implemented."); else
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

ptrdiff_t get_base_of_encoded_value(ubyte encoding, void* context)
{
    if (encoding == _DW_EH_Format.DW_EH_PE_omit)
        return 0;

    with (_DW_EH_Format) switch (encoding & 0x70) {
        case DW_EH_PE_absptr:
        case DW_EH_PE_pcrel:
        case DW_EH_PE_aligned:
            return 0;

      version(Win64) {} else
      {
        case DW_EH_PE_textrel:
            return _Unwind_GetTextRelBase (context);
        case DW_EH_PE_datarel:
            return _Unwind_GetDataRelBase (context);
        case DW_EH_PE_funcrel:
            return _Unwind_GetRegionStart (context);
      }

        default:
            fatalerror("Unsupported encoding type to get base from.");
            assert(0);
    }
}

void _d_getLanguageSpecificTables(ubyte* data, ref ubyte* callsite, ref ubyte* action, ref ubyte* classinfo_table, ref ubyte ciEncoding)
{
    if (data is null)
    {
        debug(EH_personality) printf("language specific data was null\n");
        callsite = null;
        action = null;
        classinfo_table = null;
        return;
    }
    debug(EH_personality) printf("  - LSDA: %p\n", data);

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

    debug(EH_personality) printf("  - callsite: %p, action: %p, classinfo_table: %p, ciEncoding: %d\n", callsite, action, classinfo_table, ciEncoding);
}


struct ActiveCleanupBlock {
    /// Link to the next active finally block.
    ActiveCleanupBlock* outerBlock;

    /// The exception that caused this cleanup block to be entered.
    Object dObject;

    /// The CFA (stack address, roughly) when this cleanup block was entered, as
    /// reported by libunwind.
    ///
    /// Used to determine when this pad is reached again when unwinding from
    /// somewhere within it. Note that this must somehow be related to the
    /// stack, not the instruction pointer, to properly support recursive
    /// chaining.
    ptrdiff_t cfaAddr;
}

/// Stack of active finally blocks (i.e. cleanup landing pads) that were entered
/// because of exception unwinding. Used for exception chaining.
///
/// Note that sometimes a landing pad is both a catch and a cleanup. This
/// happens for example when there is a try/finally nested inside a try/catch
/// in the same function, or has been inlined into one. Whether a catch will
/// actually execute (which terminates this strand of unwinding) or not is
/// determined by the program code and cannot be known inside the personality
/// routine. Thus, we always push such a block even before entering a catch(),
/// and have the user code call _d_eh_enter_catch() once the (possible) cleanup
/// part is done so we can pop it again. In theory, this could be optimized a
/// bit, because we only need to do it for landing pads which have this double
/// function, which we could possibly figure out form the DWARF tables. However,
/// since this makes generating the code for popping it non-trivial, this is not
/// currently done.
ActiveCleanupBlock* innermostCleanupBlock = null;

/// During the search phase of unwinding, points to the currently active cleanup
/// block (i.e. somewhere in the innermostCleanupBlock linked list, but possibly
/// not at the beginning if the search for the catch block has already continued
/// past that).
///
/// Note that this and searchPhaseCurrentCleanupBlock can just be a single
/// variable because there can never be more than one search phase running per
/// thread.
ActiveCleanupBlock* searchPhaseCurrentCleanupBlock = null;

/// During the search phase, keeps track of the type of the dynamic type of the
/// currently thrown exception (might change according to exception chaining
/// rules).
ClassInfo searchPhaseClassInfo = null;

void pushCleanupBlockRecord(ptrdiff_t cfaAddr, Object dObject)
{
    auto acb = cast(ActiveCleanupBlock*)malloc(ActiveCleanupBlock.sizeof);
    if (!acb)
    {
        // TODO: Allocate some statically to avoid problem with unwinding out of
        // memory errors. A fairly small amount of memory should suffice for
        // most applications unless people want to unwind through very deeply
        // recursive code with many finally blocks.
        fatalerror("Could not allocate memory for exception chaining.");
    }
    acb.cfaAddr = cfaAddr;
    acb.dObject = dObject;
    acb.outerBlock = innermostCleanupBlock;
    innermostCleanupBlock = acb;

    // We need to be sure that an in-flight exception is kept alive while
    // executing a finally block. This is not automatically the case if the
    // finally block always throws, because the compiler then does not need to
    // keep a reference to the object extracted from the landing pad around as
    // there is no _d_eh_resume_unwind() call.
    GC.addRoot(cast(void*)dObject);
}

void popCleanupBlockRecord()
{
    if (!innermostCleanupBlock)
    {
        fatalerror("No cleanup block record found, should have been pushed " ~
            "before entering the finally block.");
    }
    // Remove the cleanup block we installed for this handler.
    auto acb = innermostCleanupBlock;
    GC.removeRoot(cast(void*)acb.dObject);
    innermostCleanupBlock = acb.outerBlock;
    free(acb);
}


/// This is the implementation of the personality function, which is called by
/// libunwind twice per frame (search phase, unwind phase).
///
/// It is responsible to figure out whether we need to stop unwinding because of
/// a catch block or if there is a finally block to execute by reading the DWARF
/// EH tables.
extern(C) auto eh_personality_common(NativeContext)(ref NativeContext nativeContext)
{
    //
    // First, we need to find the Language-Specific Data table for this frame
    // and extract our information tables (the "language-specific" part is a bit
    // of a misnomer, in reality this is generated by LLVM).
    //
    // The callsite and action tables do not contain static-length data and will
    // be parsed as needed.
    //

    ubyte* callsite_table;
    ubyte* action_table;
    ubyte* classinfo_table; // points past the end of the table
    ubyte classinfo_table_encoding;
    ubyte* data = nativeContext.getLanguageSpecificData();
    _d_getLanguageSpecificTables(data, callsite_table, action_table, classinfo_table, classinfo_table_encoding);

    if (!callsite_table)
        return nativeContext.continueUnwind();

    //
    // Now, we need to figure out if the address of the current instruction
    // in this frame corresponds to a block which has an associated landing pad.
    //

    // The instruction pointer (ip) will point to the next instruction after
    // whatever made execution leave this frame, so substract 1 for the range
    // comparison below.
    immutable ptrdiff_t ip = nativeContext.getIP() - 1;

    // The table entries are all relative to the start address of the region.
    immutable ptrdiff_t region_start = nativeContext.getRegionStart();

    // The address of the landing pad to jump to (null if no match).
    ptrdiff_t landingPadAddr;

    // The offset in the action table corresponding to the first action for this
    // landing pad (will be zero if there are none).
    size_t actionTableStartOffset;

    ubyte* callsite_walker = callsite_table;
    while (true)
    {
        // if we've gone through the list and found nothing...
        if (callsite_walker >= action_table)
            return nativeContext.continueUnwind();

        immutable block_start_offset = *cast(uint*)callsite_walker;
        immutable block_size = *(cast(uint*)callsite_walker + 1);
        landingPadAddr = *(cast(uint*)callsite_walker + 2);
        callsite_walker = get_uleb128(callsite_walker + 3 * uint.sizeof, actionTableStartOffset);

        debug(EH_personality_verbose)
        {
            printf("  - ip=%llx %d %d %llx\n", ip, block_start_offset,
                block_size, landingPadAddr);
        }

        // since the list is sorted, as soon as we're past the ip
        // there's no handler to be found
        if (ip < region_start + block_start_offset)
            return nativeContext.continueUnwind();

        // if we've found our block, exit
        if (ip < region_start + block_start_offset + block_size)
            break;
    }

    debug(EH_personality)
    {
        printf("  - Found correct landing pad and actionTableStartOffset %d\n",
            actionTableStartOffset);
    }

    // There is no landing pad for this part of the frame, continue with the next level.
    if (!landingPadAddr)
        return nativeContext.continueUnwind();

    // We have a landing pad, adjust by region start address.
    landingPadAddr += region_start;

    immutable bool isSearchPhase = nativeContext.isSearchPhase();

    //
    // We have at least a finally landing pad in this scope. First, check if we
    // have arrived at the scope a previous exception was thrown in. In this
    // case, we need to chain exception_struct.exception_object to it or replace
    // it with the former.
    //
    immutable ptrdiff_t currentCfaAddr = nativeContext.getCfaAddress();
    ref ActiveCleanupBlock* acb()
    {
        return isSearchPhase ? searchPhaseCurrentCleanupBlock : innermostCleanupBlock;
    }

    while (acb)
    {
        debug(EH_personality)
        {
            printf("  - Current CFA: %p, Previous CFA: %p\n",
                currentCfaAddr, acb.cfaAddr);
        }

        // If the next active cleanup block is somewhere further up the stack,
        // there is nothing to do/check.
        static assert(stackGrowsDown);
        if (currentCfaAddr < acb.cfaAddr)
            break;

        Object thrownDObject = nativeContext.getThrownObject();
        auto currentClassInfo = isSearchPhase ? searchPhaseClassInfo : thrownDObject.classinfo;
        if (_d_isbaseof(currentClassInfo, Error.classinfo) && !cast(Error)acb.dObject)
        {
            // The currently unwound Throwable is an Error but the previous one
            // is not, so replace the latter with the former.
            debug(EH_personality)
            {
                printf(" ++ Replacing %p (%s) by %p (%s)\n",
                    acb.dObject,
                    acb.dObject.classinfo.name.ptr,
                    thrownDObject,
                    thrownDObject.classinfo.name.ptr);
            }

            if (!isSearchPhase)
            {
                (cast(Error)thrownDObject).bypassedException =
                    cast(Throwable)acb.dObject;
            }
        }
        else
        {
            // We are just unwinding an Exception or there was already an Error,
            // so append this Throwable to the end of the previous chain.
            if (isSearchPhase)
            {
                debug(EH_personality)
                {
                    printf(" ++ Setting up classinfo to chain %s to %p (%s, classinfo at %p)\n",
                        searchPhaseClassInfo.name.ptr, acb.dObject,
                        acb.dObject.classinfo.name.ptr, acb.dObject.classinfo);
                }
                searchPhaseClassInfo = acb.dObject.classinfo;
            }
            else
            {
                auto lastChainElem = cast(Throwable)acb.dObject;
                while (lastChainElem.next)
                {
                    lastChainElem = lastChainElem.next;
                }

                auto thisThrowable = cast(Throwable)thrownDObject;
                if (lastChainElem is thisThrowable)
                {
                    // We would need to chain an exception to itself. This can
                    // happen if somebody throws the same exception object twice.
                    // It is questionable whether this is supposed to work in the
                    // first place, but core.demangle does it when generating the
                    // backtrace for its internal exceptions during demangling a
                    // symbol as part of the default trace handler (it uses the
                    // .init value for the exception class instead of allocating
                    // new instances).
                    debug(EH_personality)
                    {
                        printf(" ++ Not chaining %p (%s) to itself\n",
                            thisThrowable, thisThrowable.classinfo.name.ptr);
                    }
                }
                else
                {
                    debug(EH_personality)
                    {
                        printf(" ++ Chaining %p (%s) to %p (%s)\n",
                            thisThrowable, thisThrowable.classinfo.name.ptr,
                            lastChainElem, lastChainElem.classinfo.name.ptr);
                    }
                    lastChainElem.next = thisThrowable;
                }

                nativeContext.overrideThrownObject(acb.dObject);
            }
        }

        // In both cases, we've executed one level of chaining.
        auto outer = acb.outerBlock;
        if (!isSearchPhase)
        {
            GC.removeRoot(cast(void*)acb.dObject);
            free(acb);
        }
        acb = outer;
    }

    //
    // Exception chaining is now done. Let's figure out what we have to do in
    // this frame.
    //

    // If there are no actions, this is a cleanup landing pad.
    if (!actionTableStartOffset)
    {
        return nativeContext.installFinallyContext(landingPadAddr);
    }

    // We have at least some attached actions. Figure out whether any of them
    // match the type of the current exception.
    immutable ci_size = get_size_of_encoded_value(classinfo_table_encoding);
    debug(EH_personality) printf("  - ci_size: %td, ci_encoding: %d\n", ci_size, classinfo_table_encoding);

    ubyte* action_walker = action_table + actionTableStartOffset - 1;
    while (true)
    {
        ptrdiff_t ti_offset;
        action_walker = get_sleb128(action_walker, ti_offset);
        debug(EH_personality) printf("  - ti_offset: %tx\n", ti_offset);

        // it is intentional that we not modify action_walker here
        // next_action_offset is from current action_walker position
        ptrdiff_t next_action_offset;
        get_sleb128(action_walker, next_action_offset);

        // negative are 'filters' which we don't use
        if (!(ti_offset >= 0))
            fatalerror("Filter actions are unsupported");

        // zero means cleanup, which we require to be the last action
        if (ti_offset == 0)
        {
            if (!(next_action_offset == 0))
                fatalerror("Cleanup action must be last in chain");
            return nativeContext.installFinallyContext(landingPadAddr);
        }

        // For catch clauses, now figure out whether the types match.
        //
        // Optimization: After the search phase, libunwind lets us know whether
        // we have found a handler in this frame the first time around. We can
        // thus skip further the comparisons if the HANDLER_FRAME flag is not
        // set.
        //
        // As a further optimization step, we could look into caching that
        // result inside _d_exception.
        if (!nativeContext.skipCatchComparison())
        {
            ClassInfo catchClassInfo = nativeContext.getCatchClassInfo(
                classinfo_table - ti_offset * ci_size, classinfo_table_encoding);
            ClassInfo exceptionClassInfo = isSearchPhase ?
                searchPhaseClassInfo : nativeContext.getThrownObject().classinfo;

            debug(EH_personality)
            {
                printf("  - Comparing catch %s to exception %s\n",
                    catchClassInfo.name.ptr, exceptionClassInfo.name.ptr);
            }
            if (_d_isbaseof(exceptionClassInfo, catchClassInfo))
            {
                return nativeContext.installCatchContext(ti_offset, landingPadAddr);
            }
        }

        debug(EH_personality) printf("  - Type mismatch, next action offset: %tx\n", next_action_offset);

        if (next_action_offset == 0)
            return nativeContext.continueUnwind();
        action_walker += next_action_offset;
    }
}


//
// implementation of personality function and helpers
//
version (GCC_UNWIND)
{

// exception struct used by the runtime.
// _d_throw allocates a new instance and passes the address of its
// _Unwind_Exception member to the unwind call. The personality
// routine is then able to get the whole struct by looking at the data
// surrounding the unwind info.
//
// Note that the code we generate for the landing pads also relies on the
// throwable object being stored at offset 0.
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

_Unwind_Reason_Code continueUnwind(_Unwind_Control_Block* ucb, _Unwind_Context_Ptr context) {
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

    _Unwind_Reason_Code rc = eh_personality_common_GCC(actions, ucb.toDException(), context);
    if (rc == _Unwind_Reason_Code.CONTINUE_UNWIND)
        return continueUnwind(ucb, context);
    return rc;
}

}
else // !ARM
{

// the personality routine gets called by the unwind handler and is responsible for
// reading the EH tables and deciding what to do
extern(C) _Unwind_Reason_Code _d_eh_personality(int ver, _Unwind_Action actions, ulong exception_class, _Unwind_Exception* exception_info, _Unwind_Context_Ptr context)
{
    debug(EH_personality_verbose)
    {
        printf(" %s Entering personality function. context: %p\n",
            (actions & _Unwind_Action.SEARCH_PHASE) ? "[S]".ptr : "[U]".ptr, context);
    }

    // check ver: the C++ Itanium ABI only allows ver == 1
    if (ver != 1)
        return _Unwind_Reason_Code.FATAL_PHASE1_ERROR;

    // check exceptionClass
    //TODO: Treat foreign exceptions with more respect
    if ((cast(char*)&exception_class)[0..8] != _d_exception_class)
        return _Unwind_Reason_Code.FATAL_PHASE1_ERROR;

    _d_exception* exception_struct = cast(_d_exception*)(cast(ubyte*)exception_info - _d_exception.unwind_info.offsetof);
    return eh_personality_common_GCC(actions, exception_struct, context);
}

} // !ARM

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
    enum eh_exception_regno = 0;
    enum eh_selector_regno = 1;
}
else version (PPC64)
{
    enum eh_exception_regno = 3;
    enum eh_selector_regno = 4;
}
else version (PPC)
{
    enum eh_exception_regno = 3;
    enum eh_selector_regno = 4;
}
else version (MIPS64)
{
    enum eh_exception_regno = 4;
    enum eh_selector_regno = 5;
}
else version (ARM)
{
    enum eh_exception_regno = 0;
    enum eh_selector_regno = 1;
}
else version (AArch64)
{
    enum eh_exception_regno = 0;
    enum eh_selector_regno = 1;
}
else
{
    enum eh_exception_regno = 0;
    enum eh_selector_regno = 2;
}

struct NativeContext_GCC
{
    _Unwind_Action actions;
    _d_exception* exception_struct;
    _Unwind_Context_Ptr context;

    ubyte* getLanguageSpecificData() { return cast(ubyte*)_Unwind_GetLanguageSpecificData(context); }
    ptrdiff_t getIP() { return _Unwind_GetIP(context); }
    ptrdiff_t getRegionStart() { return _Unwind_GetRegionStart(context); }
    bool isSearchPhase() { return (actions & _Unwind_Action.SEARCH_PHASE) != 0; }
    bool skipCatchComparison() { return !isSearchPhase() && (actions & _Unwind_Action.HANDLER_FRAME) == 0; }
    ptrdiff_t getCfaAddress() { return _Unwind_GetCFA(context); }
    Object getThrownObject() { return exception_struct.exception_object; }
    void overrideThrownObject(Object newObject) { exception_struct.exception_object = newObject; }

    ClassInfo getCatchClassInfo(void* address, ubyte encoding)
    {
        size_t catchClassInfoAddr;
        get_encoded_value(cast(ubyte*)address, catchClassInfoAddr, encoding, context);
        return cast(ClassInfo)cast(void*)catchClassInfoAddr;
    }

    _Unwind_Reason_Code continueUnwind()
    {
        return _Unwind_Reason_Code.CONTINUE_UNWIND;
    }

    _Unwind_Reason_Code installCatchContext(ptrdiff_t ti_offset, ptrdiff_t landingPadAddr)
    {
        debug(EH_personality) printf("  - Found catch clause for %p\n", exception_struct);

        if (actions & _Unwind_Action.SEARCH_PHASE)
            return _Unwind_Reason_Code.HANDLER_FOUND;

        if (!(actions & _Unwind_Action.CLEANUP_PHASE))
            fatalerror("Unknown phase");

        pushCleanupBlockRecord(getCfaAddress(), getThrownObject());

        debug(EH_personality)
        {
            printf("  - Calling catch block for %p (struct at %p)\n",
                exception_struct.exception_object, exception_struct);
        }

        debug(EH_personality_verbose) printf("  - Setting switch value to: %p\n", ti_offset);
        _Unwind_SetGR(context, eh_exception_regno, cast(ptrdiff_t)exception_struct);
        _Unwind_SetGR(context, eh_selector_regno, ti_offset);

        debug(EH_personality_verbose) printf("  - Setting landing pad to: %p\n", landingPadAddr);
        _Unwind_SetIP(context, landingPadAddr);

        return _Unwind_Reason_Code.INSTALL_CONTEXT;
    }

    _Unwind_Reason_Code installFinallyContext(ptrdiff_t landingPadAddr)
    {
        if (actions & _Unwind_Action.SEARCH_PHASE)
            return _Unwind_Reason_Code.CONTINUE_UNWIND;

        pushCleanupBlockRecord(getCfaAddress(), getThrownObject());

        debug(EH_personality)
        {
            printf("  - Calling cleanup block for %p (struct at %p)\n",
                exception_struct.exception_object, exception_struct);
        }

        _Unwind_SetGR(context, eh_exception_regno, cast(ptrdiff_t)exception_struct);
        _Unwind_SetGR(context, eh_selector_regno, 0);
        _Unwind_SetIP(context, landingPadAddr);
        return _Unwind_Reason_Code.INSTALL_CONTEXT;
    }
}

extern(C) _Unwind_Reason_Code eh_personality_common_GCC(_Unwind_Action actions,
    _d_exception* exception_struct, _Unwind_Context_Ptr context)
{
    auto nativeContext = NativeContext_GCC(actions, exception_struct, context);
    return eh_personality_common(nativeContext);
}

}
else version (Win64)
{

/*
 * The interaction between the language, the LDC API and the OS is as follows:
 *
 * The exception is raised with the throw statement. The throw statement is
 * translated to a call to _d_throw_exception. This method translate the D
 * exception into an OS exception with a call to RtlRaiseException.
 *
 * The OS now searches for an exception handler. The next frame with a
 * registered exception handler (flag UNW_EHANDLER) is located with the help of
 * RtlVirtualUnwind. Then this exception handler is called. If the return value
 * is ExceptionContinueSearch then this process is repeated with the next
 * exception handler.
 * If the exception handler wants to handle the exception (a catch clause in D)
 * then he calls RtlUnwindEx with the address of the catch clause as target.
 * RtlUnwindEx unwinds the stack, calling all registered termination handlers
 * (flag UNW_UHANDLER) and finally realizing the target context.
 *
 * In order to use the OS functionality for D finally clauses we do always stop
 * at the first catch or cleanup (finally, scope) found and realize this
 * context. If the target was a catch then we are done. Otherwise,
 * _d_eh_resume_unwind is called. The function then re-raises the original
 * exception to continue the unwind.
 *
 * In general, this is a simple approach with the nice property that no
 * additional dynamic memory is consumed. On the downside chained exceptions are
 * not handled and the possible multiple raising of an exception may be not too
 * efficient.
 */

struct NativeContext_Win64
{
    EXCEPTION_RECORD* ExceptionRecord;
    void* EstablisherFrame;
    CONTEXT* ContextRecord;
    DISPATCHER_CONTEXT* dispatch;

    ubyte* getLanguageSpecificData() { return cast(ubyte*)dispatch.HandlerData; }
    ptrdiff_t getIP() { return cast(ptrdiff_t)dispatch.ControlPc; }
    ptrdiff_t getRegionStart() { return cast(ptrdiff_t)dispatch.ImageBase + dispatch.FunctionEntry.BeginAddress; }
    bool isSearchPhase() { return (ExceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND) != 0; }
    bool skipCatchComparison() { return false; }
    ptrdiff_t getCfaAddress() { return cast(ptrdiff_t)EstablisherFrame; }
    Object getThrownObject() { return cast(Object)cast(void*)ExceptionRecord.ExceptionInformation[0]; }
    void overrideThrownObject(Object newObject) { ExceptionRecord.ExceptionInformation[0] = cast(ULONG_PTR)cast(void*)newObject; }

    ClassInfo getCatchClassInfo(void* address, ubyte encoding)
    {
        size_t catchClassInfoAddr;
        get_encoded_value(cast(ubyte*)address, catchClassInfoAddr, encoding, null);
        return cast(ClassInfo)cast(void*)catchClassInfoAddr;
    }

    EXCEPTION_DISPOSITION continueUnwind()
    {
        return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
    }

    EXCEPTION_DISPOSITION installCatchContext(ptrdiff_t ti_offset, ptrdiff_t landingPadAddr)
    {
        if (ExceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND)
        {
            ContextRecord.Rdx = ti_offset; // Selector value for cleanup
            return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
        }
        else if (ExceptionRecord.ExceptionFlags & EXCEPTION_UNWIND)
        {
            fatalerror("EXCEPTION_UNWIND and catch");
        }
        else
        {
            pushCleanupBlockRecord(getCfaAddress(), getThrownObject());

            RtlUnwindEx(EstablisherFrame, cast(PVOID) landingPadAddr, ExceptionRecord,
                cast(PVOID) ExceptionRecord.ExceptionInformation[0], ContextRecord,
                dispatch.HistoryTable);
            fatalerror("RtlUnwindEx failed");
        }
        return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
    }

    EXCEPTION_DISPOSITION installFinallyContext(ptrdiff_t landingPadAddr)
    {
        if (ExceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND)
        {
            ContextRecord.Rdx = 0; // Selector value for cleanup
            return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
        }
        else if (ExceptionRecord.ExceptionFlags & EXCEPTION_UNWIND)
        {
            fatalerror("EXCEPTION_UNWIND and cleanup");
        }
        else
        {
            pushCleanupBlockRecord(getCfaAddress(), getThrownObject());

            RtlUnwindEx(EstablisherFrame, cast(PVOID) landingPadAddr, ExceptionRecord,
                cast(PVOID) ExceptionRecord.ExceptionInformation[0], ContextRecord,
                dispatch.HistoryTable);
            fatalerror("RtlUnwindEx failed");
        }
        return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
    }
}

extern(C) EXCEPTION_DISPOSITION _d_eh_personality(EXCEPTION_RECORD* ExceptionRecord,
    void* EstablisherFrame, CONTEXT* ContextRecord, DISPATCHER_CONTEXT* DispatcherContext)
{
    auto nativeContext = NativeContext_Win64(ExceptionRecord, EstablisherFrame,
        ContextRecord, DispatcherContext);
    return eh_personality_common(nativeContext);
}

} // Win64



public: extern(C):

version (GCC_UNWIND)
{

Throwable.TraceInfo _d_traceContext(void* ptr = null);

/// Called by our compiler-generated code to throw an exception.
void _d_throw_exception(Object e)
{
    if (e is null)
    {
        fatalerror("Cannot throw null exception");
    }

    if (e.classinfo is null)
    {
        fatalerror("Cannot throw corrupt exception object with null classinfo");
    }

    auto throwable = cast(Throwable) e;

    if (throwable.info is null && cast(byte*)throwable !is typeid(throwable).init.ptr)
    {
        throwable.info = _d_traceContext();
    }

    auto exc_struct = new _d_exception;
    version (ARM)
    {
        exc_struct.unwind_info.exception_class = _d_exception_class;
    }
    else
    {
        exc_struct.unwind_info.exception_class = *cast(ulong*)_d_exception_class.ptr;
    }
    exc_struct.exception_object = e;

    debug(EH_personality)
    {
        printf("= Throwing new exception of type %s: %p (struct at %p, classinfo at %p)\n",
            e.classinfo.name.ptr, e, exc_struct, e.classinfo);
    }

    searchPhaseClassInfo = e.classinfo;
    searchPhaseCurrentCleanupBlock = innermostCleanupBlock;

    // _Unwind_RaiseException should never return unless something went really
    // wrong with unwinding.
    immutable ret = _Unwind_RaiseException(&exc_struct.unwind_info);
    fatalerror("_Unwind_RaiseException failed with reason code: %d", ret);
}

/// Called by our compiler-generate code to resume unwinding after a finally
/// block (or dtor destruction block) has been run.
void _d_eh_resume_unwind(_d_exception* exception_struct)
{
    debug(EH_personality)
    {
        printf("= Returning from cleanup block for %p (struct at %p)\n",
            exception_struct.exception_object, exception_struct);
    }

    popCleanupBlockRecord();
    _Unwind_Resume(&exception_struct.unwind_info);
}

void _d_eh_enter_catch()
{
    popCleanupBlockRecord();
}

}
else version (Win64)
{

void _d_throw_exception(Object e)
{
    debug(EH_personality) printf("Calling _d_throw_exception = %p e = %p\n", &_d_throw_exception, e);

    if (e is null)
        fatalerror("Cannot throw null exception");
    if (e.classinfo is null)
        fatalerror("Cannot throw corrupt exception object with null classinfo");

    searchPhaseClassInfo = e.classinfo;
    searchPhaseCurrentCleanupBlock = innermostCleanupBlock;

    // Initialize exception information
    EXCEPTION_RECORD ExceptionRecord;
    ExceptionRecord.ExceptionCode = STATUS_LDC_D_EXCEPTION;
    ExceptionRecord.ExceptionFlags = EXCEPTION_NONCONTINUABLE;
    ExceptionRecord.ExceptionAddress = &_d_throw_exception;
    ExceptionRecord.NumberParameters = 1;
    ExceptionRecord.ExceptionInformation[0] = cast(ULONG_PTR) cast(void*) e;

    // Raise exception
    RtlRaiseException(&ExceptionRecord);

    // This is only reached in case we did something seriously wrong
    fatalerror("_d_throw_exception: RtlRaiseException failed");
}

void _d_eh_resume_unwind(Object e)
{
    debug(EH_personality) printf("Calling _d_eh_resume_unwind = %p e = %p\n", &_d_eh_resume_unwind, e);

    popCleanupBlockRecord();

    // Initialize exception information
    EXCEPTION_RECORD ExceptionRecord;
    ExceptionRecord.ExceptionCode = STATUS_LDC_D_EXCEPTION;
    ExceptionRecord.ExceptionFlags = EXCEPTION_NONCONTINUABLE;
    ExceptionRecord.ExceptionAddress = &_d_eh_resume_unwind;
    ExceptionRecord.NumberParameters = 1;
    ExceptionRecord.ExceptionInformation[0] = cast(ULONG_PTR) cast(void*) e;

    // Raise exception
    RtlRaiseException(&ExceptionRecord);

    // This is only reached in case we did something seriously wrong
    fatalerror("_d_eh_resume_unwind: RtlRaiseException failed");
}

void _d_eh_enter_catch()
{
    popCleanupBlockRecord();
}

} // Win64
