module ldc.eh2;

/**
 * Windows SEH exception handling.
 *
 * Copyright: Copyright The LDC Developers 2013
 * License:   <a href="http://www.boost.org/LICENSE_1_0.txt">Boost License 1.0</a>.
 * Authors:   Kai Nacke <kai@redstar.de>
 */

/*          Copyright The LDC Developers 2013.
 * Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 */

private:

import core.sys.windows.windows;
import core.stdc.stdlib; // abort
import core.stdc.stdio;

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
    UNWIND_HISTORY_TABLE_ENTRY Entry[UNWIND_HISTORY_TABLE_SIZE];
}

extern(Windows) void RtlUnwindEx(PVOID TargetFrame,
                                 PVOID TargetIp,
                                 EXCEPTION_RECORD *ExceptionRecord,
                                 PVOID ReturnValue,
                                 CONTEXT *OriginalContext,
                                 UNWIND_HISTORY_TABLE *HistoryTable);
extern(Windows) void RtlRaiseException(EXCEPTION_RECORD *ExceptionRecord);

// D runtime functions
extern(C) int _d_isbaseof(ClassInfo oc, ClassInfo c);

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

extern(C) EXCEPTION_DISPOSITION _d_eh_personality(EXCEPTION_RECORD *ExceptionRecord,
                                                  void *EstablisherFrame,
                                                  CONTEXT *ContextRecord,
                                                  DISPATCHER_CONTEXT *DispatcherContext)
{

    DISPATCHER_CONTEXT *dispatch = cast(DISPATCHER_CONTEXT *) DispatcherContext;

    ubyte *ptr = cast(ubyte*) dispatch.HandlerData;
    ubyte* callsite_table;
    ubyte* action_table;
    ubyte* classinfo_table;
    ubyte classinfo_table_encoding;
    _d_getLanguageSpecificTables(cast(ubyte*) dispatch.HandlerData, callsite_table, action_table, classinfo_table, classinfo_table_encoding);
    if (callsite_table is null)
        return EXCEPTION_DISPOSITION.ExceptionContinueSearch;

    /*
      find landing pad and action table index belonging to ip by walking
      the callsite_table
    */
    ubyte* callsite_walker = callsite_table;

    // get the instruction pointer
    // will be used to find the right entry in the callsite_table
    // -1 because it will point past the last instruction
    ptrdiff_t ip = cast(ptrdiff_t)(dispatch.ControlPc - 1);

    // address block_start is relative to
    ptrdiff_t region_start = cast(ptrdiff_t)dispatch.ImageBase +
                           dispatch.FunctionEntry.BeginAddress;

    // table entries
    uint block_start_offset, block_size;
    ptrdiff_t landing_pad;
    size_t action_offset;

    while (true)
    {
        // if we've gone through the list and found nothing...
        if (callsite_walker >= action_table)
            return EXCEPTION_DISPOSITION.ExceptionContinueSearch;

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
            return EXCEPTION_DISPOSITION.ExceptionContinueSearch;

        // if we've found our block, exit
        if (ip < region_start + block_start_offset + block_size)
            break;
    }

    debug(EH_personality) printf("Found correct landing pad and actionOffset %d\n", action_offset);

    Object excobj = cast(Object)cast(void*)ExceptionRecord.ExceptionInformation[0];

    // if there's no action offset and no landing pad, continue searching/unwinding
    if (!action_offset && !landing_pad)
        return EXCEPTION_DISPOSITION.ExceptionContinueSearch;

    // if there's no action offset but a landing pad, this is a cleanup handler
    if (!action_offset && landing_pad)
    {
        if (ExceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND)
        {
            ContextRecord.Rdx = 0; // Selector value for cleanup
            return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
        }
        else if (ExceptionRecord.ExceptionFlags & EXCEPTION_UNWIND)
        {
            fatalerror("_d_eh_personality: EXCEPTION_UNWIND and cleanup");
        }
        else
        {
            RtlUnwindEx(EstablisherFrame, cast(PVOID) landing_pad, ExceptionRecord, cast(PVOID) excobj, ContextRecord, dispatch.HistoryTable);
            fatalerror("_d_eh_personality: RtlUnwindEx failed");
        }
    }

    /*
    walk action table chain, comparing classinfos using _d_isbaseof
    */
    ubyte* action_walker = action_table + action_offset - 1;

    size_t ci_size = get_size_of_encoded_value(classinfo_table_encoding);

    ptrdiff_t ti_offset, next_action_offset;
    while (true)
    {
        action_walker = get_sleb128(action_walker, ti_offset);
        // it is intentional that we not modify action_walker here
        // next_action_offset is from current action_walker position
        get_sleb128(action_walker, next_action_offset);

        // negative are 'filters' which we don't use
        if (!(ti_offset >= 0))
            fatalerror("_d_eh_personality: Filter actions are unsupported");

        // zero means cleanup, which we require to be the last action
        if (ti_offset == 0)
        {
            if (!(next_action_offset == 0))
                fatalerror("_d_eh_personality: Cleanup action must be last in chain");

            if (ExceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND)
            {
                ContextRecord.Rdx = ti_offset; // Selector value for cleanup
                return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
            }
            else if (ExceptionRecord.ExceptionFlags & EXCEPTION_UNWIND)
            {
                fatalerror("_d_eh_personality: EXCEPTION_UNWIND and cleanup");
            }
            else
            {
                RtlUnwindEx(EstablisherFrame, cast(PVOID) landing_pad, ExceptionRecord, cast(PVOID) excobj, ContextRecord, dispatch.HistoryTable);
                fatalerror("_d_eh_personality: RtlUnwindEx failed");
            }
        }

        // get classinfo for action and check if the one in the
        // exception structure is a base
        size_t catch_ci_ptr;
        get_encoded_value(classinfo_table - ti_offset * ci_size, catch_ci_ptr, classinfo_table_encoding /*, context*/);
        ClassInfo catch_ci = cast(ClassInfo)cast(void*)catch_ci_ptr;
        debug(EH_personality) printf("Comparing catch %s to exception %s\n", catch_ci.name.ptr, excobj.classinfo.name.ptr);
        if (_d_isbaseof(excobj.classinfo, catch_ci))
        {
            if (ExceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND)
            {
                ContextRecord.Rdx = ti_offset; // Selector value for cleanup
                return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
            }
            else if (ExceptionRecord.ExceptionFlags & EXCEPTION_UNWIND)
            {
                fatalerror("_d_eh_personality: EXCEPTION_UNWIND and catch");
            }
            else
            {
                RtlUnwindEx(EstablisherFrame, cast(PVOID) landing_pad, ExceptionRecord, cast(PVOID) excobj, ContextRecord, dispatch.HistoryTable);
                fatalerror("_d_eh_personality: RtlUnwindEx failed");
            }
        }

        // we've walked through all actions and found nothing...
        if (next_action_offset == 0)
            return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
        else
            action_walker += next_action_offset;
    }
}


public:

extern(C) void _d_throw_exception(Object e)
{
    debug(EH_personality) printf("Calling _d_throw_exception = %p e = %p\n", &_d_throw_exception, e);
    if (e !is null)
    {
        EXCEPTION_RECORD ExceptionRecord;

        // Initialize exception information
        ExceptionRecord.ExceptionCode = STATUS_LDC_D_EXCEPTION;
        ExceptionRecord.ExceptionFlags = EXCEPTION_NONCONTINUABLE;
        ExceptionRecord.ExceptionAddress = &_d_throw_exception;
        ExceptionRecord.NumberParameters = 1;
        ExceptionRecord.ExceptionInformation[0] = cast(ULONG_PTR) cast(void*) e;

        // Raise exception
        RtlRaiseException(&ExceptionRecord);

        // This is only reached in case we did something seriously wrong
        fprintf(stderr, "_d_throw_exception: RtlRaiseException failed");
    }
    else
        fprintf(stderr, "_d_throw_exception: No exception object provided");
    abort();
}

extern(C) void _d_eh_resume_unwind(Object e)
{
    debug(EH_personality) printf("Calling _d_eh_resume_unwind = %p e = %p\n", &_d_eh_resume_unwind, e);
    if (e !is null)
    {
        EXCEPTION_RECORD ExceptionRecord;

        // Initialize exception information
        ExceptionRecord.ExceptionCode = STATUS_LDC_D_EXCEPTION;
        ExceptionRecord.ExceptionFlags = EXCEPTION_NONCONTINUABLE;
        ExceptionRecord.ExceptionAddress = &_d_eh_resume_unwind;
        ExceptionRecord.NumberParameters = 1;
        ExceptionRecord.ExceptionInformation[0] = cast(ULONG_PTR) cast(void*) e;

        // Raise exception
        RtlRaiseException(&ExceptionRecord);

        // This is only reached in case we did something seriously wrong
        fprintf(stderr, "_d_eh_resume_unwind: RtlRaiseException failed");
    }
    else
        fprintf(stderr, "_d_eh_resume_unwind: No exception object provided");
    abort();
}

extern(C) void _d_eh_handle_collision(Object* exception_struct, Object* inflight_exception_struct)
{
    fprintf(stderr, "_d_eh_handle_collision: Not yet implemented");
    abort();
}

/*----------------------------------------------------------------------------*/
/* FIXME: The rest of the module is more or less shared with eh.d             */
/*----------------------------------------------------------------------------*/

private void _d_getLanguageSpecificTables(/*_Unwind_Context_Ptr context*/ ubyte *data, ref ubyte* callsite, ref ubyte* action, ref ubyte* ci, ref ubyte ciEncoding)
{
//    ubyte* data = cast(ubyte*)_Unwind_GetLanguageSpecificData(context);
    if (data is null)
    {
        //printf("language specific data was null\n");
        callsite = null;
        action = null;
        ci = null;
        return;
    }

    //TODO: Do proper DWARF reading here
    if (*data++ != _DW_EH_Format.DW_EH_PE_omit)
        fatalerror("DWARF header has unexpected format 1");

    ciEncoding = *data++;
    if (ciEncoding == _DW_EH_Format.DW_EH_PE_omit)
        fatalerror("Language Specific Data does not contain Types Table");

    size_t cioffset;
    data = get_uleb128(data, cioffset);
    ci = data + cioffset;

    if (*data++ != _DW_EH_Format.DW_EH_PE_udata4)
        fatalerror("DWARF header has unexpected format 2");
    size_t callsitelength;
    data = get_uleb128(data, callsitelength);
    action = data + callsitelength;

    callsite = data;
}

/*----------------------------------------------------------------------------*/
import core.stdc.stdarg;

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

/*----------------------------------------------------------------------------*/

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

private ubyte* get_encoded_value(ubyte* addr, ref size_t res, ubyte encoding /*, _Unwind_Context_Ptr context*/)
{
    ubyte* old_addr = addr;
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
            fatalerror("Not yet implemented."); //res += cast(size_t)_Unwind_GetRegionStart(context);
            break;
        case _DW_EH_Format.DW_EH_PE_textrel:
            fatalerror("Not yet implemented."); //res += cast(size_t)_Unwind_GetTextRelBase(context);
            break;
        case _DW_EH_Format.DW_EH_PE_datarel:
            fatalerror("Not yet implemented."); //res += cast(size_t)_Unwind_GetDataRelBase(context);
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
