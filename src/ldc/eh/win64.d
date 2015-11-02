/**
 * This module implements the runtime-part of LDC exceptions
 * on Windows x64.
 */
module ldc.eh.win64;

version (Win64)
{

// debug = EH_personality;
// debug = EH_personality_verbose;

import ldc.eh.common;
import core.sys.windows.windows;

private:

// C headers
extern(C)
{
    // Missing in core.sys.windows.windows
    alias ulong ULONG64;

    extern(Windows) void RaiseException(DWORD dwExceptionCode,
                                        DWORD dwExceptionFlags,
                                        DWORD nNumberOfArguments,
                                        ULONG_PTR* lpArguments);

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
        RUNTIME_FUNCTION* FunctionEntry;
        PVOID EstablisherFrame;
        PVOID TargetIp;
        CONTEXT* ContextRecord;
        EXCEPTION_ROUTINE* LanguageHandler;
        PVOID HandlerData;
        UNWIND_HISTORY_TABLE* HistoryTable;
    }

    struct RUNTIME_FUNCTION
    {
        DWORD BeginAddress;
        DWORD EndAddress;
        DWORD UnwindData;
    }

    alias extern(C) EXCEPTION_DISPOSITION function(EXCEPTION_RECORD* ExceptionRecord,
                                                   void* EstablisherFrame,
                                                   CONTEXT* ContextRecord,
                                                   DISPATCHER_CONTEXT* DispatcherContext) EXCEPTION_ROUTINE;

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
        RUNTIME_FUNCTION* FunctionEntry;
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
                                     EXCEPTION_RECORD* ExceptionRecord,
                                     PVOID ReturnValue,
                                     CONTEXT* OriginalContext,
                                     UNWIND_HISTORY_TABLE* HistoryTable);
    extern(Windows) void RtlRaiseException(EXCEPTION_RECORD* ExceptionRecord);
}


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

// Interface to the native state for ldc.eh.common.eh_personality_common().
struct NativeContext
{
    EXCEPTION_RECORD* exceptionRecord;
    void* establisherFrame;
    CONTEXT* contextRecord;
    DISPATCHER_CONTEXT* dc;

    ubyte* getLanguageSpecificData() { return cast(ubyte*)dc.HandlerData; }
    ptrdiff_t getIP()                { return cast(ptrdiff_t)dc.ControlPc; }
    ptrdiff_t getRegionStart()       { return cast(ptrdiff_t)dc.ImageBase + dc.FunctionEntry.BeginAddress; }
    bool isSearchPhase()             { return (exceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND) != 0; }
    bool skipCatchComparison()       { return false; }
    ptrdiff_t getCfaAddress()        { return cast(ptrdiff_t)establisherFrame; }
    Object getThrownObject()         { return cast(Object)cast(void*)exceptionRecord.ExceptionInformation[0]; }

    void overrideThrownObject(Object newObject)
    {
        exceptionRecord.ExceptionInformation[0] = cast(ULONG_PTR)cast(void*)newObject;
    }

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
        if (exceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND)
        {
            contextRecord.Rdx = ti_offset; // Selector value for cleanup
        }
        else if (exceptionRecord.ExceptionFlags & EXCEPTION_UNWIND)
        {
            fatalerror("EXCEPTION_UNWIND and catch");
        }
        else
        {
            pushCleanupBlockRecord(getCfaAddress(), getThrownObject());

            RtlUnwindEx(establisherFrame, cast(PVOID)landingPadAddr, exceptionRecord,
                cast(PVOID)exceptionRecord.ExceptionInformation[0], contextRecord,
                dc.HistoryTable);
            fatalerror("RtlUnwindEx failed");
        }
        return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
    }

    EXCEPTION_DISPOSITION installFinallyContext(ptrdiff_t landingPadAddr)
    {
        if (exceptionRecord.ExceptionFlags & EXCEPTION_TARGET_UNWIND)
        {
            contextRecord.Rdx = 0; // Selector value for cleanup
        }
        else if (exceptionRecord.ExceptionFlags & EXCEPTION_UNWIND)
        {
            fatalerror("EXCEPTION_UNWIND and cleanup");
        }
        else
        {
            pushCleanupBlockRecord(getCfaAddress(), getThrownObject());

            RtlUnwindEx(establisherFrame, cast(PVOID)landingPadAddr, exceptionRecord,
                cast(PVOID)exceptionRecord.ExceptionInformation[0], contextRecord,
                dc.HistoryTable);
            fatalerror("RtlUnwindEx failed");
        }
        return EXCEPTION_DISPOSITION.ExceptionContinueSearch;
    }
}

// The personality routine gets called by the unwind handler and is responsible for
// reading the EH tables and deciding what to do.
extern(C) EXCEPTION_DISPOSITION _d_eh_personality(EXCEPTION_RECORD* ExceptionRecord,
    void* EstablisherFrame, CONTEXT* ContextRecord, DISPATCHER_CONTEXT* DispatcherContext)
{
    auto nativeContext = NativeContext(ExceptionRecord, EstablisherFrame,
        ContextRecord, DispatcherContext);
    return eh_personality_common(nativeContext);
}



public extern(C):

/// Called by our compiler-generated code to throw an exception.
void _d_throw_exception(Object e)
{
    if (e is null)
        fatalerror("Cannot throw null exception");
    if (e.classinfo is null)
        fatalerror("Cannot throw corrupt exception object with null classinfo");

    debug(EH_personality)
    {
        printf("= Throwing new exception of type %s: %p (classinfo at %p)\n",
            e.classinfo.name.ptr, e, e.classinfo);
    }

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

/// Called by our compiler-generate code to resume unwinding after a finally
/// block (or dtor destruction block) has been run.
void _d_eh_resume_unwind(Object e)
{
    debug(EH_personality)
    {
        printf("= Returning from cleanup block for %p\n", e);
    }

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

Object _d_eh_enter_catch(Object e)
{
    popCleanupBlockRecord();
    return e;
}

} // Win64
