/**
 * Libunwind-based implementation of `TraceInfo`
 *
 * This module exposes an handler that uses libunwind to print stack traces.
 * It is used when druntime is packaged with `DRuntime_Use_Libunwind` or when
 * the user uses the following in `main`:
 * ---
 * import core.runtime;
 * import core.internal.backtrace.handler;
 * Runtime.traceHandler = &libunwindDefaultTraceHandler;
 * ---
 *
 * Note that this module uses `dladdr` to retrieve the function's name.
 * To ensure that local (non-library) functions have their name printed,
 * the flag `-L--export-dynamic` must be used while compiling,
 * otherwise only the executable name will be available.
 *
 * Authors: Mathias 'Geod24' Lang
 * Copyright: D Language Foundation - 2020
 * See_Also: https://www.nongnu.org/libunwind/man/libunwind(3).html
 */
module core.internal.backtrace.handler;

version (DRuntime_Use_Libunwind):

import core.internal.backtrace.dwarf;
import core.internal.backtrace.libunwind;

/// Ditto
class LibunwindHandler : Throwable.TraceInfo
{
    private static struct FrameInfo
    {
        const(void)* address;
    }

    size_t numframes;
    enum MAXFRAMES = 128;
    FrameInfo[MAXFRAMES] callstack = void;

    enum MAXPROCNAMELENGTH = 500;
    char*[MAXFRAMES] namestack = void; // will be allocated using malloc, must be freed in destructor

    /**
     * Create a new instance of this trace handler saving the current context
     *
     * Params:
     *   frames_to_skip = The number of frames leading to this one.
     *                    Defaults to 1. Note that the opApply will not
     *                    show any frames that appear before _d_throwdwarf.
     */
    public this (size_t frames_to_skip = 1) nothrow @nogc
    {
        import core.stdc.stdlib : malloc;
        import core.stdc.string : memcpy;

        unw_context_t context;
        unw_cursor_t cursor;
        unw_getcontext(&context);
        unw_init_local(&cursor, &context);

        while (frames_to_skip > 0 && unw_step(&cursor) > 0)
            --frames_to_skip;

        unw_word_t ip;
        foreach (idx, ref frame; this.callstack)
        {
            if (unw_get_reg(&cursor, UNW_REG_IP, &ip) == 0)
                frame.address = cast(void*) ip;

            unw_word_t offset;
            char* buffer = cast(char*)malloc(MAXPROCNAMELENGTH);
            switch (unw_get_proc_name(&cursor, buffer, MAXPROCNAMELENGTH, &offset))
            {
                case UNW_ESUCCESS:
                    namestack[idx] = buffer;
                    break;
                case UNW_ENOMEM:
                    // Name is longer than MAXPROCNAMELENGTH, truncated name was put in buffer.
                    // TODO: realloc larger buffer and try again.
                    namestack[idx] = buffer;
                    break;
                default:
                    immutable error_string = "<ERROR: Unable to retrieve function name>";
                    memcpy(buffer, error_string.ptr, error_string.length+1); // include 0-terminator
                    namestack[idx] = buffer;
                    break;
            }

            this.numframes++;
            if (unw_step(&cursor) <= 0)
                break;
        }
    }

    ~this()
    {
        import core.stdc.stdlib : free;
        // Need to deallocate the procedure name strings allocated in constructor.
        foreach (ref ptr; this.namestack[0..numframes])
        {
            free(ptr);
        }
    }

    ///
    override int opApply (scope int delegate(ref const(char[])) dg) const
    {
        return this.opApply((ref size_t, ref const(char[]) buf) => dg(buf));
    }

    ///
    override int opApply (scope int delegate(ref size_t, ref const(char[])) dg) const
    {
        import core.stdc.string : strlen;

        return traceHandlerOpApplyImpl(numframes,
            i => callstack[i].address,
            i => namestack[i][0..strlen(namestack[i])],
            dg);
    }

    ///
    override string toString () const
    {
        string buf;
        foreach ( i, line; this )
            buf ~= i ? "\n" ~ line : line;
        return buf;
    }
}

/**
 * Convenience function for power users wishing to test this module
 * See `core.runtime.defaultTraceHandler` for full documentation.
 */
Throwable.TraceInfo defaultTraceHandler (void* ptr = null)
{
    // avoid recursive GC calls in finalizer, trace handlers should be made @nogc instead
    import core.memory : GC;
    if (GC.inFinalizer)
        return null;

    return new LibunwindHandler();
}
