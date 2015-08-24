/**
 * This module implements the runtime-part of LDC exceptions
 * on platforms with libunwind support.
 */
module ldc.eh.libunwind;

version (Win64) {} else
{

// debug = EH_personality;
// debug = EH_personality_verbose;

import ldc.eh.common;

private:

// C headers
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


// Exception struct used by the runtime.
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

// Interface to the native state for ldc.eh.common.eh_personality_common().
struct NativeContext
{
    _Unwind_Action actions;
    _d_exception* exception_struct;
    _Unwind_Context_Ptr context;

    ubyte* getLanguageSpecificData() { return cast(ubyte*)_Unwind_GetLanguageSpecificData(context); }
    ptrdiff_t getIP()                { return _Unwind_GetIP(context); }
    ptrdiff_t getRegionStart()       { return _Unwind_GetRegionStart(context); }
    bool isSearchPhase()             { return (actions & _Unwind_Action.SEARCH_PHASE) != 0; }
    // Optimization: After the search phase, libunwind lets us know whether
    // we have found a handler in this frame the first time around. We can
    // thus skip further the comparisons if the HANDLER_FRAME flag is not
    // set.
    //
    // As a further optimization step, we could look into caching that
    // result inside _d_exception.
    bool skipCatchComparison()       { return !isSearchPhase() && (actions & _Unwind_Action.HANDLER_FRAME) == 0; }
    ptrdiff_t getCfaAddress()        { return _Unwind_GetCFA(context); }
    Object getThrownObject()         { return exception_struct.exception_object; }

    void overrideThrownObject(Object newObject)
    {
        exception_struct.exception_object = newObject;
    }

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

    // The personality routine gets called by the unwind handler and is responsible for
    // reading the EH tables and deciding what to do.
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

        auto nativeContext = NativeContext(actions, ucb.toDException(), context);
        _Unwind_Reason_Code rc = eh_personality_common(nativeContext);
        if (rc == _Unwind_Reason_Code.CONTINUE_UNWIND)
            return continueUnwind(ucb, context);
        return rc;
    }
}
else // !ARM
{
    // The personality routine gets called by the unwind handler and is responsible for
    // reading the EH tables and deciding what to do.
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
        auto nativeContext = NativeContext(actions, exception_struct, context);
        return eh_personality_common(nativeContext);
    }
}

extern(C) Throwable.TraceInfo _d_traceContext(void* ptr = null);



public extern(C):

/// Called by our compiler-generated code to throw an exception.
void _d_throw_exception(Object e)
{
    if (e is null)
        fatalerror("Cannot throw null exception");

    if (e.classinfo is null)
        fatalerror("Cannot throw corrupt exception object with null classinfo");

    auto throwable = cast(Throwable) e;

    if (throwable.info is null && cast(byte*)throwable !is typeid(throwable).init.ptr)
        throwable.info = _d_traceContext();

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

} // !Win64
