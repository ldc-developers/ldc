/**
 * The thread module provides support for thread creation and management.
 *
 * Copyright: Copyright Sean Kelly 2005 - 2012.
 * License: Distributed under the
 *      $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost Software License 1.0).
 *    (See accompanying file LICENSE)
 * Authors:   Sean Kelly, Walter Bright, Alex Rønne Petersen, Martin Nowak
 * Source:    $(DRUNTIMESRC core/thread/context.d)
 */

module core.thread.context;

// LDC: Unconditionally change ABI to support sanitizers
version (LDC) version = SupportSanitizers_ABI;

struct StackContext
{
    void* bstack, tstack;

    /// Slot for the EH implementation to keep some state for each stack
    /// (will be necessary for exception chaining, etc.). Opaque as far as
    /// we are concerned here.
    void* ehContext;
    StackContext* within;
    StackContext* next, prev;
    version (SupportSanitizers_ABI)
    {
        // Stores the thread-specific fake stack handler for this StackContext. This is used while GC scanning this StackContext's stack.
        void* asan_fakestack;
    }
}

struct Callable
{
    void opAssign(void function() fn) pure nothrow @nogc @safe
    {
        () @trusted { m_fn = fn; }();
        m_type = Call.FN;
    }
    void opAssign(void delegate() dg) pure nothrow @nogc @safe
    {
        () @trusted { m_dg = dg; }();
        m_type = Call.DG;
    }
    void opCall()
    {
        switch (m_type)
        {
            case Call.FN:
                m_fn();
                break;
            case Call.DG:
                m_dg();
                break;
            default:
                break;
        }
    }
private:
    enum Call
    {
        NO,
        FN,
        DG
    }
    Call m_type = Call.NO;
    union
    {
        void function() m_fn;
        void delegate() m_dg;
    }
}
