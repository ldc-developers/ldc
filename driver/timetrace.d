//===-- driver/timetrace.d ----------------------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Compilation time tracing, --ftime-trace.
// Supported from LLVM 10.
//
//===----------------------------------------------------------------------===//

module driver.timetrace;

import driver.ldc_version;

static if (LLVM_VERSION_MAJOR >= 10)
{
    // Forward declarations of LLVM Support functions
    extern(C++, llvm)
    {
        struct TimeTraceProfiler;
        void timeTraceProfilerEnd();

        static if (LLVM_VERSION_MAJOR < 11)
        {
            extern __gshared TimeTraceProfiler* TimeTraceProfilerInstance;
            auto getTimeTraceProfilerInstance() { return TimeTraceProfilerInstance; }
        }
        else
        {
            TimeTraceProfiler* getTimeTraceProfilerInstance();
        }
    }

    // Forward declaration of LDC D-->C++ support function
    extern(C++) void timeTraceProfilerBegin(size_t name_length, const(char)* name_ptr,
                                            size_t detail_length, const(char)* detail_ptr);

    pragma(inline, true)
    bool timeTraceProfilerEnabled() {
        version (LDC) {
            import ldc.intrinsics: llvm_expect;
            return llvm_expect(getTimeTraceProfilerInstance() !is null, false);
        } else {
            return getTimeTraceProfilerInstance() !is null;
        }
    }

    /// RAII helper class to call the begin and end functions of the time trace
    /// profiler.  When the object is constructed, it begins the section; and when
    /// it is destroyed, it stops it.
    struct TimeTraceScope
    {
        @disable this();
        @disable this(this);

        this(string name) {
            if (timeTraceProfilerEnabled())
                timeTraceProfilerBegin(name.length, name.ptr, 0, null);
        }
        this(string name, lazy string detail) {
            if (timeTraceProfilerEnabled())
                timeTraceProfilerBegin(name.length, name.ptr, detail.length, detail.ptr);
        }

        ~this() {
            if (timeTraceProfilerEnabled())
                timeTraceProfilerEnd();
        }
    }
}
else
{
    bool timeTraceProfilerEnabled() { return false; }
    struct TimeTraceScope
    {
        @disable this();
        @disable this(this);

        this(string name) { }
        this(string name, lazy string detail) { }

        ~this() { }
    }
}
