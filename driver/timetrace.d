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
//
// The time trace profile is output in the Chrome Trace Event Format, described
// here: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
//
//===----------------------------------------------------------------------===//

module driver.timetrace;

import dmd.errors;
import dmd.globals;
import dmd.root.array;
import dmd.root.file;
import dmd.root.outbuffer;
import dmd.root.string : toDString;

// Thread local profiler instance (multithread currently not supported because compiler is single-threaded)
TimeTraceProfiler* timeTraceProfiler = null;

// processName pointer is captured
extern(C++)
void initializeTimeTrace(uint timeGranularity, uint memoryGranularity, const(char)* processName)
{
    assert(timeTraceProfiler is null, "Double initialization of timeTraceProfiler");
    timeTraceProfiler = new TimeTraceProfiler(timeGranularity, memoryGranularity, processName);
}

extern(C++)
void deinitializeTimeTrace()
{
    if (timeTraceProfilerEnabled())
    {
        object.destroy(timeTraceProfiler);
        timeTraceProfiler = null;
    }
}

pragma(inline, true)
extern(C++)
bool timeTraceProfilerEnabled()
{
    version (LDC)
    {
        import ldc.intrinsics: llvm_expect;
        return llvm_expect(timeTraceProfiler !is null, false);
    }
    else
    {
        return timeTraceProfiler !is null;
    }
}

const(char)[] getTimeTraceProfileFilename(const(char)* filename_cstr)
{
    const(char)[] filename;
    if (filename_cstr)
    {
        filename = filename_cstr.toDString();
    }
    if (filename.length == 0)
    {
        if (global.params.objfiles[0])
        {
            filename = global.params.objfiles[0].toDString() ~ ".time-trace";
        }
        else
        {
            filename = "out.time-trace";
        }
    }
    return filename;
}

extern(C++)
void writeTimeTraceProfile(const(char)* filename_cstr)
{
    if (!timeTraceProfiler)
        return;

    const filename = getTimeTraceProfileFilename(filename_cstr);

    OutBuffer buf;
    timeTraceProfiler.writeToBuffer(&buf);
    if (filename == "-")
    {
        // Write to stdout
        import core.stdc.stdio : fwrite, stdout;
        size_t n = fwrite(buf[].ptr, 1, buf.length, stdout);
        if (n != buf.length)
        {
            error(Loc.initial, "Error writing --ftime-trace profile to stdout");
            fatal();
        }
    }
    else if (!File.write(filename, buf[]))
    {
        error(Loc.initial, "Error writing --ftime-trace profile: could not open '%*.s'", cast(int) filename.length, filename.ptr);
        fatal();
    }
}

// Pointers should not be stored, string copies must be made.
extern(C++)
void timeTraceProfilerBegin(const(char)* name_ptr, const(char)* detail_ptr, Loc loc)
{
    import dmd.root.rmem;
    import core.stdc.string : strdup;

    assert(timeTraceProfiler);

    // `loc` contains a pointer to a string, so we need to duplicate that string too.
    if (loc.filename)
        loc.filename = strdup(loc.filename);

    timeTraceProfiler.beginScope(mem.xstrdup(name_ptr).toDString(),
                                 mem.xstrdup(detail_ptr).toDString(), loc);
}

extern(C++)
void timeTraceProfilerEnd()
{
    assert(timeTraceProfiler);
    timeTraceProfiler.endScope();
}



struct TimeTraceProfiler
{
    uint timeGranularity;
    uint memoryGranularity;
    const(char)[] processName;
    const(char)[] pidtid_string = `"pid":101,"tid":101`;

    timer_t beginningOfTime;
    Array!CounterEvent counterEvents;
    Array!DurationEvent durationEvents;
    Array!DurationEvent durationStack;

    // timeBegin / time_scale = time in microseconds
    static if (is(typeof(&QueryPerformanceFrequency)))
        timer_t time_scale = 1_000;
    else
        enum time_scale = 1_000;

    struct CounterEvent
    {
        size_t memoryInUse;
        ulong allocatedMemory;
        size_t numberOfGCCollections;
        timer_t timepoint;
    }
    struct DurationEvent
    {
        const(char)[] name;
        const(char)[] details;
        Loc loc;
        timer_t timeBegin;
        timer_t timeDuration;
    }

    @disable this();
    @disable this(this);

    this(uint timeGranularity, uint memoryGranularity, const(char)* processName)
    {
        this.timeGranularity = timeGranularity;
        this.memoryGranularity = memoryGranularity;
        this.processName = processName.toDString();

        static if (is(typeof(&QueryPerformanceFrequency)))
        {
            timer_t freq;
            QueryPerformanceFrequency(&freq);
            time_scale = freq / 1_000_000;
        }

        timer_t time;
        QueryPerformanceCounter(&time);
        this.beginningOfTime = time / time_scale;
    }

    void beginScope(const(char)[] name, const(char)[] details, Loc loc)
    {
        timer_t time;
        QueryPerformanceCounter(&time);

        DurationEvent event;
        event.name = name;
        event.details = details;
        event.loc = loc;
        event.timeBegin = time / time_scale;
        durationStack.push(event);

        //counterEvents.push(generateCounterEvent(event.timeBegin));
    }

    void endScope()
    {
        timer_t timeEnd;
        QueryPerformanceCounter(&timeEnd);
        timeEnd /= time_scale;

        DurationEvent event = durationStack.pop();
        event.timeDuration = timeEnd - event.timeBegin;
        if (event.timeDuration >= timeGranularity)
        {
            // Event passes the logging threshold
            event.timeBegin -= beginningOfTime;
            durationEvents.push(event);
            counterEvents.push(generateCounterEvent(timeEnd-beginningOfTime));
        }
    }

    CounterEvent generateCounterEvent(timer_t timepoint)
    {
        static import dmd.root.rmem;
        CounterEvent counters;
        if (dmd.root.rmem.mem.isGCEnabled)
        {
            static if (__VERSION__ >= 2085)
            {
                import core.memory : GC;
                auto stats = GC.stats();
                auto profileStats = GC.profileStats();

                counters.allocatedMemory = stats.allocatedInCurrentThread;
                counters.memoryInUse = stats.usedSize;
                counters.numberOfGCCollections = profileStats.numCollections;
            }
        }
        else
        {
            counters.allocatedMemory = dmd.root.rmem.heaptotal;
            counters.memoryInUse = dmd.root.rmem.heaptotal - dmd.root.rmem.heapleft;
        }
        counters.timepoint = timepoint;
        return counters;
    }

    void writeToBuffer(OutBuffer* buf)
    {
        writePrologue(buf);
        writeEvents(buf);
        writeEpilogue(buf);
    }

    void writePrologue(OutBuffer* buf)
    {
        buf.write("{\n\"beginningOfTime\":");
        buf.print(beginningOfTime);
        buf.write(",\n\"traceEvents\": [\n");
    }

    void writeEpilogue(OutBuffer* buf)
    {
        buf.write("]\n}\n");
    }

    void writeEvents(OutBuffer* buf)
    {
        writeMetadataEvents(buf);
        writeCounterEvents(buf);
        writeDurationEvents(buf);
        // Remove the trailing comma (and newline!) to obtain valid JSON.
        if ((*buf)[buf.length()-2] == ',')
        {
            buf.setsize(buf.length()-2);
            buf.writeByte('\n');
        }
    }

    void writeMetadataEvents(OutBuffer* buf)
    {
        // {"ph":"M","ts":0,"args":{"name":"bin/ldc2"},"name":"thread_name","pid":0,"tid":0},

        buf.write(`{"ph":"M","ts":0,"args":{"name":"`);
        buf.write(processName);
        buf.write(`"},"name":"process_name",`);
        buf.write(pidtid_string);
        buf.write("},\n");
        buf.write(`{"ph":"M","ts":0,"args":{"name":"`);
        buf.write(processName);
        buf.write(`"},"cat":"","name":"thread_name",`);
        buf.write(pidtid_string);
        buf.write("},\n");
    }

    void writeCounterEvents(OutBuffer* buf)
    {
        // {"ph":"C","name":"ctr","ts":111,"args": {"Allocated_Memory_bytes":  0, "hello":  0}},

        foreach (event; counterEvents)
        {
            buf.write(`{"ph":"C","name":"ctr","ts":`);
            buf.print(event.timepoint);
            buf.write(`,"args": {"memoryInUse_bytes":`);
            buf.print(event.memoryInUse);
            buf.write(`,"allocatedMemory_bytes":`);
            buf.print(event.allocatedMemory);
            buf.write(`,"GC collections":`);
            buf.print(event.numberOfGCCollections);
            buf.write("},");
            buf.write(pidtid_string);
            buf.write("},\n");
        }
    }

    void writeDurationEvents(OutBuffer* buf)
    {
        // {"ph":"X","name": "Sema1: somename","ts":111,"dur":222,"loc":"filename.d:123","args": {"detail": "something", "loc":"filename.d:123"},"pid":0,"tid":0}

        void writeLocation(Loc loc)
        {
            if (loc.filename)
            {
                buf.writestring(loc.filename);
                if (loc.linnum)
                {
                    buf.writeByte(':');
                    buf.print(loc.linnum);
                }
            }
            else
            {
                buf.write(`<no file>`);
            }
        }

        foreach (event; durationEvents)
        {
            buf.write(`{"ph":"X","name": "`);
            writeEscapeJSONString(buf, event.name);
            buf.write(`","ts":`);
            buf.print(event.timeBegin);
            buf.write(`,"dur":`);
            buf.print(event.timeDuration);
            buf.write(`,"loc":"`);
            writeLocation(event.loc);
            buf.write(`","args":{"detail": "`);
            writeEscapeJSONString(buf, event.details);
            // Also output loc data in the "args" field so it shows in trace viewers that do not support the "loc" variable
            buf.write(`","loc":"`);
            writeLocation(event.loc);
            buf.write(`"},`);
            buf.write(pidtid_string);
            buf.write("},\n");
        }
    }
}


/// RAII helper class to call the begin and end functions of the time trace
/// profiler.  When the object is constructed, it begins the section; and when
/// it is destroyed, it stops it.
struct TimeTraceScope
{
    @disable this();
    @disable this(this);

    this(lazy string name, Loc loc = Loc())
    {
        if (timeTraceProfilerEnabled())
        {
            assert(timeTraceProfiler);
            // `loc` contains a pointer to a string, so we need to duplicate that too.
            import core.stdc.string : strdup;
            if (loc.filename)
                loc.filename = strdup(loc.filename);
            timeTraceProfiler.beginScope(name.dup, "", loc);
        }
    }
    this(lazy string name, lazy string detail, Loc loc = Loc())
    {
        if (timeTraceProfilerEnabled())
        {
            assert(timeTraceProfiler);
            // `loc` contains a pointer to a string, so we need to duplicate that too.
            import core.stdc.string : strdup;
            if (loc.filename)
                loc.filename = strdup(loc.filename);
            timeTraceProfiler.beginScope(name.dup, detail.dup, loc);
        }
    }

    ~this()
    {
        if (timeTraceProfilerEnabled())
            timeTraceProfilerEnd();
    }
}


private void writeEscapeJSONString(OutBuffer* buf, const(char[]) str)
{
    foreach (char c; str)
    {
        switch (c)
        {
        case '\n':
            buf.writestring("\\n");
            break;
        case '\r':
            buf.writestring("\\r");
            break;
        case '\t':
            buf.writestring("\\t");
            break;
        case '\"':
            buf.writestring("\\\"");
            break;
        case '\\':
            buf.writestring("\\\\");
            break;
        case '\b':
            buf.writestring("\\b");
            break;
        case '\f':
            buf.writestring("\\f");
            break;
        default:
            if (c < 0x20)
                buf.printf("\\u%04x", c);
            else
            {
                // Note that UTF-8 chars pass through here just fine
                buf.writeByte(c);
            }
            break;
        }
    }
}


/// Implementation of clock based on rt/trace.d:
/**
 * Contains support code for code profiling.
 *
 * Copyright: Copyright Digital Mars 1995 - 2017.
 * License: Distributed under the
 *      $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost Software License 1.0).
 *    (See accompanying file LICENSE)
 * Authors:   Walter Bright, Sean Kelly, the LDC team
 * Source: $(DRUNTIMESRC rt/_trace.d)
 */

alias long timer_t;

version (Windows)
{
    extern (Windows)
    {
        export int QueryPerformanceCounter(timer_t *);
        export int QueryPerformanceFrequency(timer_t *);
    }
}
else version (AArch64)
{
    // We cannot use ldc.intrinsics.llvm_readcyclecounter because that is not an accurate
    // time counter (it is a counter of CPU cycles, where here we want a time clock).
    // Also, priviledged execution rights are needed to enable correct counting with
    // ldc.intrinsics.llvm_readcyclecounter on AArch64.
    extern (D) void QueryPerformanceCounter(timer_t* ctr)
    {
        asm { "mrs %0, cntvct_el0" : "=r" (*ctr); }
    }
    extern (D) void QueryPerformanceFrequency(timer_t* freq)
    {
        asm { "mrs %0, cntfrq_el0" : "=r" (*freq); }
    }
}
else version (LDC)
{
    extern (D) void QueryPerformanceCounter(timer_t* ctr)
    {
        import ldc.intrinsics: llvm_readcyclecounter;
        *ctr = llvm_readcyclecounter();
    }
}
else
{
    extern (D) void QueryPerformanceCounter(timer_t* ctr)
    {
        version (D_InlineAsm_X86)
        {
            asm
            {
                naked                   ;
                mov       ECX,EAX       ;
                rdtsc                   ;
                mov   [ECX],EAX         ;
                mov   4[ECX],EDX        ;
                ret                     ;
            }
        }
        else version (D_InlineAsm_X86_64)
        {
            asm
            {
                naked                   ;
                // rdtsc can produce skewed results without preceding lfence/mfence.
                // this is what GNU/Linux does, but only use mfence here.
                // see https://github.com/torvalds/linux/blob/03b9730b769fc4d87e40f6104f4c5b2e43889f19/arch/x86/include/asm/msr.h#L130-L154
                mfence                  ; // serialize rdtsc instruction.
                rdtsc                   ;
                mov   [RDI],EAX         ;
                mov   4[RDI],EDX        ;
                ret                     ;
            }
        }
        else
        {
            static assert(0);
        }
    }
}
