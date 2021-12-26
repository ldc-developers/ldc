// AddressSanitizer: Test that the GC properly scans ASan's fakestack when enabled. Requires runtime support.

// REQUIRES: ASan, RTSupportsSanitizers

// Test without and with fake stack enabled.

// Note on debug lineinfo: on macOS the executable contains a link back to the
// object files for debug info. Therefore the order of text execution is important,
// i.e. we should finish all testing on one compiled executable before recompiling
// with different conditional compilation settings (because it will overwrite the
// object files from the previous compilation).

// RUN: %ldc %s -of=%t1%exe && %t1%exe -O
// RUN: %ldc -g -fsanitize=address %s -of=%t_asan%exe -O
// RUN: %t_asan%exe
// RUN: env %env_asan_opts=detect_stack_use_after_return=true %t_asan%exe 2>&1 | FileCheck %s

import core.memory;
import core.thread;
import std.stdio;

import ldc.attributes;
@weak
pragma(inline, false)
void doNotOptimizeParameter(T)(ref T)
{
}

void test_nulling_triggers_collection()
{
    void*[100] a; // Large enough so it will be on fakestack heap (not inlined in local stack frame)
    GC.collect();
    auto initialSize = GC.stats.usedSize;
    //writeln(__LINE__, " ", GC.stats.usedSize);
    a[50] = GC.malloc(100);
    //writeln(__LINE__, " ", GC.stats.usedSize);
    auto allocatedSize = GC.stats.usedSize - initialSize;
    assert(allocatedSize >= 100);
    a[50] = null;
    doNotOptimizeParameter(a);
    //writeln(__LINE__, " ", GC.stats.usedSize);
    GC.collect();
    //writeln(__LINE__, " ", GC.stats.usedSize);
    assert(GC.stats.usedSize == initialSize);
    doNotOptimizeParameter(a);
}

void test_non_null_does_not_trigger_collection(uint index)
{
    void*[100] a; // Large enough so it will be on fakestack heap (not inlined in local stack frame)
    //writefln("Address range a, line %u: 0x%x - 0x%x", __LINE__, cast(size_t)&a[0], cast(size_t)&a[$-1]);

    GC.collect();
    auto initialSize = GC.stats.usedSize;
    //writeln("GC.stats.usedSize, line ", __LINE__, ": ", GC.stats.usedSize);
    a[index] = GC.malloc(100);
    //writeln("GC.stats.usedSize, line ", __LINE__, ": ", GC.stats.usedSize);
    //writefln("Address a[index], line %u: 0x%x", __LINE__, cast(size_t) a[index]);
    auto allocatedSize = GC.stats.usedSize - initialSize;
    assert(allocatedSize >= 100);
    doNotOptimizeParameter(a);
    //writeln("GC.stats.usedSize, line ", __LINE__, ": ", GC.stats.usedSize);
    GC.collect();
    writeln("GC.stats.usedSize, line ", __LINE__, ": ", GC.stats.usedSize);
    assert(GC.stats.usedSize == initialSize + allocatedSize); // This fails if GC scanning does not support ASan's fakestack.
    GC.collect();
    doNotOptimizeParameter(a);
}

void main()
{
    test_nulling_triggers_collection();
    test_non_null_does_not_trigger_collection(0);
    test_non_null_does_not_trigger_collection(50);
    test_non_null_does_not_trigger_collection(99);

    // Also test threading
    auto t = new Thread({
                            test_non_null_does_not_trigger_collection(0);
                            test_non_null_does_not_trigger_collection(50);
                            test_non_null_does_not_trigger_collection(99);
                        });
    t.start();
    t.join();
}

// Make sure that the function calls did indeed happen.
// CHECK: GC.stats.usedSize
// CHECK: GC.stats.usedSize
// CHECK: GC.stats.usedSize
// CHECK: GC.stats.usedSize
// CHECK: GC.stats.usedSize
// CHECK: GC.stats.usedSize
