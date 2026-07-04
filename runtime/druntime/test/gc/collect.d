module core.thread.collect; // get access to package core.thread

import core.thread.threadbase;
import core.memory;
import core.cpuid;

version (LDC) import ldc.attributes : optStrategy;

@optStrategy("none") // LDC: do not optimize away GC allocations
void main()
{
    auto threads = threadsPerCPU();
    if (threads > 1) // no parallel scanning on single-core CPU
    {
        assert(ll_nThreads == 0);
        auto p = new int[10]; // ensure, the proto-gc gets replaced by the conservative GC
        GC.collect();
        assert(ll_nThreads > 0);
    }
}
