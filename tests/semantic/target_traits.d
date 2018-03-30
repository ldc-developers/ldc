// Tests LDC-specific target __traits

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-apple-darwin -mcpu=haswell -d-version=CPU_HASWELL -c %s
// RUN: %ldc -mtriple=x86_64-apple-darwin -mcpu=pentium -mattr=+fma -d-version=ATTR_FMA -c %s
// RUN: %ldc -mtriple=x86_64-apple-darwin -mcpu=pentium -mattr=+fma,-sse -d-version=ATTR_FMA_MINUS_SSE -c %s

// Important: LLVM's default CPU selection already enables some features (like sse3)

void main()
{
    version (CPU_HASWELL)
    {
        static assert(__traits(targetCPU) == "haswell");
        static assert(__traits(targetHasFeature, "sse3"));
        static assert(__traits(targetHasFeature, "sse4.1"));
        static assert(!__traits(targetHasFeature, "sse4"));
        static assert(!__traits(targetHasFeature, "sse4a"));
        static assert(!__traits(targetHasFeature, "unrecognized feature"));
    }
    version (ATTR_FMA)
    {
        static assert(__traits(targetHasFeature, "sse"));
        static assert(__traits(targetHasFeature, "sse2"));
        static assert(__traits(targetHasFeature, "sse3"));
        static assert(__traits(targetHasFeature, "sse4.1"));
        static assert(__traits(targetHasFeature, "fma"));
        static assert(__traits(targetHasFeature, "avx"));
        static assert(!__traits(targetHasFeature, "avx2"));
        static assert(!__traits(targetHasFeature, "unrecognized feature"));
    }
    version (ATTR_FMA_MINUS_SSE)
    {
        // All implied features must be enabled for targetHasFeature to return true
        static assert(!__traits(targetHasFeature, "fma"));
    }
}
