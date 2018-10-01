// Tests LDC-specific target __traits

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-apple-darwin -mcpu=nehalem -d-version=CPU_NEHALEM -c %s
// RUN: %ldc -mtriple=i686-pc-linux -mcpu=pentium -mattr=+fma -d-version=ATTR_FMA -c %s
// RUN: %ldc -mtriple=i686-pc-linux -mcpu=pentium -mattr=+fma,-sse -d-version=ATTR_FMA_MINUS_SSE -c %s
// RUN: %ldc -mtriple=x86_64-apple-darwin -mcpu=haswell -d-version=CPU_HASWELL -c %s

// Important: LLVM's default CPU selection already enables some features (like sse3)

void main()
{
    version (CPU_NEHALEM)
    {
        static assert(__traits(targetCPU) == "nehalem");
        static assert(__traits(targetHasFeature, "sse3"));
        static assert(__traits(targetHasFeature, "sse4.1"));
        static assert(!__traits(targetHasFeature, "avx"));
        static assert(!__traits(targetHasFeature, "sse4"));
        static assert(!__traits(targetHasFeature, "sse4a"));
        static assert(!__traits(targetHasFeature, "unrecognized feature"));
        version(D_AVX) static assert(false);
        version(D_AVX2) static assert(false);
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
        version(D_AVX) {} else static assert(false);
        version(D_AVX2) static assert(false);
    }
    version (ATTR_FMA_MINUS_SSE)
    {
        // All implied features must be enabled for targetHasFeature to return true
        static assert(!__traits(targetHasFeature, "fma"));
        version(D_AVX) static assert(false);
        version(D_AVX2) static assert(false);
    }
    version (CPU_HASWELL)
    {
        static assert(__traits(targetHasFeature, "sse"));
        static assert(__traits(targetHasFeature, "sse2"));
        static assert(__traits(targetHasFeature, "sse3"));
        static assert(__traits(targetHasFeature, "sse4.1"));
        static assert(__traits(targetHasFeature, "avx"));
        static assert(__traits(targetHasFeature, "avx2"));
        version(D_AVX) {} else static assert(false);
        version(D_AVX2) {} else static assert(false);
    }
}
