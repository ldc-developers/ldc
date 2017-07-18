// Tests diagnostics of __traits(targetCPU) and __traits(targetHasFeature, ...)

// RUN: not %ldc -c -w %s 2>&1 | FileCheck %s

void main()
{
// CHECK: Warning: ignoring arguments for __traits targetCPU
    enum a = __traits(targetCPU, 1);

// CHECK: Error: __traits targetHasFeature expects one argument, not 0
    enum b = __traits(targetHasFeature);
// CHECK: Error: __traits targetHasFeature expects one argument, not 2
    enum c = __traits(targetHasFeature, "fma", 1);
// CHECK: Error: expression expected as argument of __traits targetHasFeature
    enum d = __traits(targetHasFeature, main);
// CHECK: Error: string expected as argument of __traits targetHasFeature instead of 1
    enum e = __traits(targetHasFeature, 1);

// CHECK: 夜畔' is not a recognized feature for this target (ignoring feature)
    enum f = __traits(targetHasFeature, "夜畔");
}

