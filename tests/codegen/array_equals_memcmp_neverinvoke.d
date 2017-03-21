// Tests that memcmp array comparisons `call` memcmp instead of `invoke`.

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix=LLVM < %t.ll

// When the user defines memcmp, it overrides the prototype defined by LDC.
// The user's prototype does not have the nounwind attribute, and a call to memcmp may become `invoke`.
extern(C) int memcmp(void*, void*, size_t);

void foo();

// Test that memcmp is not `invoked`
// LLVM-LABEL: define{{.*}} @{{.*}}never_invoke
void never_invoke(bool[2] a, bool[2] b)
{
    try
    {
        // LLVM: call i32 @memcmp({{.*}}, {{.*}}, i{{32|64}} 2)
        auto result = a == b;
        foo(); // Compiler has to assume that this may throw
    }
    catch (Exception e)
    {
    }
}
