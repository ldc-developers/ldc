// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

void foo()
{
    auto impl(T)(lazy T field)
    {
        // Make sure `field` is a closure variable with delegate type (LL struct).
        // CHECK: %nest.impl = type { { i8*, i32 (i8*)* } }
        auto ff() { return field; }
        auto a = field;
        return ff() + a;
    }
    auto r = impl(123);
    assert(r == 246);
}

void main() { foo(); }
