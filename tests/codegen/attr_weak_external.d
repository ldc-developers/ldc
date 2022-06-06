// Test that an imported @weak function does not result in an extern_weak reference.

// RUN: %ldc -c -I%S -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import inputs.attr_weak_external_input: weak_definition_seven;

void foo()
{
    auto a = &weak_definition_seven;
}

// CHECK-NOT: extern_weak
