// Make sure that -i includes an imported custom module, but excludes
// druntime and Phobos modules.

// Make sure it links and check the -v output:
// RUN: %ldc -I%S -i %s -v | FileCheck %s

// CHECK-NOT: {{^code .*}}
// CHECK:     {{^code include_imports2}}
// CHECK-NOT: {{^code .*}}
// CHECK:     {{^code include_imports}}
// CHECK-NOT: {{^code .*}}

static import core.stdc.math;       // druntime
static import std.math;             // Phobos
import ldc.attributes : assumeUsed; // LDC-specific druntime
import inputs.include_imports2 : bar;

@assumeUsed
void test(string s, double x)
{
    const r1 = core.stdc.math.log(x);
    const r2 = std.math.log(x);
    bar();
}

void main()
{
    test("abc", 0.5);
}
