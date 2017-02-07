// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct Empty {}
struct OnlyZeroSizedFields { int[0] dummy; }
struct Container { Empty empty; OnlyZeroSizedFields onlyZeroSizedFields; }

// Make sure all of these structs are returned in registers (no sret).
// CHECK-NOT: %.sret_arg

Empty makeEmpty() { return Empty(); }
OnlyZeroSizedFields makeOnlyZeroSizedFields() { return OnlyZeroSizedFields(); }
Container makeContainer() { return Container(); }

void main()
{
    makeEmpty();
    makeOnlyZeroSizedFields();
    makeContainer();
}
