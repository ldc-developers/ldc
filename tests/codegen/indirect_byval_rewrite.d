// Ensures efficient indirect by-value passing via IndirectByvalRewrite.

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-windows-msvc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct Big { size_t a, b; }
struct WithPostblit { Big b; this(this) {} }

Big makeBig() { return Big(123, 456); }

void foo(Big, WithPostblit, Big, WithPostblit);

// CHECK: define {{.*}}_D22indirect_byval_rewrite3bar
void bar()
{
    // CHECK:      %bigLValue = alloca %indirect_byval_rewrite.Big
    Big bigLValue;
    // CHECK-NEXT: %withPostblitLValue = alloca %indirect_byval_rewrite.WithPostblit
    WithPostblit withPostblitLValue;

    // * 1st arg: bigLValue bitcopy, copied by IndirectByvalRewrite
    // CHECK-NEXT: %.hidden_copy_for_IndirectByvalRewrite = alloca %indirect_byval_rewrite.Big
    // * 2nd arg: temporary withPostblitLValue copy (copied by frontend, incl. postblit)
    // CHECK-NEXT: %__copytmp{{[0-9]*}} = alloca %indirect_byval_rewrite.WithPostblit
    // * 3rd arg: sret temporary filled by makeBig()
    // CHECK-NEXT: %.sret_tmp = alloca %indirect_byval_rewrite.Big
    // * 4th arg: WithPostblit() literal
    // CHECK-NEXT: %.structliteral = alloca %indirect_byval_rewrite.WithPostblit
    foo(bigLValue, withPostblitLValue, makeBig(), WithPostblit());

    // no more allocas!
    // CHECK-NOT: alloca

    // CHECK: ret void
}
