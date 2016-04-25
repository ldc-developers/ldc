// Test VCP of functions returning void (no phi node)

// RUN: %ldc -c -output-ll -fprofile-instr-generate -fprofile-virtual-calls -of=%t.ll %s && FileCheck %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw  -fprofile-virtual-calls -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -O3 -release -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata -fprofile-virtual-calls %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t2.ll

module mod;

import ldc.attributes : weak;

class A
{
    @weak // disable inlining
    void doNothing(int i)
    {
    }
}

@weak // disable LLVM reasoning about this function
A select_mostlyA(int i)
{
    return new A();
}

// PROFGEN-LABEL: @_D3mod12vtable_optimFZv(
// PROFUSE-LABEL: @_D3mod12vtable_optimFZv(
void vtable_optim()
{
    for (int i; i < 2000; ++i)
    {
        A a = select_mostlyA(i); // 1600 As


        // PROFUSE:  [[REGCOND1:%[0-9]+]] = icmp eq %mod.A.__vtbl* [[REG1:%[0-9]+]], @_D3mod1A6__vtblZ
        // PROFUSE:  br i1 [[REGCOND1]], label %pgo.vtable.true, label %pgo.vtable.false, !prof ![[VTABbr:[0-9]+]]
        // PROFUSE:  pgo.vtable.true:
        // PROFUSE-NEXT:  call void @_D3mod1A9doNothingMFiZv
        // PROFUSE:  pgo.vtable.false:
        // PROFUSE-NEXT:  [[TMP1:%.*]] = getelementptr {{.*}} [[REG1]]
        // PROFUSE-NEXT:  [[TMP2:%[0-9]+]] = load {{.*}} [[TMP1]]
        // PROFUSE-NEXT:  call void [[TMP2]](

        a.doNothing(i);
    }
}

// PROFGEN-LABEL: @_Dmain(
// PROFUSE-LABEL: @_Dmain(
int main()
{
    vtable_optim();
    return 0;
}

// PROFUSE-DAG: ![[VTABbr]] = !{!"branch_weights", i32 2001, i32 1}
