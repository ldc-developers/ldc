// Test VCP of invoke function calls

// RUN: %ldc -c -output-ll -fprofile-instr-generate -fprofile-indirect-calls -fprofile-virtual-calls -of=%t.ll %s && FileCheck %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw  -fprofile-indirect-calls -fprofile-virtual-calls -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -O3 -release -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata -fprofile-indirect-calls -fprofile-virtual-calls %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t2.ll

module mod;

import ldc.attributes : weak;

class A
{
    @weak // disable inlining
    int doNothing(int i)
    {
        return i;
    }
}

class B : A
{
    override int doNothing(int a)
    {
        return a;
    }
}

class C : A
{
}

@weak // disable LLVM reasoning about this function
A select_mostlyA(int i)
{
    if (i < 1600)
        return new A();
    else if (i < 1800)
        return new B();
    else
        return new C();
}

@weak // disable LLVM reasoning about this function
void consume_value(int i)
{
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
        // PROFUSE-NEXT:  [[VCPtrue:%[0-9]+]] = invoke i32 @_D3mod1A9doNothingMFiZi(
        // PROFUSE:  pgo.vtable.false:
        // PROFUSE-NEXT:  [[TMP1:%.*]] = getelementptr {{.*}} [[REG1]]
        // PROFUSE-NEXT:  [[TMP2:%[0-9]+]] = load {{.*}} [[TMP1]]
        // PROFUSE-NEXT:  [[VCPfalse:%[0-9]+]] = invoke i32 [[TMP2]](

        int num;
        try
        {
            num = a.doNothing(i);
        }
        finally
        {
            consume_value(num);
        }
    }
}

// PROFGEN-LABEL: @_Dmain(
// PROFUSE-LABEL: @_Dmain(
int main()
{
    vtable_optim();
    return 0;
}

// PROFUSE-DAG: ![[VTABbr]] = !{!"branch_weights", i32 1601, i32 401}
