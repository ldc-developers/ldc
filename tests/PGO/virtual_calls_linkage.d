// Test instrumentation and optimization of virtual class method calls
// where the classes are spread across files.

// RUN: %ldc -I%S -O3 -c -output-ll -fprofile-instr-generate -fprofile-indirect-calls -fprofile-virtual-calls -of=%t.genonly.ll %s && FileCheck %s --check-prefix=PROFGEN < %t.genonly.ll

// Test separate compilation where one file is not instrumented
// RUN: %ldc -I%S -c -of=%t.input%obj %S/inputs/virtual_calls_input.d \
// RUN:   &&  %ldc -I%S -fprofile-instr-generate=%t.profraw  -fprofile-indirect-calls -fprofile-virtual-calls %t.input%obj -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -I%S -O3 -release -c -output-ll -of=%t.ll -fprofile-instr-use=%t.profdata -fprofile-indirect-calls -fprofile-virtual-calls %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t.ll \
// RUN:   &&  %ldc -I%S -O3 -release -of=%t%exe -fprofile-instr-use=%t.profdata -fprofile-indirect-calls -fprofile-virtual-calls %t.input%obj %s

// Test separate compilation where both files are instrumented
// RUN: %ldc -I%S -c -fprofile-instr-generate=%t2.profraw -fprofile-indirect-calls -fprofile-virtual-calls -of=%t.input_prof%obj %S/inputs/virtual_calls_input.d \
// RUN:   &&  %ldc -I%S -fprofile-instr-generate=%t2.profraw -fprofile-indirect-calls -fprofile-virtual-calls %t.input_prof%obj -run %s  \
// RUN:   &&  %profdata merge %t2.profraw -o %t2.profdata \
// RUN:   &&  %ldc -I%S -O3 -release -c -output-ll -of=%t2.ll -fprofile-instr-use=%t2.profdata -fprofile-indirect-calls -fprofile-virtual-calls %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t2.ll \
// RUN:   &&  %ldc -I%S -O3 -release -of=%t2%exe -fprofile-instr-use=%t2.profdata -fprofile-indirect-calls -fprofile-virtual-calls %t.input%obj %s

// Test at-once compilation. Should inline the optimized call!
// RUN: %ldc -I%S -singleobj -fprofile-instr-generate=%t3.profraw -fprofile-indirect-calls -fprofile-virtual-calls %S/inputs/virtual_calls_input.d -run %s  \
// RUN:   &&  %profdata merge %t3.profraw -o %t3.profdata \
// RUN:   &&  %ldc -I%S -singleobj -O3 -release -c -output-ll -of=%t3.ll -fprofile-instr-use=%t3.profdata -fprofile-indirect-calls -fprofile-virtual-calls %S/inputs/virtual_calls_input.d %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE_ATONCE < %t3.ll

// Test presence of __profd_ structures for pointer->hash translation by ldc-profdata
// PROFGEN: @__profd__D6inputs19virtual_calls_input1A6__vtblZ
// PROFGEN: @__profd__D3mod1B6__vtblZ
// PROFGEN: @__profd__D3mod1C6__vtblZ
// PROFGEN: @__profd__D6inputs19virtual_calls_input1D6__vtblZ

module mod;

import ldc.attributes : weak;
import inputs.virtual_calls_input;

class B : A
{
    override int getNum(int a)
    {
        return a * 7;
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
A select_ABC(int i)
{
    if (i < 600)
        return new A();
    else if (i < 1400)
        return new C();
    else
        return new B();
}

@weak // disable LLVM reasoning about this function
void consume_value(int i)
{
}

// PROFGEN-LABEL: @_D3mod12vtable_optimFZv(
// PROFUSE-LABEL: @_D3mod12vtable_optimFZv(
// PROFUSE_ATONCE-LABEL: @_D3mod12vtable_optimFZv(
void vtable_optim()
{
    for (int i; i < 2000; ++i)
    {
        A a = select_mostlyA(i); // 1600 As

        // PROFUSE:  [[REGCOND1:%[0-9]+]] = icmp eq %inputs.virtual_calls_input.A.__vtbl* [[REG1:%[0-9]+]], @_D6inputs19virtual_calls_input1A6__vtblZ
        // PROFUSE:  br i1 [[REGCOND1]], label %pgo.vtable.true, label %pgo.vtable.false, !prof ![[VTABbr:[0-9]+]]
        // PROFUSE:  pgo.vtable.true:
        // PROFUSE-NEXT:  call i32 @_D6inputs19virtual_calls_input1A6getNumMFiZi
        // PROFUSE:  pgo.vtable.false:
        // PROFUSE-NEXT:  [[TMP1:%.*]] = getelementptr {{.*}} [[REG1]]
        // PROFUSE-NEXT:  [[TMP2:%[0-9]+]] = load {{.*}} [[TMP1]]
        // PROFUSE-NEXT:  call i32 [[TMP2]](

        // PROFUSE_ATONCE:  [[REGCOND1:%[0-9]+]] = icmp eq %inputs.virtual_calls_input.A.__vtbl* [[REG1:%[0-9]+]], @_D6inputs19virtual_calls_input1A6__vtblZ
        // PROFUSE_ATONCE:  br i1 [[REGCOND1]], label %pgo.vtable.true, label %pgo.vtable.false, !prof ![[VTABbr:[0-9]+]]
        // PROFUSE_ATONCE:  pgo.vtable.true:
        // PROFUSE_ATONCE-NEXT:  shl
        // PROFUSE_ATONCE:  pgo.vtable.false:
        // PROFUSE_ATONCE-NEXT:  [[TMP1:%.*]] = getelementptr {{.*}} [[REG1]]
        // PROFUSE_ATONCE-NEXT:  [[TMP2:%[0-9]+]] = load {{.*}} [[TMP1]]
        // PROFUSE_ATONCE-NEXT:  call i32 [[TMP2]](

        int num = a.getNum(i);
        consume_value(num);

        // Test that external types are recognized even with separate unprofiled compilation
        // PROFUSE:  {{%[0-9]+}} = icmp eq %{{.*}} @_D6inputs19virtual_calls_input1D6__vtblZ
        // PROFUSE_ATONCE:  {{%[0-9]+}} = icmp eq %{{.*}} @_D6inputs19virtual_calls_input1D6__vtblZ
        D d = createD(i); // 2000 Ds
        d.doNothing();
    }
}

// PROFGEN-LABEL: @_Dmain(
// PROFUSE-LABEL: @_Dmain(
// PROFUSE_ATONCE-LABEL: @_Dmain(
int main()
{
    vtable_optim();
    return 0;
}

// PROFUSE-DAG: ![[VTABbr]] = !{!"branch_weights", i32 1601, i32 401}
// PROFUSE_ATONCE-DAG: ![[VTABbr]] = !{!"branch_weights", i32 1601, i32 401}
