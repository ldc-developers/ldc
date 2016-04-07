// Test instrumentation and optimization of virtual class method calls
// Note the interplay between -fprofile-indirect-calls and -fprofile-virtual-calls

// RUN: %ldc -c -output-ll -fprofile-instr-generate -fprofile-indirect-calls -fprofile-virtual-calls -of=%t.ll %s && FileCheck %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw  -fprofile-indirect-calls -fprofile-virtual-calls -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -O3 -release -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata -fprofile-indirect-calls -fprofile-virtual-calls %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t2.ll

// PROFGEN: llvm.used {{.*}} @__profd__D3mod1A6__vtblZ

module mod;

import ldc.attributes : weak;

class A
{
    int getNum(int a)
    {
        return a * 2;
    }
}

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
    else if (i < 1900)
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
void vtable_optim()
{
    for (int i; i < 2000; ++i)
    {
        A a = select_mostlyA(i); // 1600 As

        // PROFUSE:  [[REGCOND1:%[0-9]+]] = icmp eq %mod.A.__vtbl* [[REG1:%[0-9]+]], @_D3mod1A6__vtblZ
        // PROFUSE:  br i1 [[REGCOND1]], label %pgo.vtable.true, label %pgo.vtable.false, !prof ![[VTABbr:[0-9]+]]
        // PROFUSE:  pgo.vtable.true:
        // PROFUSE-NEXT:  shl
        // PROFUSE:  pgo.vtable.false:
        // PROFUSE-NEXT:  [[TMP1:%.*]] = getelementptr {{.*}} [[REG1]]
        // PROFUSE-NEXT:  [[TMP2:%[0-9]+]] = load {{.*}} [[TMP1]]
        // PROFUSE-NEXT:  call i32 [[TMP2]](

        int num = a.getNum(i);
        consume_value(num);
    }
}

// PROFGEN-LABEL: @_D3mod14indirect_optimFZv(
// PROFUSE-LABEL: @_D3mod14indirect_optimFZv(
void indirect_optim()
{
    for (int i; i < 2000; ++i)
    {
        A a = select_ABC(i); // A+C = 1400

        // No VTable check, instead the "normal" indirect call optimization
        // PROFUSE:  [[REGCOND2:%[0-9]+]] = icmp eq i32 ({{.*}})* [[REG2:%a.getNum]], @_D3mod1A6getNumMFiZi
        // PROFUSE:  br i1 [[REGCOND2]], label %if.true, label %if.false, !prof ![[INDIRbr:[0-9]+]]
        // PROFUSE:  if.true
        // PROFUSE-NEXT:  shl
        // PROFUSE:  if.false
        // PROFUSE-NEXT:  call i32 [[REG2]](

        int num = a.getNum(i);
        consume_value(num);
    }
}

// PROFGEN-LABEL: @_Dmain(
// PROFUSE-LABEL: @_Dmain(
int main()
{
    vtable_optim();
    indirect_optim();
    return 0;
}

// PROFUSE-DAG: ![[VTABbr]] = !{!"branch_weights", i32 1601, i32 401}
// PROFUSE-DAG: ![[INDIRbr]] = !{!"branch_weights", i32 1900, i32 100}


version(none)
{
// Attempt to generate the same machine code without profiling
pragma(LDC_profile_instr, false)
{

void* getVTableSymbol(A)() pure {
    version (X86_64)
    {
        import ldc.llvmasm;
        return __asm!(void *)("leaq __D" ~ A.mangleof[1..$] ~ "6__vtblZ(%rip), $0", "=r");
    }
    else
    {
        // cast(void*)typeid(A).vtbl.ptr is suboptimal because TypeInfo is not constant
        return cast(void*)typeid(A).vtbl.ptr;
    }
}

void vtable_optim_manual()
{
    for (int i; i < 2000; ++i)
    {
        A a = select_mostlyA(i); // 1600 As

        int num;
        if (a.__vptr == getVTableSymbol!A)
        {
            num = a.A.getNum(i);
        }
        else
        {
            num = a.getNum(i);
        }

        consume_value(num);
    }
}
}
}
