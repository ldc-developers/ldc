// RUN: %ldc -enable-dynamic-compile -run %s

import std.array;
import std.string;
import std.stdio;

import ldc.attributes;
import ldc.dynamic_compile;
import ldc.intrinsics;
import ldc.llvmasm;

version (ARM) version = ARM_Any;
version (AArch64) version = ARM_Any;

version (MIPS32) version = MIPS_Any;
version (MIPS64) version = MIPS_Any;

version (PPC) version = PPC_Any;
version (PPC64) version = PPC_Any;

version (RISCV32) version = RISCV_Any;
version (RISCV64) version = RISCV_Any;

@dynamicCompile void* foo()
{
    version (X86)
        return __asm!(void*)("movl %esp, $0", "=r");
    else version (X86_64)
        return __asm!(void*)("movq %rsp, $0", "=r");
    else version (ARM_Any)
        return __asm!(void*)("mov $0, sp", "=r");
    else version (PPC_Any)
        return __asm!(void*)("mr $0, 1", "=r");
    else version (MIPS_Any)
        return __asm!(void*)("move $0, $$sp", "=r");
    else
        return llvm_frameaddress(0); // soft-skip the test
}

void main(string[] args)
{
    auto dump = appender!(char[])();
    CompilerSettings settings;
    settings.optLevel = 3;
    settings.dumpHandler = (DumpStage stage, in char[] str)
    {
      if (DumpStage.FinalAsm == stage)
      {
        dump.put(str);
        write(str);
      }
    };
    compileDynamicCode(settings);
    void* result = foo();
    assert(result != null);
}
