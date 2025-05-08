// RUN: %ldc -mtriple=aarch64-none-elf -c %s
// RUN: %ldc -mtriple=riscv64-unknown-elf -c %s

alias va_list = void*;

pragma(LDC_va_arg) T va_arg(T)(va_list ap);

int foo(va_list ap) {
    return va_arg!(int)(ap);
}
