// RUN: %ldc -c %s

alias va_list = void*;

pragma(LDC_va_arg) T va_arg(T)(va_list ap);

int foo(va_list ap) {
    return va_arg!(int)(ap);
}
