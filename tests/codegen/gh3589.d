// UNSUPPORTED: Windows || Darwin

// emit duplicate instantiated globals into two object files:
// RUN: %ldc -c %S/inputs/gh3589_a.d -I%S/inputs -of=%t_a.o
// RUN: %ldc -c %S/inputs/gh3589_b.d -I%S/inputs -of=%t_b.o

// link & run:
// RUN: %ldc -I%S/inputs %t_a.o %t_b.o -run %s

extern extern(C) __gshared {
    // magic linker symbols to refer to the start and end of test_section
    byte __start_test_section;
    byte __stop_test_section;
}

void main() {
    import gh3589_a, gh3589_b;
    assert(a_info == b_info);

    const sectionSize = cast(size_t) (&__stop_test_section - &__start_test_section);
    assert(sectionSize == 4);
}
