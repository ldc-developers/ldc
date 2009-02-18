extern(C) int printf(char*, ...);

ulong retval() {
    version (X86)
    asm { naked; mov EAX, 0xff; mov EDX, 0xaa; ret; }
    else version (X86_64)
    asm { naked; mov RAX, 0xaa000000ff; ret; }
}

ulong retval2() {
    return (cast(ulong)0xaa << 32) | 0xff;
}

void main() {
    auto a = retval();
    auto b = retval2();
    printf("%llu\n%llu\n", retval(), retval2());
    assert(a == 0xaa000000ff);
    assert(a == b);
}
