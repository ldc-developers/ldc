extern(C) int printf(char*, ...);

ulong retval() {
    version (X86)
    asm { naked; mov EAX, 0xff; mov EDX, 0xaa; ret; }
    else version (X86_64)
    asm { naked; mov EAX, 0xff; mov EDX, 0xaa; ret; }
}

ulong retval2() {
    return (cast(ulong)0xaa << 32) | 0xff;
}

void main() {
    ulong a,b;
    a = retval();
    b = retval2();
    printf("%llu\n%llu\n", retval(), retval2());
    version (X86)
    {
    assert(a == 0x000000aa000000ff);
    assert(a == b);
    }
    else version (X86_64)
    {
    assert(a == 0xff);
    assert(b == 0x000000aa000000ff);
    }
}
