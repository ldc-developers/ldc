// RUN: %ldc -run %s

enum offset = 0xFFFF_FFFF_0000_0000UL;
void main() {
    assert((cast(ulong)&main) != (cast(ulong)&main + offset));
}
