alias __vector(int[4]) int4;
alias __vector(long[2]) long2;

void testVectorCast() {
    // TODO: Depending on the exact semantics of vector types, which are not
    // clear from the D reference manual, this test might be endian-sensitive.
    int4 a = [1, 0, 2, 0];
    auto b = cast(long2) a;
    assert(b.array == [1, 2]);
}

void main() {
    testVectorCast();
}