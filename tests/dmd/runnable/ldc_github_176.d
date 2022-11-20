alias __vector(float[4]) float4;

void testSplatInit() {
    float4 a = 1, b = 2;
    assert((a + b).array == [3, 3, 3, 3]);
}

void testArrayInit() {
    float4 c = [1, 2, 3, 4];
    assert(c.array == [1, 2, 3, 4]);
}

void testArrayPtrAssign() {
    float4 c = 1, d = 0;
    d.array[0] = 1;
    d.ptr[1] = 2;
    assert((c + d).array == [2, 3, 1, 1]);
}

void main() {
    testSplatInit();
    testArrayInit();
    testArrayPtrAssign();
}
