// RUN: %ldc -run %s

void runTest(T)() {
    {
        T v = 1.0 + 1.0i;
        assert(v++ == 1.0 + 1.0i);
        assert(v-- == 2.0 + 1.0i);
        assert(v == 1.0 + 1.0i);
    }
    {
        T v = 1.0 + 1.0i;
        assert(++v == 2.0 + 1.0i);
        assert(--v == 1.0 + 1.0i);
        assert(v == 1.0 + 1.0i);
    }
}

void main () {
    runTest!cfloat();
    runTest!cdouble();
    runTest!creal();
}
