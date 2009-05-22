extern(C) int printf(char*, ...);

long foo(ref int p) {
    try { return 0; }
    finally {
        p++;
        throw new Object;
    }
}

void main() {
    int p = 0;
    try {
        foo(p);
        assert(0);
    } catch {
    }
    printf("Number of types scope(exit) was executed : %d\n", p);
    assert(p == 1);
}