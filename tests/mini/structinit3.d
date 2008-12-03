struct S {
    int a; int b; int c; int d = 7;
}
void test(int i) {
    S s = { 1, i };   // q.a = 1, q.b = i, q.c = 0, q.d = 7
    assert(s.a == 1);
    assert(s.b == i);
    assert(s.c == 0); // line 8
    assert(s.d == 7);
}
void main() {
    test(42);
}
