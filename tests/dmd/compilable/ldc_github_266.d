template Tuple(Stuff ...) {
    alias Stuff Tuple;
}
struct S {
    int i;
    alias Tuple!i t;
    void a() {
        auto x = t;
    }
    void b() {
        auto x = t;
    }
}
