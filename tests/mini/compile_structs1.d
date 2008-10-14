struct Foo { int a, b, c; union Data { } Data data; }
struct Bar { int a, b; }
struct Baz { int a; union { Foo foo; Bar bar; } }
void test() { Baz baz; if (baz.bar.a) return; }
