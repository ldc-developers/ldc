// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

// CHECK-DAG: %gh2235.Foo = type <{
align(2) struct Foo {
    long y;
    byte z;
}

// CHECK-DAG: %gh2235.Bar = type <{
class Bar {
    union {
        bool b;
        Foo foo;
    }
    byte x;

    void set(Foo f) {
        x = 99;
        foo = f;
    }
}

void main() {
    Bar bar = new Bar();
    Foo f;
    bar.set(f);
    assert(bar.x == 99);
}
