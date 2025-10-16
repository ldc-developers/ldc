// RUN: %ldc -c %s

struct S {
    ~this() {}
}

int foo(S[] ss...) { return 0; }

void bar(bool a) {
    const r = a ? foo(S()) : foo(S());
}
