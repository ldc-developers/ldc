// RUN: %ldc -run %s

shared class A {
}

shared class B : A {
}

void main() {
    shared A a1 = cast(shared A) A.classinfo.create();
    shared A a2 = cast(shared A) B.classinfo.create();
}
