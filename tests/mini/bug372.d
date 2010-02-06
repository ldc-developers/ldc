class A {
    class B {
    }
}

class C : A {
    void test () {
        B foo = new B();
    }
}

int main () {
    return 0;
}
