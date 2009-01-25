interface I {}
class A : I {}
class B : A {}

void main () {
    A a = new A;
    I i = a;

    assert(!cast(B)a);
    assert(!cast(B)i);
}
