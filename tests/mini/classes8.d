class A {
    int i;
    int l;
    this(bool b,bool b2=false) {
        if (b) this = new B;
        i = 4;
        if (b2) this = new C;
        l = 64;
    }
}
class B : A{
    this() {
        super(false);
    }
}
class C : A{
    this() {
        super(false);
    }
}
void main() {
}
