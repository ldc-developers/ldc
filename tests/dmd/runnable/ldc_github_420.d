import core.simd : double2;
struct Foo {
    double2 x;
    this(uint) {
        x = [0.0, 0.0];
    }
}
void main() {
    Foo y = Foo();
}