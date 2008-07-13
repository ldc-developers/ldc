module bug17;

struct Vec
{
    Vec opAdd(ref Vec b) { return Vec(); }
    Vec opMul(double a) { return Vec(); }
}
void main()
{
    Vec foo;
    Vec bar;
    auto whee=foo+bar*1f;
}
