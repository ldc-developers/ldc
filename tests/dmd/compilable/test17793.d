// LDC: don't enforce -mcpu
// required_args: -mcpu=avx2
import core.simd;

static if (__traits(compiles, double4))
{
    double4 foo();
    void test(double[4]);

    void main()
    {
        test(foo().array);
    }
}
