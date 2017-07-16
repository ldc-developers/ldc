// RUN: %ldc -run %s

@safe:

int step;
int[] globalArray;

void reset(int initialStep)
{
    step = initialStep;
    globalArray = [ -1, -2, -3, -4 ];
}

int[] getBaseSlice()
{
    assert(step++ == 0);
    return globalArray;
}

ref int[] getBaseSliceRef()
{
    assert(step++ == 0);
    return globalArray;
}

int getLowerBound(size_t dollar)
{
    assert(step++ == 1);
    assert(dollar == 4);
    globalArray = null;
    return 1;
}

int getUpperBound(size_t dollar, size_t expectedDollar)
{
    assert(step++ == 2);
    assert(dollar == expectedDollar);
    globalArray = [ 1, 2, 3 ];
    return 3;
}

// https://github.com/ldc-developers/ldc/issues/1433
void main()
{
    reset(0);
    auto r = getBaseSlice()[getLowerBound($) .. getUpperBound($, 4)];
    assert(r == [ -2, -3 ]); // old buffer

    // LDC and GDC treat $ as lvalue and load <base>.length each time it is accessed
    // DMD apparently treats it as rvalue and loads it once at the beginning (=> wrong bounds check)
    version(DigitalMars)
        enum expectedDollar = 4;
    else
        enum expectedDollar = 0;

    reset(1);
    r = globalArray[getLowerBound($) .. getUpperBound($, expectedDollar)];
    assert(r == [ 2, 3 ]); // new buffer

    reset(0);
    r = getBaseSliceRef()[getLowerBound($) .. getUpperBound($, expectedDollar)];
    version(DigitalMars)
        assert(r == [ -2, -3 ]); // old buffer
    else
        assert(r == [ 2, 3 ]); // new buffer

    testBoundsCheck();
}

void testBoundsCheck() @trusted // @trusted needed for catching Errors, otherwise @safe
{
    import core.exception : RangeError;
    reset(1);
    try
    {
        auto r = globalArray[getLowerBound($) .. 2]; // null[1 .. 2]
        assert(0); // fails for DMD
    }
    catch (RangeError) {}
}
