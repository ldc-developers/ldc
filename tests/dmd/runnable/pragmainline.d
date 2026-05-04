// REQUIRED_ARGS: -wi
// EXTRA_FILES: imports/pragmainline_a.d
/* TEST_OUTPUT:
---
---
*/


import imports.pragmainline_a;

auto anonclass()
{
    return new class {
        pragma(inline, true)
        final size_t foo()
        {
            return value();
        }
    };
}

auto testAlwaysInline()
{
    size_t var;

    foreach (d; Data("string"))
    {
        var = d.length();
    }

    assert(var == 6);

    var = anonclass().foo();

    assert(var == 10);

    auto nested = (size_t i) {
        return i - value();
    };

    var = nested(var);

    assert(var == 0);
}

// LDC: inlining an *indirect* call (of the function pointer returned by `bar()`)
//      requires enabled optimizations
version (LDC) version (D_Optimized) version = LDC_Optimized;

void main()
{
    immutable baz = () => 1;
    version (LDC_Optimized)
        assert(foo() == bar()());
    assert(foo() == baz());
    version (LDC_Optimized)
        assert(bar()() == baz());

    testAlwaysInline();

    bool caught = false;
    try
        throws(throws(1));
    catch (Exception e)
        caught = true;
    assert(caught);
}
