// REQUIRED_ARGS: -gs
// The "-gs" flag is just here to test it somewhere in the entire test suite.

public alias extern (C) void function(void*) Bar;

public interface Test
{
    public void foo(Bar bar)
    in
    {
        assert(bar);
    }
}

void main() {}
