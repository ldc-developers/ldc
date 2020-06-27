// Some ABIs don't pass empty static arrays or empty PODs as arguments.
// This can get hairy, e.g., if such a parameter is captured.

// RUN: %ldc -c -g %s

struct EmptyPOD {}

EmptyPOD emptyPOD(EmptyPOD e)
{
    EmptyPOD nested() { return e; }
    nested();
    return e;
}

int[0] emptySArray(int[0] a)
{
    int[0] nested() { return a; }
    nested();
    return a;
}

void tuple()
{
    auto test(T...)(T params)
    {
        auto nested() { return params[0]; }
        nested();
        return params[0];
    }

    test(EmptyPOD());

    int[0] a;
    test(a);
}
