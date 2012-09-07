module ldc.eh2;

private import core.stdc.stdlib; // abort
private import rt.util.console;

extern(C) void _d_throw_exception(Object e)
{
    if (e !is null)
    {
        // Raise exception
    }
    console("_d_throw_exception not yet implemented");
    abort();
}
