/*
 * Temporary exception handling stubs
 */

import util.console;

private extern(C) void abort();

extern(C) void _d_throw_exception(Object e)
{
    console("Exception: ");
    if (e !is null)
    {
        console(e.toString())("\n");
    }
    abort();
}
