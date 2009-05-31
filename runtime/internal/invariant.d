
/*
 * Placed into the Public Domain
 * written by Walter Bright
 * www.digitalmars.com
 */

extern(C) void _d_invariant(Object o)
{   ClassInfo c;

    //printf("__d_invariant(%p)\n", o);

    // BUG: needs to be filename/line of caller, not library routine
    assert(o !is null);	// just do null check, not invariant check

    c = o.classinfo;
    do
    {
	if (c.classInvariant)
	{
	    void delegate() inv;
	    inv.ptr = cast(void*) o;
	    inv.funcptr = c.classInvariant;
	    inv();
	}
	c = c.base;
    } while (c);
}
