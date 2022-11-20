void main()
{
    import std.container;
    auto a = Array!int(0, 1, 2, 3, 4, 5, 6, 7, 8);
    a.linearRemove(a[4 .. 6]);
    import std.stdio : writeln;
    writeln(a);
    assert(a == Array!int(0, 1, 2, 3, 6, 7, 8));
}