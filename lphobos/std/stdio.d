module std.stdio;

import std.traits;

void _writef(T)(T t) {
    static if (is(T == char)) {
        printf("%c", t);
    }
    else static if (is(T : char[])) {
        printf("%.*s", t.length, t.ptr);
    }
    else static if (is(T : long)) {
        printf("%ld", t);
    }
    else static if (is(T : ulong)) {
        printf("%lu", t);
    }
    else static if (is(T : real)) {
        printf("%f", t);
    }
    else static if (is(T : Object)) {
        _writef(t.toString());
    }
    else static if(isArray!(T)) {
        _writef('[');
        if (t.length) {
            _writef(t[0]);
            foreach(v; t[1..$]) {
                _writef(','); _writef(v);
            }
        }
        _writef(']');
    }
    else static assert(0, "Cannot writef:"~T.tostring);
}

void writef(T...)(T t)
{
    foreach(v;t) _writef(v);
}
void writefln(T...)(T t)
{
    writef(t, '\n');
}
