module std.stdio;

import std.traits;

void _writef(T)(T t) {
  static if(is(T: Object)) _writef(t.toString()); else
  static if(is(T==char)) printf("%c", t); else
  static if(is(T: char[])) printf("%.*s", t.length, t.ptr); else
  static if(isArray!(T)) {
    _writef('[');
    if (t.length) _writef(t[0]);
    for (int i=1; i<t.lengthi; ++i) { _writef(','); _writef(t[i]); }
    _writef(']');
  } else
  static if(is(T==int)) printf("%i", t); else
  static assert(false, "Cannot print "~T.stringof);
}

void writef(T...)(T t) {
  foreach (v; t) _writef(v);
}
void writefln(T...)(T t) {
  writef(t, "\n"[]);
}
