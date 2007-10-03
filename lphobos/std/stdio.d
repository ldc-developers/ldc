module std.stdio;

void _writef(T)(T t) {
  //static if(is(T: Object)) _writef(t.toString()); else
  static if(is(T: char[])) printf("%.*s", t.length, t.ptr); else
  static if(is(T==int)) printf("%i", t); else
  static assert(false, "Cannot print "~T.stringof);
}

void writef(T...)(T t) {
  foreach (v; t) _writef(v);
}
void writefln(T...)(T t) {
  writef(t, "\n"[]);
}
