// RUN: %ldc -c %s

extern(C++):

__gshared const int dblreg = 1;

pragma(mangle, dblreg.mangleof)
extern __gshared const int bla;
