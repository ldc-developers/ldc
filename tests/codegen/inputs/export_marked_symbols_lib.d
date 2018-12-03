extern(C++): // so that we can manually declare the symbols in another module too

export __gshared int exportedGlobal;
__gshared int normalGlobal;

export void exportedFoo() {}
void normalFoo() {}
