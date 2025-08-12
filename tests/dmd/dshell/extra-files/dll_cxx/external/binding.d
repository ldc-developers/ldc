module binding;

version (LDC) // automatic dllimport for *extern(D)* globals only
{
    pragma(mangle, "testExternalImportVar")
    __gshared extern int testExternalImportVar;
}
else
{
    extern (C) __gshared extern int testExternalImportVar;
}
