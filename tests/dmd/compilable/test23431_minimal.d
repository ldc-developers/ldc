// https://issues.dlang.org/show_bug.cgi?id=23431
// REQUIRED_ARGS: -lowmem
module object;

version (LDC) // more thorough checks
{
    class Object {}
    class TypeInfo_Class;
}

alias string  = immutable(char)[];
class Throwable { }
class Exception : Throwable
{
    this(string )
    {
    }
}

class Error { }

void test23431()
{
    int a;

    try
    {
        throw new Exception("test1");
        a++;
    }
    finally
    {
    }
}
