// https://issues.dlang.org/show_bug.cgi?id=23433
module object;

version (LDC) // more thorough checks
{
    class Object {}
    class TypeInfo_Class;
}

class Throwable { }
class Exception : Throwable { this(immutable(char)[]) { } }

void test23433()
{
    try
    {
        throw new Exception("ice");
    }
    finally
    {
    }
}
