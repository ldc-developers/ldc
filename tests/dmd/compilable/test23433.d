// https://issues.dlang.org/show_bug.cgi?id=23433
module object;

version (LDC) // more thorough checks
{
    class Object {}
    class TypeInfo_Class;
}

class Throwable { }
class Exception : Throwable { this(immutable(char)[]) { } }

// Needed to lower `new Exception("ice")` to it.
T _d_newclassT(T)()
if (is(T == class))
{
    return null;
}

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
