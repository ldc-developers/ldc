module bug10;
import std.stdio;
class C
{
    char[] msg;

    this()
    {
    }
    this(char[] msg)
    {
        this.msg = msg;
    }
}

void main()
{
    auto c = new C();
    c.msg = "world";
    auto b = new C("hello");
    printf("%.*s\n", b.msg.length, b.msg.ptr);
    printf("%.*s\n", c.msg.length, c.msg.ptr);
}
