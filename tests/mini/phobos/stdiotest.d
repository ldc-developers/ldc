module stdiotest;

import std.stdio;

T typed(T)(T x)
{
    return x;
}

void main()
{
    /*char[] str = "hello";
    writefln(str);

    writefln("hello world");*/

    char[] fmt = "%s";
    writefln(2.0f);

    /*{writefln(typed!(byte)(1));}
    {writefln(typed!(short)(2));}
    {writefln(typed!(int)(3));}
    {writefln(typed!(long)(-4));}
    {writefln(typed!(ulong)(5));}
    {writefln("%f", typed!(float)(6));}
    {writefln("%f", typed!(double)(7));}
    {writefln("%f", typed!(real)(8));}*/
}
