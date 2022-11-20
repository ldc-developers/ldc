import std.stdio;

void foo(bool[] err = null)
{
    if (err !is null)
    {
        if (err[0])
        {
            writeln(err);
        }
        else
        {
            writeln("Nothing to do.");
        }
    }
    else
    {
        writeln("Null input.");
    }
}

void main()
{
    foo();
    bool[] err = [false, false, false];
    foo(err);
    err[0] = true;
    foo(err);
}

