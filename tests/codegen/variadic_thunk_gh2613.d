// RUN: %ldc -run %s

interface Stream
{
    void write(...);
}

class OutputStream : Stream
{
    void write(...)
    {
        import core.vararg;
        auto arg = va_arg!string(_argptr);
        assert(arg == "Hello world");
    }
}

void main()
{
    Stream stream = new OutputStream;
    stream.write("Hello world");
}
