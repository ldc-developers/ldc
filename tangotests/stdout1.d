import tango.io.Stdout;

void main()
{
    Stdout("Hello World").newline;
    Stdout.formatln("{} {}", "Hello", "World");
}
