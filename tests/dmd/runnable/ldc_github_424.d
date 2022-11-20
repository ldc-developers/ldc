import core.simd;
import std.stdio;

void main()
{
    // ldc: Error: expression 'cast(immutable(__vector(ubyte[16LU])))cast(ubyte)123u' is not a constant
    static immutable ubyte16 vec1 = 123;
    writeln(vec1.array);

    // ldc: infinite loop
    static immutable ubyte16 vec2 = [123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123];
    writeln(vec2.array);
}
