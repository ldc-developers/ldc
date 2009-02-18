module asm9;

version(X86)            version = DoSome;
else version(X86_64)    version = DoSome;

T add(T, T t)(T a) {
    asm {
        add a, t;
    }
    return a;
}

void main() {
    version (DoSome) {
        assert(add!(ubyte, 20)(10) == 30);
        assert(add!(ushort, 20_000)(10_000) == 30_000);
        assert(add!(uint, 2_000_000)(1_000_000) == 3_000_000);
    }
    version(X86_64) {
        // 64-bit immediates aren't allowed on "ADD", nor are
        // unsigned 32-bit ones, so make the template parameter
        // fit in a 32-bit signed int.
        // These values were chosen so that the lower 32-bits overflow
        // and we can see the upper half of the 64-bit input increment.
        auto result = add!(long, 2_000_000_000)(21_000_000_000);
        assert(result == 23_000_000_000);
    }
}
