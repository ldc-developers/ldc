module inputs.inlinables_asm;

import ldc.attributes;

extern (C): // simplify mangling for easier function name matching

pragma(inline, true) extern (C) void naked_asm_func()
{
    asm pure nothrow @nogc
    {
        naked;
        nop;
    }
}
