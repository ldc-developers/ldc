module inputs.inlinables_asm;

import ldc.attributes;
import ldc.llvmasm;

extern (C): // simplify mangling for easier function name matching

pragma(inline, true) extern (C) @naked void naked_asm_func()
{
	return __asm("nop", "");
}
