/**
 * Contains ABI specific definitions for variable argument lists.
 *
 * Copyright: Authors 2016
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   Kai Nacke
 */
module ldc.internal.vararg;

version (AArch64)
{
    // AAPCS64 defines this parameter control block in section 7.1.4.
    // Handling of variable argument lists is described in appendix B.
    // http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf 
    extern (C++, std) struct __va_list
    {
        void* __stack;
        void* __gr_top;
        void* __vr_top;
        int __gr_offs;
        int __vr_offs;
    }
}
else version (ARM)
{
    // Need std::__va_list for C++ mangling compatability
    // section AAPCS 7.1.4
    extern (C++, std) struct __va_list
    {
        void *__ap;
    }
}
