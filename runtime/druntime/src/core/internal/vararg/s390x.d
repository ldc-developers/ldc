module core.internal.vararg.s390x;

version (SystemZ)  : import core.stdc.stdarg : alignUp;

nothrow:

// Layout of this struct must match __gnuc_va_list for C ABI compatibility
struct __va_list_tag
{
    long __gpr = 0; // no regs
    long __fpr = 0; // no fp regs
    void* __overflow_arg_area;
    void* __reg_save_area;
}

alias __va_list = __va_list_tag;

/**
 * Making it an array of 1 causes va_list to be passed as a pointer in
 * function argument lists
 */
alias va_list = __va_list*;

/// Compile-time `va_arg` extraction for s390x
T va_arg(T)(va_list ap)
{
    static if (is(T U == __argTypes))
    {
        static if (U.length == 0 || U[0].sizeof > 8 || is(U[0] == __vector))
        {
            // Always passed in memory (varying vectors are passed in parameter area)
            auto p = *cast(T*) ap.__overflow_arg_area;
            ap.__overflow_arg_area = p + T.alignof.alignUp;
            return p;
        }
        else static if (U.length == 1)
        {
            // Arg is passed in one register
            alias T1 = U[0];
            static if (is(T1 == double) || is(T1 == float))
            {
                // Maybe passed in $fr registers
                if (ap.__fpr <= 4)
                {
                    // Passed in $fr registers (FPR region starts at +0x80)
                    auto p = cast(T*) ap.__reg_save_area + 128 + ap.__fpr * 8;
                    ap.__fpr++;
                    return *p;
                }
                else
                {
                    // overflow arguments
                    auto p = cast(T*) ap.__overflow_arg_area;
                    // no matter the actual size of the fp variable
                    // parameter slot is always 8-byte-wide (f32 is extended to f64)
                    ap.__overflow_arg_area += 8;
                    return *p;
                }
            }
            else
            {
                // Maybe passed in $r (GPR) registers
                if (ap.__gpr <= 5)
                {
                    // Passed in $gpr registers (GPR region starts at +0x10)
                    auto p = cast(T*) ap.__reg_save_area + 16 + ap.__gpr * 8;
                    ap.__gpr++;
                    return *p;
                }
                else
                {
                    // overflow arguments
                    auto p = cast(T*) ap.__overflow_arg_area;
                    // no matter the actual size of the gpr variable
                    // parameter slot is always 8-byte-wide (after ABI adjustments)
                    ap.__overflow_arg_area += 8;
                    return *p;
                }
            }
        }
        else
        {
            static assert(false);
        }
    }
    else
    {
        static assert(false, "not a valid argument type for va_arg");
    }
}

/// Runtime `va_arg` extraction for s390x
void va_arg()(va_list ap, TypeInfo ti, void* parmn)
{
    TypeInfo arg1, arg2;
    if (TypeInfo_Struct ti_struct = cast(TypeInfo_Struct) ti)
    {
        // handle single-float element struct
        const rtFields = ti_struct.offTi();
        if (rtFields && rtFields.length == 1)
        {
            TypeInfo field1TypeInfo = rtFields[0].ti;
            if (field1TypeInfo is typeid(float) || field1TypeInfo is typeid(double))
            {
                auto tsize = field1TypeInfo.tsize;
                auto toffset = rtFields[0].offset;
                parmn[0..tsize] = p[toffset..tsize];
                return;
            }
        }
    }

    if (!ti.argTypes(arg1, arg2))
    {
        TypeInfo_Vector v1 = arg1 ? cast(TypeInfo_Vector) arg1 : null;
        if (arg1 && (arg1.tsize <= 8 && !v1))
        {
            auto tsize = arg1.tsize;
            // Maybe passed in $r (GPR) registers
            if (ap.__gpr <= 5)
            {
                // Passed in $gpr registers (GPR region starts at +0x10)
                auto p = cast(T*) ap.__reg_save_area + 16 + ap.__gpr * 8;
                ap.__gpr++;
                parmn[0..tsize] = p[0..tsize];
            }
            else
            {
                // overflow arguments
                auto p = cast(T*) ap.__overflow_arg_area;
                // no matter the actual size of the gpr variable
                // parameter slot is always 8-byte-wide (after ABI adjustments)
                ap.__overflow_arg_area += 8;
                parmn[0..tsize] = p[0..tsize];
            }
        }
        else if (arg1 && (arg1 is typeid(float) || arg1 is typeid(double)))
        {
            // Maybe passed in $fr registers
            if (ap.__fpr <= 4)
            {
                // Passed in $fr registers (FPR region starts at +0x80)
                auto p = cast(T*) ap.__reg_save_area + 128 + ap.__fpr * 8;
                ap.__fpr++;
                parmn[0..tsize] = p[0..tsize];
            }
            else
            {
                // overflow arguments
                auto p = cast(T*) ap.__overflow_arg_area;
                // no matter the actual size of the fp variable
                // parameter slot is always 8-byte-wide (f32 is extended to f64)
                ap.__overflow_arg_area += 8;
                parmn[0..tsize] = p[0..tsize];
            }
        }
        else
        {
            assert(false, "unhandled va_arg type!");
        }
        assert(!arg2);
    }
    else
    {
        assert(false, "not a valid argument type for va_arg");
    }
}
