// RUN: %ldc -c %s

alias T = int;

T[] arrayAlloc(size_t size);

T[] Σ(size_t length)
{
    size_t elemSize = T.sizeof;
    size_t arraySize;

    if (length == 0 || elemSize == 0)
        return null;

    version (D_InlineAsm_X86)
    {
        asm pure nothrow @nogc
        {
            mov     EAX, elemSize       ;
            mul     EAX, length         ;
            mov     arraySize, EAX      ;
            jnc     Lcontinue           ;
        }
    }
    else version (D_InlineAsm_X86_64)
    {
        asm pure nothrow @nogc
        {
            mov     RAX, elemSize       ;
            mul     RAX, length         ;
            mov     arraySize, RAX      ;
            jnc     Lcontinue           ;
        }
    }
    else
    {
        import core.checkedint : mulu;

        bool overflow = false;
        arraySize = mulu(elemSize, length, overflow);
        if (!overflow)
            goto Lcontinue;
    }

Loverflow:
    assert(0);

Lcontinue:
    auto arr = arrayAlloc(arraySize);
    if (!arr.ptr)
        goto Loverflow;
    return arr;
}
