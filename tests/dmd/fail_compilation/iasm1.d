// DISABLED: LDC_not_x86

// REQUIRED_ARGS: -m64
/*
// LDC: compiles
test_output:
---
fail_compilation/iasm1.d(103): Error: bad type/size of operands `and`
fail_compilation/iasm1.d(104): Error: bad type/size of operands `and`
---
*/

// https://issues.dlang.org/show_bug.cgi?id=15999

#line 100

void test100(ulong bar)
{
    asm { and RAX, 0xFFFFFFFF00000000; ret; }
    asm { and RAX, 0x00000000FFFFFFFF; ret; }
}

/***********************************************/

/*
TEST_OUTPUT:
---
fail_compilation/iasm1.d(213): Error: invalid operand
fail_compilation/iasm1.d(213): Error: invalid operand
---
*/

// https://issues.dlang.org/show_bug.cgi?id=15239

#line 200

struct T
{
    template opDispatch(string Name, P...)
    {
        static void opDispatch(P) {}
    }
}

void test2()
{
    asm
    {
        call T.foo;
    }
}

/*********************************************/

/* TEST_OUTPUT:
---
fail_compilation/iasm1.d(306): Error: end of instruction expected, not `R8`
fail_compilation/iasm1.d(307): Error: end of instruction expected, not `RDX`
fail_compilation/iasm1.d(310): Error: end of instruction expected, not `RCX`
---
*/

// https://issues.dlang.org/show_bug.cgi?id=17616
// https://issues.dlang.org/show_bug.cgi?id=18373

#line 300

void test3()
{
    asm
    {
        naked;
        mov RAX,[R9][R10]R8;
        mov RAX,[3]RDX;
	mov RAX,[RIP][RIP];
	mov RAX,[RIP][RCX];
	mov RAX,[RIP]RCX;
    }
}

/*********************************************/

/*
TEST_OUTPUT:
---
fail_compilation/iasm1.d(403): Error: missing `]`
---
*/

#line 400

void test4()
{
    asm { inc [; }
}

/*********************************************/

/* TEST_OUTPUT:
---
fail_compilation/iasm1.d(505): Error: function `iasm1.test5` label `L1` is undefined
---
*/

#line 500

void test5()
{
    asm
    {
        jmp L1;
    L2:
        nop;
    }
}

/*********************************************/

/*
// LDC: compiles
test_output:
---
fail_compilation/iasm1.d(615): Error: delegate `iasm1.test6.__foreachbody1` label `L1` is undefined
---
*/

#line 600

struct S
{
    static int opApply(int delegate(ref int) dg)
    {
        return 0;
    }
}

void test6()
{
    foreach(f; S)
    {
        asm
        {
            jmp L1;
        }
        goto L1;
    }
    L1:;
}
