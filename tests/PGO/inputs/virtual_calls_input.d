module inputs.virtual_calls_input;

import ldc.attributes;

class A
{
    int getNum(int a)
    {
        return a * 2;
    }
}

class D
{
    void doNothing()
    {
    }
}

@weak // disable LLVM reasoning about this function
D createD(int i)
{
    return new D();
}

