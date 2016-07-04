module inputs.inlinables_staticvar;

/+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/

private int atModuleScope;

pragma(inline, true) void addToModuleScopeInline(int i)
{
    atModuleScope += i;
}

pragma(inline, false) void addToModuleScopeOutline(int i)
{
    atModuleScope += i;
}

pragma(inline, false) bool equalModuleScope(int i)
{
    return atModuleScope == i;
}

/+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/

pragma(inline, true) bool addAndCheckInsideFunc(int checkbefore, int increment)
{
    static int insideFunc;

    if (insideFunc != checkbefore)
        return false;

    insideFunc += increment;
    return true;
}

pragma(inline, false) bool addAndCheckInsideFuncIndirect(int checkbefore, int increment)
{
    return addAndCheckInsideFunc(checkbefore, increment);
}

/+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/

pragma(inline, true) bool addAndCheckInsideNestedFunc(int checkbefore, int increment)
{
    pragma(inline, true)
    bool addCheckNested(int checkbefore, int increment)
    {
        static int insideFunc;

        if (insideFunc != checkbefore)
            return false;

        insideFunc += increment;
        return true;
    }

    return addCheckNested(checkbefore, increment);
}

pragma(inline, false) bool addAndCheckInsideNestedFuncIndirect(int checkbefore, int increment)
{
    return addAndCheckInsideNestedFunc(checkbefore, increment);
}

/+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/

pragma(inline, true) bool addAndCheckNestedStruct(int checkbefore, int increment)
{
    struct NestedStruct
    {
        static int structValue;
    }

    if (NestedStruct.structValue != checkbefore)
        return false;

    NestedStruct.structValue += increment;
    return true;
}

pragma(inline, false) bool addAndCheckNestedStructIndirect(int checkbefore, int increment)
{
    return addAndCheckNestedStruct(checkbefore, increment);
}

/+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++/
