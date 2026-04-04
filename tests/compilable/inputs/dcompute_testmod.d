template hasIndirections(T)
{
    static if (is(T == enum))
        enum hasIndirections = hasIndirections;
}
