// See Github issue 1933.

// RUN: %ldc -c -release -g -O0 %s
// RUN: %ldc -c -release -g -O3 %s

ptrdiff_t countUntil(T, N)(T haystack, N needle)
{
    ptrdiff_t result;
    foreach (elem; haystack) {
        if (elem == needle)
            return result;
        result++;
    }
    return -1;
}

bool foo(alias pred, N)(N haystack)
{
    foreach (elem; haystack) {
        if (pred(elem))
            return true;
    }
    return false;
}

struct MatchTree
{
    struct Tag
    {
    }

    struct Terminal
    {
        string[] varNames;
    }

    Tag[] m_terminalTags;
    Terminal term;

    void rebuildGraph()
    {

        MatchGraphBuilder builder;
        uint process()
        {
            auto aaa = m_terminalTags.length;
            foreach (t; builder.m_nodes)
            {
                auto bbb = term.varNames.countUntil(t.var);
                assert(m_terminalTags.foo!(u => t.index));
            }
            return 0;
        }
    }
}

struct MatchGraphBuilder
{
    struct TerminalTag
    {
        size_t index;
        string var;
    }

    TerminalTag[] m_nodes;
}
