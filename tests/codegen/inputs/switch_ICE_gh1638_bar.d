// Don't make any changes to this file without consulting Github issue 1638 first.

module switch_ICE_gh1638_bar;

import switch_ICE_gh1638;
import std.conv;

static class F
{
    static Q[] X;

    public static void A(int x)
    {
        emplace(X[x].Y);
        return;
    }
}
