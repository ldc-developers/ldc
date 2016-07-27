// Test for ICE bug Github issue 1638

// Don't make any changes to this file without consulting GH #1638 first.

// RUN: %ldc -I%S %s %S/inputs/switch_ICE_gh1638_bar.d -c
// RUN: %ldc -I%S %S/inputs/switch_ICE_gh1638_bar.d %s -c

module switch_ICE_gh1638;
public struct Q
{
    C* Y;
}

enum E
{
    A,
    C,
    B
}

struct C
{
    int function(int s, int v) foo = (int s, int v) {

        with (E) switch (s)
        {
            // Crash goes away case A is removed.
        case A:
            if (v == B || v == C)
                return 1;
            break;
        default:
            return 0;
        }
        return 0;
    };
}
