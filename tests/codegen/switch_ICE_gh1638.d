// Test for ICE bug Github issue 1638

// Don't make any changes/additions to this file without consulting GH #1638 first.

// RUN: %ldc -I%S %S/inputs/switch_ICE_gh1638_bar.d %s -c
// RUN: %ldc -I%S %s %S/inputs/switch_ICE_gh1638_bar.d -c

import switch_ICE_gh1638_bar;

int main()
{
    return T().fun(123);
}
