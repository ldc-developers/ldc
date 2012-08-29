module tangotests.stdout2;

import tango.io.Stdout;

void main()
{
    Stdout.formatln("{} {} {}", "a", "b", 1.0);
}
