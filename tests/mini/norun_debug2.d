module mini.norun_debug2;

import tango.stdc.stdlib : rand;

void main()
{
    size_t iter;
    while (iter < 25)
    {
        if (rand() % 20 == 10)
            *cast(int*)1 = 0;
        ++iter;
    }
    assert(0);
}
