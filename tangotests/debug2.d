module tangotests.debug2;

import tango.stdc.stdlib : rand;

void main()
{
    size_t iter;
    while (iter < 25)
    {
        if (rand() % 20 == 10)
            *cast(int*)null = 0;
        ++iter;
    }
    assert(0);
}
