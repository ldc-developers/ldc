module tangotests.volatile1;

import tango.stdc.stdlib;

void main()
{
    int var = rand();
    {
        int i = var;
        volatile;
        int j = i;
    }
}
