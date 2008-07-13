module tangotests.gc2;

import tango.core.Memory;

void main()
{
    char[] tmp = new char[2500];
    GC.collect();
}
