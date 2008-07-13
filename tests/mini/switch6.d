module tangotests.switch3;

void main()
{
    int i = 2;

    switch(i)
    {
        case 0,1,4,5,6,7,8,9:
            assert(0);
        case 2:
            return;
        case 3:
        {
            i++;
            case 11:
                i++;
        }
        return;
    }
}
