module goto1;

void main()
{
    int i;
    goto lbl;
    i++;
lbl:
    assert(i == 0);
}

