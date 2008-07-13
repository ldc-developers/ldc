module bug5;

struct hah {
    static hah f()
    {
        hah res;
        return res;
    }
    hah g()
    {
        return hah.init;
    }
}

void main()
{
}
