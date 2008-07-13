module nested1;

void func(int i)
{
    (){
        assert(i == 3);
    }();
}

void main()
{
    func(3);
}
