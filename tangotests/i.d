interface IN
{
    void func();
}
abstract class AC
{
    abstract void func();
    long ll;
}
class C : AC
{
    void func()
    {
    }
}

void main()
{
    scope c = new C;
}
