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

void func()
{
    scope c = new C;
}
