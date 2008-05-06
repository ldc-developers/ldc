extern(C) int printf(char*,...);

class MyClass
{
    this(int i = 4)
    {
        inner = this.new InnerClass;
    }

    class InnerClass : Object.Monitor
    {
        void lock() {}
        void unlock() {}
    }

    InnerClass inner;
}

void func()
{
    scope c = new MyClass(42);
}

void main()
{
    func();
}