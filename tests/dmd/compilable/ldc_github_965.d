class A{}
void fun() {
    A a;
    auto b=a?typeid(a):typeid(a);
}