module inputs.typeinfo_on_demand2;

extern(C++) class MyClass : MyInterface1 {
    void method() {}
}

extern(C++) interface MyInterface1 {
    void method();
}

extern(C++) interface MyInterface2 {
    void method();
}