// Tests that TypeInfo is generated per CU to enable on demand usage

// RUN: %ldc -of%t_lib%obj -betterC -c %S/inputs/typeinfo_on_demand2.d
// RUN: %ldc -I%S %t_lib%obj -run %s

import inputs.typeinfo_on_demand2;

void main() {
  MyChildClass mcc = new MyChildClass;
  mcc.method();
}

extern(C++) class MyChildClass : MyClass, MyInterface2 {
    override void method() {}
}