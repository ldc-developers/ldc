class A(T)
{
   void foo(void delegate (T) d) {}

   void bar()
   {
     foo(delegate void (T t) {});
   }
}

class B: A!(B) {}

class C: A!(C) {}
