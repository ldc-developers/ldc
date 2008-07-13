module nested4;

void func(void delegate() dg) {
  auto v = (){
    dg();
  };
}

void main()
{
    func({});
}
