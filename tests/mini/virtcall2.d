enum {
 COMMON,
 INPUT,
 OUTPUT,
 CONDUIT,
 OTHER
}

interface Common
{ int common(); }

interface Input : Common
{ int input(); }

interface Output : Common
{ int output(); }

interface Conduit : Input, Output
{ abstract int conduit(); }

class Abstract : Conduit
{
  abstract int conduit();
  abstract int output();
  abstract int input();
  int common() { return COMMON; }
}

interface Other
{ int other(); }

class Impl : Abstract, Other
{
  int conduit() { return CONDUIT; }
  int output() { return OUTPUT; }
  int other() { return OTHER; }
  int input() { return INPUT; }
}

void main()
{
  auto impl = new Impl;
  
  {
    auto i = impl;
    assert(i.common() == COMMON);
    assert(i.input() == INPUT);
    assert(i.output() == OUTPUT);
    assert(i.conduit() == CONDUIT);
    assert(i.other() == OTHER);
  }
  
  {
    Abstract i = impl;
    assert(i.common() == COMMON);
    assert(i.input() == INPUT);
    assert(i.output() == OUTPUT);
    assert(i.conduit() == CONDUIT);
  }
  
  {
    Conduit i = impl;
    assert(i.common() == COMMON);
    assert(i.input() == INPUT);
    assert(i.output() == OUTPUT);
    assert(i.conduit() == CONDUIT);
  }

  {
    Output i = impl;
    assert(i.output() == OUTPUT);
  }
  
  {
    Common i = impl;
    assert(i.common() == COMMON);
  }  
}
