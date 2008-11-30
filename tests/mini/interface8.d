interface InputStream
{
  void foo();
}

interface OutputStream
{
  void bar();
}

interface IConduit : InputStream, OutputStream
{
  abstract uint bufferSize();
}

class Conduit : IConduit
{
  abstract uint bufferSize();
  abstract void foo();
  abstract void bar();
}

interface Selectable
{
  void car();
}

class DeviceConduit : Conduit, Selectable
{
        override uint bufferSize ()
        {
                return 1024 * 16;
        }
  override void foo() {}
  override void bar() {}
  override void car() {}
  int handle;
}

class ConsoleConduit : DeviceConduit
{
  override void foo() {}
  bool redirected;
}

class OtherConduit : Conduit
{
  abstract uint bufferSize();
  override void foo() {}
  override void bar() {}
}

void main()
{
  auto c = new ConsoleConduit;
  IConduit ci = c;
  assert(c.bufferSize == ci.bufferSize);
}

