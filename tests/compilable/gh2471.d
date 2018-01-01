// RUN: %ldc -c %s

struct Iterator(T : T[])
{
  alias ThisType = Iterator!(T[]);
  alias ItemType = T;

  this(T[] container, size_t index = 0)
  {
    container_ = container;
    index_ = index;
  }

  U opCast(U)() const
    if (is(U == ItemType))
  {
    return this ? container_[index_] : ItemType.init;
  }

  ref ThisType opUnary(string op)()
    if (op == "++" || op == "--")
  {
    mixin(op ~ "index_;");
    return this;
  }

  @property ItemType value() const        { return container_[index_]; }

  alias value this;

  private T[] container_  = null;
  private size_t index_   = 0;
}

void main()
{
  string test = "abcde";
  auto it = Iterator!string(test);
  char ch = it++;
}
