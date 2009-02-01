struct Color {
  uint c;
  static Color opCall(uint _c) { Color ret; ret.c = _c; return ret; }
}

// run at compile time
static const Color white = Color(0xffffffff);

void main()
{
  assert(white.c == 0xffffffff);
}
