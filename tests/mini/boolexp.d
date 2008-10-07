bool got() { return true; }
extern(C) int printf(char*, ...);
void main()
{
  bool b = true && got();
  printf("%d\n", b ? 1 : 0);
}
