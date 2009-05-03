void main()
{
  static int[] b = [1, 2];
  b[0] = 2;
  
  typedef int[] ia = [1,2];
  static ia a;
  a[0] = 5;
}