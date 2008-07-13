extern(C) int printf(char*, ...);

void main()
{
     int[0][10] arr;
     printf("%u\n", &arr[9] - &arr[0]);
     int[0] arr1;
     printf("%p\n", &arr1);
     void[0] arr2;
     printf("%p\n", &arr2);
}
