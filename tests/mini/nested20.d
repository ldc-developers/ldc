extern(C) int printf(char*, ...);

void main()
{
    int[] arr = [1];
    foreach(s; arr)
    {
        void foo()
	{
	    printf("n %d\n", s);
	}

	printf("o1 %d\n", s);
	foo();
	printf("o2 %d\n", s);
    }
}
