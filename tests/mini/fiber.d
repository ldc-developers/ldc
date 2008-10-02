private import
	tango.core.Thread;

extern(C) void printf(char*, ...);

void foo()
{
		printf("-- I am here\n");
		Fiber.yield();
		printf("-- Now I am here\n");
}

void main()
{
	Fiber f = new Fiber(&foo);

	printf("Running f once\n");
	f.call();
	printf("Running f again\n");
	f.call();
}
