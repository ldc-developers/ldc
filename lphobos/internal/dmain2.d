/*
 * Placed into the Public Domain.
 * written by Walter Bright
 * www.digitalmars.com
 */

module internal.dmain2;
import object;
import std.c.stdio;
import std.c.string;
import std.c.stdlib;
import std.string;

extern (C) void _STI_monitor_staticctor();
extern (C) void _STD_monitor_staticdtor();
extern (C) void _STI_critical_init();
extern (C) void _STD_critical_term();
extern (C) void gc_init();
extern (C) void gc_term();
extern (C) void _minit();
extern (C) void _moduleCtor();
extern (C) void _moduleDtor();
extern (C) void _moduleUnitTests();

extern (C) bool no_catch_exceptions;

/***********************************
 * The D main() function supplied by the user's program
 */
int main(char[][] args);

/***********************************
 * Substitutes for the C main() function.
 * It's purpose is to wrap the call to the D main()
 * function and catch any unhandled exceptions.
 */

extern (C) int main(size_t argc, char **argv)
{
    char[] *am;
    char[][] args;
    int result;
    int myesp;
    int myebx;

    version (linux)
    {
	_STI_monitor_staticctor();
	_STI_critical_init();
	gc_init();
	am = cast(char[] *) malloc(argc * (char[]).sizeof);
	// BUG: alloca() conflicts with try-catch-finally stack unwinding
	//am = (char[] *) alloca(argc * (char[]).sizeof);
    }
    version (Win32)
    {
	gc_init();
	_minit();
	am = cast(char[] *) alloca(argc * (char[]).sizeof);
    }

    if (no_catch_exceptions)
    {
	_moduleCtor();
	_moduleUnitTests();

	for (size_t i = 0; i < argc; i++)
	{
	    auto len = strlen(argv[i]);
	    am[i] = argv[i][0 .. len];
	}

	args = am[0 .. argc];

	result = main(args);
	_moduleDtor();
	gc_term();
    }
    else
    {
	try
	{
	    _moduleCtor();
	    _moduleUnitTests();

	    for (size_t i = 0; i < argc; i++)
	    {
		auto len = strlen(argv[i]);
		am[i] = argv[i][0 .. len];
	    }

	    args = am[0 .. argc];

	    result = main(args);
	    _moduleDtor();
	    gc_term();
	}
	catch (Object o)
	{
	    version (none)
	    {
		printf("Error: ");
		o.print();
	    }
	    else
	    {
		auto foo = o.toString();
		fprintf(stderr, "Error: %.*s\n", foo.length, foo.ptr);
	    }
	    exit(EXIT_FAILURE);
	}
    }

    version (linux)
    {
	free(am);
	_STD_critical_term();
	_STD_monitor_staticdtor();
    }
    return result;
}

