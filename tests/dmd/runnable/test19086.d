// REQUIRED_ARGS: -g
// REQUIRED_ARGS(linux freebsd dragonflybsd): -L-export-dynamic
// LDC (FreeBSD's libexecinfo apparently doesn't like elided frame pointers): REQUIRED_ARGS(freebsd): -link-defaultlib-debug -frame-pointer=all
// DISABLED: LDC_win32 // no file/line info for the `run19086` frame (and only that frame), even without -O
// PERMUTE_ARGS:
// DISABLED: osx

void run19086()
{
	version (LDC) pragma(inline, false);
	long x = 1;
	int y = 0;
#line 20
    throw newException();
}

// moved here to keep run19086 short
Exception newException() { return new Exception("hi"); }

void test19086()
{
	try
	{
		run19086();
	}
	catch(Exception e)
	{
		int line = findLineStackTrace(e.toString(), "run19086");
		assert(line >= 20 && line <= 21);
	}
}

int findLineStackTrace(string msg, string func)
{
    // find line number of _Dmain in stack trace
    // on linux:   file.d:line _Dmain [addr]
    // on windows: addr in _Dmain at file.d(line)
    int line = 0;
    bool found = false;
    for (size_t pos = 0; pos + func.length < msg.length; pos++)
    {
        if (msg[pos] == '\n')
        {
            line = 0;
            found = false;
        }
        else if ((msg[pos] == ':' || msg[pos] == '(') && line == 0)
        {
            for (pos++; pos < msg.length && msg[pos] >= '0' && msg[pos] <= '9'; pos++)
                line = line * 10 + msg[pos] - '0';
            if (line > 0 && found)
                return line;
        }
        else if (msg[pos .. pos + func.length] == func)
        {
            found = true;
            if (line > 0 && found)
                return line;
        }
    }
    return 0;
}

void main()
{
	test19086();
}
