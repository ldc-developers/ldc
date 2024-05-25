import dshell;

int main()
{
	version (Windows)
	{
		auto cmd = "$DMD -m$MODEL -c $EXTRA_FILES" ~ SEP ~ "issue24111.c";
		run(cmd);

		import std.process: environment;
		version (LDC)
		{
			// if VSINSTALLDIR is set, LDC assumes INCLUDE is set up too
			environment.remove("VSINSTALLDIR");
		}
		environment.remove("INCLUDE");
		run(cmd);
	}
	return 0;
}
