/*******************************************************************************
        Illustrates use of the Arguments class.
*******************************************************************************/

import tango.util.Arguments;
import tango.io.Stdout;
import tango.io.FileConduit;
import tango.text.stream.LineIterator;

void usage()
{
	Stdout("Usage: [OPTIONS]... FILES...").newline;
	Stdout("This is a program that does something.").newline;
	Stdout.newline;
	Stdout("OPTIONS: ").newline;
	Stdout("Output this help message:          -?,      --help").newline;
	Stdout("Do cool things to your files:      -c, -C,  --cool").newline;
	Stdout("Use filename as response file:     -r, -R,  --response").newline;
}

void main(char[][] cmdlArgs)
{
	char[][] implicitArguments = ["files"];
	
	char[][][] argumentAliases;
	argumentAliases ~= ["help",          "?"];
	argumentAliases ~= ["cool",     "C", "c"];
	argumentAliases ~= ["response", "R", "r"];
	
	auto args = new Arguments(cmdlArgs, implicitArguments, argumentAliases);

	bool fileExistsValidation(char[] arg)
	{
		bool rtn;
		FilePath argFile = new FilePath(arg);
		rtn = argFile.exists;
		if (!rtn)
			Stdout.format("Specified path does not exist: {}", arg).newline;
		return rtn;
	}

	bool singleFileValidation(char[][] args, inout char[] invalidArg)
	{
		if (args.length > 1)
		{
			Stdout("Cannot specify multiple paths for argument.").newline;
			invalidArg = args[1];
		}
		else
			return true;
		return false;
	}

	args.addValidation("response", &fileExistsValidation);
	args.addValidation("response", &singleFileValidation);
	args.addValidation("files", true, true);

	bool argsValidated = true;
	try
		args.validate;
	catch (ArgumentException ex)
	{
		if (ex.reason == ArgumentException.ExceptionReason.MISSING_ARGUMENT)
			Stdout.format("Missing Argument: {} ({})", ex.name, ex.msg).newline;
		else if (ex.reason == ArgumentException.ExceptionReason.MISSING_PARAMETER)
			Stdout.format("Missing Parameter to Argument: {} ({})", ex.name, ex.msg).newline;
		else if (ex.reason == ArgumentException.ExceptionReason.INVALID_PARAMETER)
			Stdout.format("Invalid Parameter: {} ({})", ex.name, ex.msg).newline;
		Stdout.newline;
		argsValidated = false;
	}

	if ((!argsValidated) || ("help" in args))
		usage();
	else
	{// ready to run
		if ("response" in args)
		{
            Stdout(args["response"][0]).newline;
            auto file = new FileConduit(args["response"][0]);
            auto lines = new LineIterator!(char)(file);
            char[][] arguments;
            foreach (line; lines)
                arguments ~= line.dup;
            args.parse(arguments, implicitArguments, argumentAliases);
		}
		if ("cool" in args)
		{
            Stdout ("Listing the files to be actioned in a cool way.").newline;
            foreach (char[] file; args["files"])
                Stdout.format("{}", file).newline;
            Stdout ("Cool and secret action performed.").newline;
		}
		if ("x" in args)
			Stdout.format("User set the X factor to '{}'", args["x"]).newline;
	}
}
