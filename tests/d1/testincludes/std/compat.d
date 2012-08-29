module std.compat;

extern (C) int printf(char *, ...);

alias char[] string;
alias wchar[] wstring;
alias dchar[] dstring;

alias Exception Error;
alias bool      bit;
