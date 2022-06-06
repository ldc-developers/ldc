//===-- gen/cpp-imitating-naming.d --------------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module gen.cpp_imitating_naming;

private string dTemplateToCPlusPlus(const(string) name) @safe pure
{
	import std.string : indexOf, indexOfAny;

	ptrdiff_t start = 0;
	ptrdiff_t index = name.indexOf('!');

	string result;

	if (index != -1)
	{
		result ~= name[start .. index];
		result ~= '<';

		start = index + 1;

		if (name[index + 1] == '(')
		{
			start++;
			index++;

			size_t enclosedParentheses = 0;

			while (true)
			{
				switch (name[index])
				{
				case '(':
					enclosedParentheses++;
					break;
				case ')':
					enclosedParentheses--;
					break;
				default:
					break;
				}

				if (enclosedParentheses == 0)
					break;

				if (index == cast(ptrdiff_t) name.length - 1)
					break;

				index++;
			}

			if (name[index] != ')')
				index++;
		}
		else
		{
			index = name.indexOfAny(" ,.:<>()[]", start);

			if (index == -1)
				index = name.length;
		}

		result ~= name[start .. index];
		result ~= '>';

		start = index;

		if (start != name.length && name[start] == ')')
			start++;
	}

	if (start != name.length)
		result ~= name[start .. $];

	return result;
}

private string dArrayToCPlusPlus(const(string) name) @safe pure
{
	import std.ascii : isDigit;
	import std.string : indexOf, lastIndexOfAny, replace;

	ptrdiff_t index = name.indexOf('[');

	while (index != -1 && name[index + 1].isDigit)
		index = name.indexOf('[', index + 1);

	string result = name;

	if (index > 0)
	{
		index--;

		if (name[index] == '>' || name[index] == ')')
		{
			size_t enclosedParentheses = 0;
			size_t enclosedChevrons = 0;

			while (true)
			{
				switch (name[index])
				{
				case '(':
					enclosedParentheses++;
					break;
				case ')':
					enclosedParentheses--;
					break;
				case '>':
					enclosedChevrons++;
					break;
				case '<':
					enclosedChevrons--;
					break;
				default:
					break;
				}

				if (enclosedParentheses == 0 && enclosedChevrons == 0)
					break;

				if (index == 0)
					break;

				index--;
			}
		}

		if (index > 0)
		{
			index = name.lastIndexOfAny(" ,<(", index - 1);
			index = index != -1 ? index + 1 : 0;
		}

		ptrdiff_t bracketsStart = name.indexOf('[', index);
		ptrdiff_t bracketsIndex = bracketsStart;

		size_t enclosedSquareBrackets = 0;

		while (true)
		{
			switch (name[bracketsIndex])
			{
			case '[':
				enclosedSquareBrackets++;
				break;
			case ']':
				enclosedSquareBrackets--;
				break;
			default:
				break;
			}

			if (enclosedSquareBrackets == 0)
				break;

			if (bracketsIndex == cast(ptrdiff_t) name.length - 1)
				break;

			bracketsIndex++;
		}

		bracketsIndex++;

		immutable string search = name[index .. bracketsIndex];
		immutable string value = name[index .. bracketsStart];

		if (name[bracketsIndex - 1] == ']')
			bracketsIndex--;

		bracketsStart++;

		immutable string key = name[bracketsStart .. bracketsIndex];

		if (key.length == 0)
		{
			immutable string replaceString = "slice<" ~ value ~ ">";
			result = name.replace(search, replaceString);
		}
		else
		{
			immutable string pairKeyValue = key ~ ", " ~ value;
			immutable string replaceString = "associative_array<" ~ pairKeyValue ~ ">";
			result = name.replace(search, replaceString);
		}
	}

	return result;
}

private string convertDToCPlusPlus(alias identifierModifier)(const(string) name) @safe pure
{
	string result = name;
	string previousResult;

	do
	{
		previousResult = result;
		result = identifierModifier(result);
	}
	while (result != previousResult);

	return result;
}

///
string convertDIdentifierToCPlusPlus(const(string) name) @safe pure
{
	import std.string : replace;

	string result = name.replace(".", "::");

	result = result.convertDToCPlusPlus!dTemplateToCPlusPlus;
	result = result.convertDToCPlusPlus!dArrayToCPlusPlus;

	return result;
}

///
extern (C++, ldc)
const(char)* convertDIdentifierToCPlusPlus(const(char)* name, size_t nameLength) @trusted pure
{
	import std.exception : assumeUnique;
	import std.string : toStringz;

	return name[0 .. nameLength].assumeUnique.convertDIdentifierToCPlusPlus.toStringz;
}
