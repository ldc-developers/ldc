//===-- driver/config.d - LDC config file parsing -----------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Parsing engine for the LDC config file (ldc2.conf).
//
//===----------------------------------------------------------------------===//
module driver.config;

import core.stdc.ctype;
import core.stdc.stdio;
import core.stdc.string;


class Setting
{
    enum Type
    {
        scalar,
        array,
        group,
    }

    this(string name, Type type)
    {
        _name = name;
        _type = type;
    }

    @property string name() const
    {
        return _name;
    }

    @property Type type() const
    {
        return _type;
    }

    private string _name;
    private Type _type;
}


class ScalarSetting : Setting
{
    this(string name, string val)
    {
        super(name, Type.scalar);
        _val = val;
    }

    @property string val() const
    {
        return _val;
    }

    private string _val;
}


class ArraySetting : Setting
{
    this(string name, string[] vals)
    {
        super(name, Type.array);
        _vals = vals;
    }

    @property const(string)[] vals() const
    {
        return _vals;
    }

    private string[] _vals;
}

class GroupSetting : Setting
{
    this(string name, Setting[] children)
    {
        super(name, Type.group);
        _children = children;
    }

    @property const(Setting)[] children() const
    {
        return _children;
    }

    private Setting[] _children;
}


Setting[] parseConfigFile(const(char)* filename)
{
    auto dFilename = filename[0 .. strlen(filename)].idup;

    auto file = fopen(filename, "r");
    if (!file)
    {
        throw new Exception(
            "could not open config file " ~
            dFilename ~ " for reading");
    }

    fseek(file, 0, SEEK_END);
    const fileLength = ftell(file);
    rewind(file);

    auto content = new char[fileLength];
    const numRead = fread(content.ptr, 1, fileLength, file);
    content = content[0 .. numRead];

    auto parser = new Parser(cast(string) content, dFilename);
    return parser.parseConfig();
}


private:

/+

What follows is a recursive descent parser that reads the following
EBNF grammar.
It is a subset of the libconfig grammar (http://www.hyperrealm.com/libconfig).

config  =   { ows , setting } , ows ;
setting =   name , (":" | "=") , value , [";" | ","] ;
name    =   alpha , { alpha | digit | "_" | "-" } ;
value   =   string | array | group ;
array   =   "[" , ows ,
                { string , ows , "," , ows } ,
            "]" ;
group   =   "{" , ows , { setting , ows } , "}" ;
string  =   quotstr, { ows , quotstr } ;
quotstr =   '"' , { ? any char but '"', '\n' and '\r' ? | escseq } , '"' ;
escseq  =   "\" , ["\" | '"' | "r" | "n" | "t" ] ;
alpha   =   ? any char between "a" and "z" included
                    or between "A" and "Z" included ? ;
digit   =   ? any char between "0" and "9" included ? ;
ows     =   [ ws ] ; (* optional white space *)
ws      =   ? white space (space, tab, line feed ...) ? ;


Single line comments are also supported. They start with "//" and span until
line feed.
The "//" sequence is however allowed within strings and doesn't need to be
escaped.
White space are significant only within strings.
Physical line feeds are not allowed within strings. To span a string over
multiple lines, use concatenation ("hello " "world" == "hello world").
The following escape sequences are allowed in strings:
  - \\
  - \"
  - \r
  - \n
  - \t

+/

enum Token
{
    name,
    assign,         // ':' or '='
    str,
    lbrace,         // '{'
    rbrace,         // '}'
    lbracket,       // '['
    rbracket,       // ']'
    semicolon,      // ';'
    comma,          // ','
    unknown,
    eof,
}

string humanReadableToken(in Token tok)
{
    final switch(tok)
    {
    case Token.name:        return `"name"`;
    case Token.assign:      return `':' or '='`;
    case Token.str:         return `"string"`;
    case Token.lbrace:      return `'{'`;
    case Token.rbrace:      return `'}'`;
    case Token.lbracket:    return `'['`;
    case Token.rbracket:    return `']'`;
    case Token.semicolon:   return `';'`;
    case Token.comma:       return `','`;
    case Token.unknown:     return `"unknown token"`;
    case Token.eof:         return `"end of file"`;
    }
}

class Parser
{
    string filename;
    string content;
    int index;
    int lineNum = 1;

    char lastChar = ' ';

    static struct Ahead
    {
        Token tok;
        string s;
    }
    Ahead ahead;
    Ahead* aheadp;

    this(string content, string filename = "")
    {
        this.filename = filename;
        this.content = content;
    }

    void error(in string msg)
    {
        enum fmt = "Error while reading config file: %.*s\nline %d: %.*s";
        char[1024] buf;
        auto len = snprintf(buf.ptr, buf.length, fmt,
                filename.length, filename.ptr, lineNum, msg.length, msg.ptr);
        throw new Exception(buf[0 .. len].idup);
    }

    char getChar()
    {
        if (index == content.length)
            return '\0';
        const c = content[index++];
        if (c == '\n')
            ++lineNum;
        return c;
    }

    Token getTok(out string outStr)
    {
        if (aheadp)
        {
            immutable tok = aheadp.tok;
            outStr = aheadp.s;
            aheadp = null;
            return tok;
        }

        while (isspace(lastChar))
        {
            lastChar = getChar();
        }

        if (lastChar == '/')
        {
            lastChar = getChar();
            if (lastChar != '/')
            {
                outStr = "/";
                return Token.unknown;
            }

            do
            {
                lastChar = getChar();
            } while (lastChar != '\n' && lastChar != '\0');

            return getTok(outStr);
        }

        if (isalpha(lastChar))
        {
            string name;
            do
            {
                name ~= lastChar;
                lastChar = getChar();
            }
            while (isalnum(lastChar) || lastChar == '_' || lastChar == '-');
            outStr = name;
            return Token.name;
        }

        switch (lastChar)
        {
        case ':':
        case '=':
            lastChar = getChar();
            return Token.assign;
        case ';':
            lastChar = getChar();
            return Token.semicolon;
        case ',':
            lastChar = getChar();
            return Token.comma;
        case '{':
            lastChar = getChar();
            return Token.lbrace;
        case '}':
            lastChar = getChar();
            return Token.rbrace;
        case '[':
            lastChar = getChar();
            return Token.lbracket;
        case ']':
            lastChar = getChar();
            return Token.rbracket;
        case '\0':
            return Token.eof;
        default:
            break;
        }

        if (lastChar == '"')
        {
            string str;
            while (lastChar == '"')
            {
                while (1)
                {
                    lastChar = getChar();
                    if (lastChar == '"') break;
                    if (lastChar == '\n' || lastChar == '\r')
                    {
                        error("Unexpected end of line in string literal");
                    }
                    else if (lastChar == '\0')
                    {
                        error("Unexpected end of file in string literal");
                    }
                    if (lastChar == '\\')
                    {
                        lastChar = getChar();
                        switch(lastChar)
                        {
                        case '\\':
                        case '"':
                            break;
                        case 'r':
                            lastChar = '\r';
                            break;
                        case 'n':
                            lastChar = '\n';
                            break;
                        case 't':
                            lastChar = '\t';
                            break;
                        default:
                            error("Unexpected escape sequence: \\" ~ lastChar);
                            break;
                        }
                    }
                    str ~= lastChar;
                }
                lastChar = getChar();
                while (isspace(lastChar)) lastChar = getChar();
            }

            outStr = str;
            return Token.str;
        }

        outStr = [lastChar];
        lastChar = getChar();
        return Token.unknown;
    }

    void ungetTok(in Token tok, in string s)
    {
        assert(!aheadp, "can only have one look ahead");
        ahead.tok = tok;
        ahead.s = s;
        aheadp = &ahead;
    }

    void unexpectedTokenError(in Token tok, in Token expected, string s)
    {
        s = s.length ? " ("~s~")" : "";
        error("Was expecting token " ~ humanReadableToken(expected) ~
                ". Got " ~ humanReadableToken(tok) ~ s ~ " instead.");
    }

    string accept(in Token expected)
    {
        string s;
        immutable tok = getTok(s);
        if (tok != expected)
        {
            unexpectedTokenError(tok, expected, s);
        }
        return s;
    }

    Setting[] parseConfig()
    {
        Setting[] res;
        while (1)
        {
            {
                string s;
                auto t = getTok(s);
                if (t == Token.eof)
                {
                    break;
                }
                ungetTok(t, s);
            }
            res ~= parseSetting();
        }
        return res;
    }

    Setting parseSetting()
    {
        immutable name = accept(Token.name);

        accept(Token.assign);

        Setting res = parseValue(name);

        string s;
        immutable t = getTok(s);
        if (t != Token.semicolon && t != Token.comma)
        {
            ungetTok(t, s);
        }

        return res;
    }

    Setting parseValue(string name)
    {
        string s;
        auto t = getTok(s);
        if (t == Token.str)
        {
            return new ScalarSetting(name, s);
        }
        else if (t == Token.lbracket)
        {
            string[] arrVal;
            while (1)
            {
                // get string or rbracket
                t = getTok(s);
                switch(t)
                {
                case Token.str:
                    arrVal ~= s;
                    break;
                case Token.rbracket:
                    return new ArraySetting(name, arrVal);
                default:
                    unexpectedTokenError(t, Token.str, s);
                    assert(false);
                }

                // get comma or rbracket
                t = getTok(s);
                switch(t)
                {
                case Token.comma:
                    break;
                case Token.rbracket:
                    return new ArraySetting(name, arrVal);;
                default:
                    unexpectedTokenError(t, Token.comma, s);
                    assert(false);
                }
            }
        }
        else if (t == Token.lbrace)
        {
            Setting[] grpVal;
            while (1)
            {
                t = getTok(s);
                if (t == Token.rbrace)
                {
                    return new GroupSetting(name, grpVal);
                }
                ungetTok(t, s);
                grpVal ~= parseSetting();
            }
        }
        error("Was expecting value.");
        assert(false);
    }
}

unittest
{
    static void testScalar(string input, string expected)
    {
        auto setting = new Parser(input).parseValue(null);
        assert(setting.type == Setting.Type.scalar);
        assert((cast(ScalarSetting) setting).val == expected);
    }

    testScalar(`"abc\r\ndef\t\"quoted/\\123\""`,
                "abc\r\ndef\t\"quoted/\\123\"");
    testScalar(`"concatenated" " multiline"
                " strings"`, "concatenated multiline strings");

    enum input =
`// comment

// comment
group-1:
{
    // comment
    scalar = "abc";
    // comment
    array_1 = [ "a", "b" ];
    array_2 = [
        "c",
    ];
};
// comment
group-2: { emptyArray = []; };
`;

    auto settings = new Parser(input).parseConfig();
    assert(settings.length == 2);

    assert(settings[0].name == "group-1");
    assert(settings[0].type == Setting.Type.group);
    auto group1 = cast(GroupSetting) settings[0];
    assert(group1.children.length == 3);

    assert(group1.children[0].name == "scalar");
    assert(group1.children[0].type == Setting.Type.scalar);
    assert((cast(ScalarSetting) group1.children[0]).val == "abc");

    assert(group1.children[1].name == "array_1");
    assert(group1.children[1].type == Setting.Type.array);
    auto array1 = cast(ArraySetting) group1.children[1];
    assert(array1.vals.length == 2);
    assert(array1.vals[0] == "a");
    assert(array1.vals[1] == "b");

    assert(group1.children[2].name == "array_2");
    assert(group1.children[2].type == Setting.Type.array);
    auto array2 = cast(ArraySetting) group1.children[2];
    assert(array2.vals.length == 1);
    assert(array2.vals[0] == "c");

    assert(settings[1].name == "group-2");
    assert(settings[1].type == Setting.Type.group);
    auto group2 = cast(GroupSetting) settings[1];
    assert(group2.children.length == 1);

    assert(group2.children[0].name == "emptyArray");
    assert(group2.children[0].type == Setting.Type.array);
    auto emptyArray = cast(ArraySetting) group2.children[0];
    assert(emptyArray.vals.length == 0);
}
