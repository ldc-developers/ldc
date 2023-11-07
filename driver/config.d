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
    import dmd.location : Loc;
    import dmd.root.string : toDString;
    import dmd.utils;

    const dFilename = filename.toDString;
    auto content = readFile(Loc.initial, dFilename).extractSlice();

    // skip UTF-8 BOM
    if (content.length >= 3 && content[0 .. 3] == "\xEF\xBB\xBF")
        content = content[3 .. $];

    auto parser = Parser(cast(string) content, cast(string) dFilename);
    return parser.parseConfig();
}


private:

/+

What follows is a recursive descent parser that reads the following
EBNF grammar.
It is a subset of the libconfig grammar (http://www.hyperrealm.com/libconfig).

config  =   { ows , setting } , ows ;
setting =   (name | string) , (":" | "=") , value , [";" | ","] ;
name    =   alpha , { alpha | digit | "_" | "-" } ;
value   =   string | array | group ;
array   =   "[" , ows ,
                { string , ows , "," , ows } ,
            "]" ;
group   =   "{" , ows , { setting , ows } , "}" ;
string  =   ( quotstr , { ows , quotstr } ) |
            ( btstr , { ows, btstr } ) ;
quotstr =   '"' , { ? any char but '"', '\n' and '\r' ? | escseq } , '"' ;
escseq  =   "\" , ["\" | '"' | "r" | "n" | "t" ] ;
btstr   =   '`' , { ? any char but '`' ? } , '`' ;
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

struct Parser
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

    this(string content, string filename = null)
    {
        this.filename = filename;
        this.content = content;
    }

    void error(in string msg)
    {
        enum fmt = "Error while reading config file: %.*s\nline %d: %.*s";
        char[1024] buf;
        auto len = snprintf(buf.ptr, buf.length, fmt, cast(int) filename.length,
                            filename.ptr, lineNum, cast(int) msg.length, msg.ptr);
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
            }
            while (lastChar != '\n' && lastChar != '\0');
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

        if (lastChar == '`')
        {
            string str;
            while (lastChar == '`')
            {
                while (1)
                {
                    lastChar = getChar();
                    if (lastChar == '`') break;
                    if (lastChar == '\0')
                    {
                        error("Unexpected end of file in string literal");
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
        string name;
        auto t = getTok(name);
        if (t != Token.name && t != Token.str)
        {
            unexpectedTokenError(t, Token.name, name);
            assert(false);
        }

        accept(Token.assign);

        Setting res = parseValue(name);

        string s;
        t = getTok(s);
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
                    return new ArraySetting(name, arrVal);
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
        auto setting = Parser(input).parseValue(null);
        assert(setting.type == Setting.Type.scalar);
        assert((cast(ScalarSetting) setting).val == expected);
    }

    testScalar(`""`, "");
    testScalar(`"abc\r\ndef\t\"quoted/\\123\""`,
                "abc\r\ndef\t\"quoted/\\123\"");
    testScalar(`"concatenated" " multiline"
                " strings"`, "concatenated multiline strings");
    testScalar("`abc\n\\ //comment \"`",
                "abc\n\\ //comment \"");
    testScalar(`"Üņïčöđë"`, "Üņïčöđë");
}

unittest
{
    static void testArray(string input, string[] expected)
    {
        auto setting = Parser(input).parseValue(null);
        assert(setting.type == Setting.Type.array);
        assert((cast(ArraySetting) setting).vals == expected);
    }

    testArray(`[]`, []);
    testArray(`[ "a" ]`, [ "a" ]);
    testArray(`[ "a", ]`, [ "a" ]);
    testArray(`[ "a", "b" ]`, [ "a", "b" ]);
    testArray(`[
            // comment
            "a",
            // comment
            "b"
        ]`, [ "a", "b" ]);
}

unittest
{
    enum input =
`// comment

// comment
group-1_2: {};
// comment
"86(_64)?-.*linux\\.?":
{
    // comment
    scalar = "abc";
    // comment
    Array_1-2 = [ "a" ];
};
`;

    auto settings = Parser(input).parseConfig();
    assert(settings.length == 2);

    assert(settings[0].name == "group-1_2");
    assert(settings[0].type == Setting.Type.group);
    assert((cast(GroupSetting) settings[0]).children == []);

    assert(settings[1].name == "86(_64)?-.*linux\\.?");
    assert(settings[1].type == Setting.Type.group);
    auto group2 = cast(GroupSetting) settings[1];
    assert(group2.children.length == 2);

    assert(group2.children[0].name == "scalar");
    assert(group2.children[0].type == Setting.Type.scalar);
    assert((cast(ScalarSetting) group2.children[0]).val == "abc");

    assert(group2.children[1].name == "Array_1-2");
    assert(group2.children[1].type == Setting.Type.array);
    assert((cast(ArraySetting) group2.children[1]).vals == [ "a" ]);
}
