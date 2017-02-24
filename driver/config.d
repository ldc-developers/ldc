//===-- driver/configfile.d - LDC config file parsing -------------*- D -*-===//
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
    this (string name, string val)
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
    this (string name, string[] vals)
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
    this (string name, Setting[] children)
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
    auto parser = new Parser(filename);
    return parser.parseConfig();
}

private:

/+

What follows is a recursive descent parser that reads the following EBNF grammar

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


Single line comments are also supported in the form of "//" until line feed.
The "//" sequence is however allowed within strings and doesn't need to be
escaped.
Line feed are not allowed within strings. To span a string over multiple lines,
use concatenation. ("string 1" "string 2")

+/


immutable(char)* toStringz(in string s)
{
    auto nullTerm = s ~ '\0';
    return nullTerm.ptr;
}


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
    case Token.name:        return "\"name\"";
    case Token.assign:      return "':' or '='";
    case Token.str:         return "\"string\"";
    case Token.lbrace:      return "'{'";
    case Token.rbrace:      return "'}'";
    case Token.lbracket:    return "'['";
    case Token.rbracket:    return "']'";
    case Token.semicolon:   return "';'";
    case Token.comma:       return "','";
    case Token.unknown:     return "\"unknown token\"";
    case Token.eof:         return "\"end of file\"";
    }
}

class Parser
{
    const(char)[] filename;
    FILE* file;
    int lineNum;

    int lastChar = ' ';

    struct Ahead
    {
        Token tok;
        string s;
    }
    Ahead ahead;
    Ahead* aheadp;

    this (const(char)* filename)
    {
        this.filename = filename[0 .. strlen(filename)];
        file = fopen(filename, "r");
        if (!file)
        {
            throw new Exception(
                "could not open config file " ~
                this.filename.idup ~ " for reading");
        }
        this.file = file;
    }

    void error(in string msg)
    {
        enum fmt = "Error while reading config file: %s\nline %d: %s";
        char[1024] buf;
        // filename was null terminated
        auto len = snprintf(buf.ptr, 1024, fmt,
                filename.ptr, lineNum, toStringz(msg));
        throw new Exception(buf[0 .. len].idup);
    }

    int getChar()
    {
        int c = fgetc(file);
        if (c == '\n') lineNum += 1;
        return c;
    }

    void ungetChar(int c)
    {
        ungetc(c, file);
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

        while(isspace(lastChar))
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
            else do
            {
                lastChar = getChar();
            }
            while(lastChar != '\n' && lastChar != EOF);
            return getTok(outStr);
        }

        if (isalpha(lastChar))
        {
            string name;
            do
            {
                name ~= cast(char)lastChar;
                lastChar = getChar();
            }
            while (isalnum(lastChar) || lastChar == '-');
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
        case EOF:
            return Token.eof;
        default:
            break;
        }

        if (lastChar == '"')
        {
            string str;
            while (lastChar == '"')
            {
                while(1)
                {
                    lastChar = getChar();
                    if (lastChar == '"') break;
                    if (lastChar == '\n' || lastChar == '\r')
                    {
                        error("Unexpected end of line in string literal");
                    }
                    else if (lastChar == EOF)
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
                            error("Unexpected escape sequence: \\"~cast(char)lastChar);
                            break;
                        }
                    }
                    str ~= cast(char)lastChar;
                }
                lastChar = getChar();
                while(isspace(lastChar)) lastChar = getChar();
            }

            outStr = str;
            return Token.str;
        }

        outStr = [cast(char)lastChar];
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
        error("Was expecting token "~humanReadableToken(expected)~
                ". Got "~humanReadableToken(tok)~s~" instead.");
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
        while(1)
        {
            {
                string s;
                auto t = getTok(s);
                if (t == Token.eof) break;
                else ungetTok(t, s);
            }
            res ~= parseSetting();
        }
        return res;
    }

    Setting parseSetting()
    {
        immutable name = accept(Token.name);

        accept(Token.assign);

        string val;
        string[] arrVal;
        Setting[] grpVal;

        Setting res;

        final switch(parseValue(val, arrVal, grpVal))
        {
        case Setting.Type.scalar:
            res = new ScalarSetting(name, val);
            break;
        case Setting.Type.array:
            res = new ArraySetting(name, arrVal);
            break;
        case Setting.Type.group:
            res = new GroupSetting(name, grpVal);
            break;
        }

        string s;
        immutable t = getTok(s);
        if (t != Token.semicolon && t != Token.comma)
        {
            ungetTok(t, s);
        }

        return res;
    }

    Setting.Type parseValue(out string val,
                            out string[] arrVal,
                            out Setting[] grpVal)
    {
        string s;
        auto t = getTok(s);
        if (t == Token.str)
        {
            val = s;
            return Setting.Type.scalar;
        }
        else if (t == Token.lbracket)
        {
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
                    return Setting.Type.array;
                default:
                    unexpectedTokenError(t, Token.str, s);
                    assert(false);
                }

                // get commar or rbracket
                t = getTok(s);
                switch(t)
                {
                case Token.comma:
                    break;
                case Token.rbracket:
                    return Setting.Type.array;
                default:
                    unexpectedTokenError(t, Token.comma, s);
                    assert(false);
                }
            }
        }
        else if (t == Token.lbrace)
        {
            while (1)
            {
                t = getTok(s);
                if (t == Token.rbrace)
                {
                    return Setting.Type.group;
                }
                ungetTok(t, s);
                grpVal ~= parseSetting();
            }
        }
        error("Was expecting value.");
        assert(false);
    }
}
