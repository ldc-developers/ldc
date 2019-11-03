/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/lexer.d, _lexer.d)
 * Documentation:  https://dlang.org/phobos/dmd_lexer.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/lexer.d
 */

module dmd.lexer;

import core.stdc.ctype;
import core.stdc.errno;
import core.stdc.stdarg;
import core.stdc.stdio;
import core.stdc.string;
import core.stdc.time;

import dmd.entity;
import dmd.errors;
import dmd.globals;
import dmd.id;
import dmd.identifier;
import dmd.root.ctfloat;
import dmd.root.outbuffer;
import dmd.root.port;
import dmd.root.rmem;
import dmd.tokens;
import dmd.utf;
import dmd.utils;

nothrow:

private enum LS = 0x2028;       // UTF line separator
private enum PS = 0x2029;       // UTF paragraph separator

/********************************************
 * Do our own char maps
 */
private static immutable cmtable = () {
    ubyte[256] table;
    foreach (const c; 0 .. table.length)
    {
        if ('0' <= c && c <= '7')
            table[c] |= CMoctal;
        if (c_isxdigit(c))
            table[c] |= CMhex;
        if (c_isalnum(c) || c == '_')
            table[c] |= CMidchar;

        switch (c)
        {
            case 'x': case 'X':
            case 'b': case 'B':
                table[c] |= CMzerosecond;
                break;

            case '0': .. case '9':
            case 'e': case 'E':
            case 'f': case 'F':
            case 'l': case 'L':
            case 'p': case 'P':
            case 'u': case 'U':
            case 'i':
            case '.':
            case '_':
                table[c] |= CMzerosecond | CMdigitsecond;
                break;

            default:
                break;
        }

        switch (c)
        {
            case '\\':
            case '\n':
            case '\r':
            case 0:
            case 0x1A:
            case '\'':
                break;
            default:
                if (!(c & 0x80))
                    table[c] |= CMsinglechar;
                break;
        }
    }
    return table;
}();

private
{
    enum CMoctal  = 0x1;
    enum CMhex    = 0x2;
    enum CMidchar = 0x4;
    enum CMzerosecond = 0x8;
    enum CMdigitsecond = 0x10;
    enum CMsinglechar = 0x20;
}

private bool isoctal(const char c) pure @nogc @safe
{
    return (cmtable[c] & CMoctal) != 0;
}

private bool ishex(const char c) pure @nogc @safe
{
    return (cmtable[c] & CMhex) != 0;
}

private bool isidchar(const char c) pure @nogc @safe
{
    return (cmtable[c] & CMidchar) != 0;
}

private bool isZeroSecond(const char c) pure @nogc @safe
{
    return (cmtable[c] & CMzerosecond) != 0;
}

private bool isDigitSecond(const char c) pure @nogc @safe
{
    return (cmtable[c] & CMdigitsecond) != 0;
}

private bool issinglechar(const char c) pure @nogc @safe
{
    return (cmtable[c] & CMsinglechar) != 0;
}

private bool c_isxdigit(const int c) pure @nogc @safe
{
    return (( c >= '0' && c <= '9') ||
            ( c >= 'a' && c <= 'f') ||
            ( c >= 'A' && c <= 'F'));
}

private bool c_isalnum(const int c) pure @nogc @safe
{
    return (( c >= '0' && c <= '9') ||
            ( c >= 'a' && c <= 'z') ||
            ( c >= 'A' && c <= 'Z'));
}

unittest
{
    //printf("lexer.unittest\n");
    /* Not much here, just trying things out.
     */
    string text = "int"; // We rely on the implicit null-terminator
    scope diagnosticReporter = new StderrDiagnosticReporter(global.params.useDeprecated);
    scope Lexer lex1 = new Lexer(null, text.ptr, 0, text.length, 0, 0, diagnosticReporter);
    TOK tok;
    tok = lex1.nextToken();
    //printf("tok == %s, %d, %d\n", Token::toChars(tok), tok, TOK.int32);
    assert(tok == TOK.int32);
    tok = lex1.nextToken();
    assert(tok == TOK.endOfFile);
    tok = lex1.nextToken();
    assert(tok == TOK.endOfFile);
    tok = lex1.nextToken();
    assert(tok == TOK.endOfFile);
}

unittest
{
    // We don't want to see Lexer error output during these tests.
    uint errors = global.startGagging();
    scope(exit) global.endGagging(errors);

    // Test malformed input: even malformed input should end in a TOK.endOfFile.
    static immutable char[][] testcases =
    [   // Testcase must end with 0 or 0x1A.
        [0], // not malformed, but pathological
        ['\'', 0],
        ['\'', 0x1A],
        ['{', '{', 'q', '{', 0],
        [0xFF, 0],
        [0xFF, 0x80, 0],
        [0xFF, 0xFF, 0],
        [0xFF, 0xFF, 0],
        ['x', '"', 0x1A],
    ];

    foreach (testcase; testcases)
    {
        scope diagnosticReporter = new StderrDiagnosticReporter(global.params.useDeprecated);
        scope Lexer lex2 = new Lexer(null, testcase.ptr, 0, testcase.length-1, 0, 0, diagnosticReporter);
        TOK tok = lex2.nextToken();
        size_t iterations = 1;
        while ((tok != TOK.endOfFile) && (iterations++ < testcase.length))
        {
            tok = lex2.nextToken();
        }
        assert(tok == TOK.endOfFile);
        tok = lex2.nextToken();
        assert(tok == TOK.endOfFile);
    }
}

/***********************************************************
 */
class Lexer
{
    private __gshared OutBuffer stringbuffer;

    Loc scanloc;            // for error messages
    Loc prevloc;            // location of token before current

    const(char)* p;         // current character

    Token token;

    private
    {
        const(char)* base;      // pointer to start of buffer
        const(char)* end;       // pointer to last element of buffer
        const(char)* line;      // start of current line

        bool doDocComment;      // collect doc comment information
        bool anyToken;          // seen at least one token
        bool commentToken;      // comments are TOK.comment's
        int inTokenStringConstant; // can be larger than 1 when in nested q{} strings
        int lastDocLine;        // last line of previous doc comment

        DiagnosticReporter diagnosticReporter;
        Token* tokenFreelist;
    }

  nothrow:

    /*********************
     * Creates a Lexer for the source code base[begoffset..endoffset+1].
     * The last character, base[endoffset], must be null (0) or EOF (0x1A).
     *
     * Params:
     *  filename = used for error messages
     *  base = source code, must be terminated by a null (0) or EOF (0x1A) character
     *  begoffset = starting offset into base[]
     *  endoffset = the last offset to read into base[]
     *  doDocComment = handle documentation comments
     *  commentToken = comments become TOK.comment's
     *  diagnosticReporter = the diagnostic reporter to use
     */
    this(const(char)* filename, const(char)* base, size_t begoffset,
        size_t endoffset, bool doDocComment, bool commentToken,
        DiagnosticReporter diagnosticReporter) pure
    in
    {
        assert(diagnosticReporter !is null);
    }
    body
    {
        this.diagnosticReporter = diagnosticReporter;
        scanloc = Loc(filename, 1, 1);
        //printf("Lexer::Lexer(%p,%d)\n",base,length);
        //printf("lexer.filename = %s\n", filename);
        token = Token.init;
        this.base = base;
        this.end = base + endoffset;
        p = base + begoffset;
        line = p;
        this.doDocComment = doDocComment;
        this.commentToken = commentToken;
        this.inTokenStringConstant = 0;
        this.lastDocLine = 0;
        //initKeywords();
        /* If first line starts with '#!', ignore the line
         */
        if (p && p[0] == '#' && p[1] == '!')
        {
            p += 2;
            while (1)
            {
                char c = *p++;
                switch (c)
                {
                case 0:
                case 0x1A:
                    p--;
                    goto case;
                case '\n':
                    break;
                default:
                    continue;
                }
                break;
            }
            endOfLine();
        }
    }

    /// Returns: `true` if any errors occurred during lexing or parsing.
    final bool errors()
    {
        return diagnosticReporter.errorCount > 0;
    }

    /// Returns: a newly allocated `Token`.
    Token* allocateToken() pure nothrow @safe
    {
        if (tokenFreelist)
        {
            Token* t = tokenFreelist;
            tokenFreelist = t.next;
            t.next = null;
            return t;
        }
        return new Token();
    }

    /// Frees the given token by returning it to the freelist.
    private void releaseToken(Token* token) pure nothrow @nogc @safe
    {
        if (mem.isGCEnabled)
            *token = Token.init;
        token.next = tokenFreelist;
        tokenFreelist = token;
    }

    final TOK nextToken()
    {
        prevloc = token.loc;
        if (token.next)
        {
            Token* t = token.next;
            memcpy(&token, t, Token.sizeof);
            releaseToken(t);
        }
        else
        {
            scan(&token);
        }
        //printf(token.toChars());
        return token.value;
    }

    /***********************
     * Look ahead at next token's value.
     */
    final TOK peekNext()
    {
        return peek(&token).value;
    }

    /***********************
     * Look 2 tokens ahead at value.
     */
    final TOK peekNext2()
    {
        Token* t = peek(&token);
        return peek(t).value;
    }

    /****************************
     * Turn next token in buffer into a token.
     */
    final void scan(Token* t)
    {
        const lastLine = scanloc.linnum;
        Loc startLoc;
        t.blockComment = null;
        t.lineComment = null;

        while (1)
        {
            t.ptr = p;
            //printf("p = %p, *p = '%c'\n",p,*p);
            t.loc = loc();
            switch (*p)
            {
            case 0:
            case 0x1A:
                t.value = TOK.endOfFile; // end of file
                // Intentionally not advancing `p`, such that subsequent calls keep returning TOK.endOfFile.
                return;
            case ' ':
            case '\t':
            case '\v':
            case '\f':
                p++;
                continue; // skip white space
            case '\r':
                p++;
                if (*p != '\n') // if CR stands by itself
                    endOfLine();
                continue; // skip white space
            case '\n':
                p++;
                endOfLine();
                continue; // skip white space
            case '0':
                if (!isZeroSecond(p[1]))        // if numeric literal does not continue
                {
                    ++p;
                    t.unsvalue = 0;
                    t.value = TOK.int32Literal;
                    return;
                }
                goto Lnumber;

            case '1': .. case '9':
                if (!isDigitSecond(p[1]))       // if numeric literal does not continue
                {
                    t.unsvalue = *p - '0';
                    ++p;
                    t.value = TOK.int32Literal;
                    return;
                }
            Lnumber:
                t.value = number(t);
                return;

            case '\'':
                if (issinglechar(p[1]) && p[2] == '\'')
                {
                    t.unsvalue = p[1];        // simple one character literal
                    t.value = TOK.charLiteral;
                    p += 3;
                }
                else
                    t.value = charConstant(t);
                return;
            case 'r':
                if (p[1] != '"')
                    goto case_ident;
                p++;
                goto case '`';
            case '`':
                wysiwygStringConstant(t);
                return;
            case 'x':
                if (p[1] != '"')
                    goto case_ident;
                p++;
                auto start = p;
                auto hexString = new OutBuffer();
                t.value = hexStringConstant(t);
                hexString.write(start[0 .. p - start]);
                error("Built-in hex string literals are obsolete, use `std.conv.hexString!%s` instead.", hexString.extractChars());
                return;
            case 'q':
                if (p[1] == '"')
                {
                    p++;
                    delimitedStringConstant(t);
                    return;
                }
                else if (p[1] == '{')
                {
                    p++;
                    tokenStringConstant(t);
                    return;
                }
                else
                    goto case_ident;
            case '"':
                escapeStringConstant(t);
                return;
            case 'a':
            case 'b':
            case 'c':
            case 'd':
            case 'e':
            case 'f':
            case 'g':
            case 'h':
            case 'i':
            case 'j':
            case 'k':
            case 'l':
            case 'm':
            case 'n':
            case 'o':
            case 'p':
                /*case 'q': case 'r':*/
            case 's':
            case 't':
            case 'u':
            case 'v':
            case 'w':
                /*case 'x':*/
            case 'y':
            case 'z':
            case 'A':
            case 'B':
            case 'C':
            case 'D':
            case 'E':
            case 'F':
            case 'G':
            case 'H':
            case 'I':
            case 'J':
            case 'K':
            case 'L':
            case 'M':
            case 'N':
            case 'O':
            case 'P':
            case 'Q':
            case 'R':
            case 'S':
            case 'T':
            case 'U':
            case 'V':
            case 'W':
            case 'X':
            case 'Y':
            case 'Z':
            case '_':
            case_ident:
                {
                    while (1)
                    {
                        const c = *++p;
                        if (isidchar(c))
                            continue;
                        else if (c & 0x80)
                        {
                            const s = p;
                            const u = decodeUTF();
                            if (isUniAlpha(u))
                                continue;
                            error("char 0x%04x not allowed in identifier", u);
                            p = s;
                        }
                        break;
                    }
                    Identifier id = Identifier.idPool(cast(char*)t.ptr, cast(uint)(p - t.ptr));
                    t.ident = id;
                    t.value = cast(TOK)id.getValue();
                    anyToken = 1;
                    if (*t.ptr == '_') // if special identifier token
                    {
                        __gshared bool initdone = false;
                        __gshared char[11 + 1] date;
                        __gshared char[8 + 1] time;
                        __gshared char[24 + 1] timestamp;
                        if (!initdone) // lazy evaluation
                        {
                            initdone = true;
                            time_t ct;
                            .time(&ct);
                            const p = ctime(&ct);
                            assert(p);
                            sprintf(&date[0], "%.6s %.4s", p + 4, p + 20);
                            sprintf(&time[0], "%.8s", p + 11);
                            sprintf(&timestamp[0], "%.24s", p);
                        }
                        if (id == Id.DATE)
                        {
                            t.ustring = date.ptr;
                            goto Lstr;
                        }
                        else if (id == Id.TIME)
                        {
                            t.ustring = time.ptr;
                            goto Lstr;
                        }
                        else if (id == Id.VENDOR)
                        {
                            t.ustring = global.vendor.xarraydup.ptr;
                            goto Lstr;
                        }
                        else if (id == Id.TIMESTAMP)
                        {
                            t.ustring = timestamp.ptr;
                        Lstr:
                            t.value = TOK.string_;
                            t.postfix = 0;
                            t.len = cast(uint)strlen(t.ustring);
                        }
                        else if (id == Id.VERSIONX)
                        {
                            t.value = TOK.int64Literal;
                            t.unsvalue = global.versionNumber();
                        }
                        else if (id == Id.EOFX)
                        {
                            t.value = TOK.endOfFile;
                            // Advance scanner to end of file
                            while (!(*p == 0 || *p == 0x1A))
                                p++;
                        }
                    }
                    //printf("t.value = %d\n",t.value);
                    return;
                }
            case '/':
                p++;
                switch (*p)
                {
                case '=':
                    p++;
                    t.value = TOK.divAssign;
                    return;
                case '*':
                    p++;
                    startLoc = loc();
                    while (1)
                    {
                        while (1)
                        {
                            const c = *p;
                            switch (c)
                            {
                            case '/':
                                break;
                            case '\n':
                                endOfLine();
                                p++;
                                continue;
                            case '\r':
                                p++;
                                if (*p != '\n')
                                    endOfLine();
                                continue;
                            case 0:
                            case 0x1A:
                                error("unterminated /* */ comment");
                                p = end;
                                t.loc = loc();
                                t.value = TOK.endOfFile;
                                return;
                            default:
                                if (c & 0x80)
                                {
                                    const u = decodeUTF();
                                    if (u == PS || u == LS)
                                        endOfLine();
                                }
                                p++;
                                continue;
                            }
                            break;
                        }
                        p++;
                        if (p[-2] == '*' && p - 3 != t.ptr)
                            break;
                    }
                    if (commentToken)
                    {
                        t.loc = startLoc;
                        t.value = TOK.comment;
                        return;
                    }
                    else if (doDocComment && t.ptr[2] == '*' && p - 4 != t.ptr)
                    {
                        // if /** but not /**/
                        getDocComment(t, lastLine == startLoc.linnum, startLoc.linnum - lastDocLine > 1);
                        lastDocLine = scanloc.linnum;
                    }
                    continue;
                case '/': // do // style comments
                    startLoc = loc();
                    while (1)
                    {
                        const c = *++p;
                        switch (c)
                        {
                        case '\n':
                            break;
                        case '\r':
                            if (p[1] == '\n')
                                p++;
                            break;
                        case 0:
                        case 0x1A:
                            if (commentToken)
                            {
                                p = end;
                                t.loc = startLoc;
                                t.value = TOK.comment;
                                return;
                            }
                            if (doDocComment && t.ptr[2] == '/')
                            {
                                getDocComment(t, lastLine == startLoc.linnum, startLoc.linnum - lastDocLine > 1);
                                lastDocLine = scanloc.linnum;
                            }
                            p = end;
                            t.loc = loc();
                            t.value = TOK.endOfFile;
                            return;
                        default:
                            if (c & 0x80)
                            {
                                const u = decodeUTF();
                                if (u == PS || u == LS)
                                    break;
                            }
                            continue;
                        }
                        break;
                    }
                    if (commentToken)
                    {
                        p++;
                        endOfLine();
                        t.loc = startLoc;
                        t.value = TOK.comment;
                        return;
                    }
                    if (doDocComment && t.ptr[2] == '/')
                    {
                        getDocComment(t, lastLine == startLoc.linnum, startLoc.linnum - lastDocLine > 1);
                        lastDocLine = scanloc.linnum;
                    }
                    p++;
                    endOfLine();
                    continue;
                case '+':
                    {
                        int nest;
                        startLoc = loc();
                        p++;
                        nest = 1;
                        while (1)
                        {
                            char c = *p;
                            switch (c)
                            {
                            case '/':
                                p++;
                                if (*p == '+')
                                {
                                    p++;
                                    nest++;
                                }
                                continue;
                            case '+':
                                p++;
                                if (*p == '/')
                                {
                                    p++;
                                    if (--nest == 0)
                                        break;
                                }
                                continue;
                            case '\r':
                                p++;
                                if (*p != '\n')
                                    endOfLine();
                                continue;
                            case '\n':
                                endOfLine();
                                p++;
                                continue;
                            case 0:
                            case 0x1A:
                                error("unterminated /+ +/ comment");
                                p = end;
                                t.loc = loc();
                                t.value = TOK.endOfFile;
                                return;
                            default:
                                if (c & 0x80)
                                {
                                    uint u = decodeUTF();
                                    if (u == PS || u == LS)
                                        endOfLine();
                                }
                                p++;
                                continue;
                            }
                            break;
                        }
                        if (commentToken)
                        {
                            t.loc = startLoc;
                            t.value = TOK.comment;
                            return;
                        }
                        if (doDocComment && t.ptr[2] == '+' && p - 4 != t.ptr)
                        {
                            // if /++ but not /++/
                            getDocComment(t, lastLine == startLoc.linnum, startLoc.linnum - lastDocLine > 1);
                            lastDocLine = scanloc.linnum;
                        }
                        continue;
                    }
                default:
                    break;
                }
                t.value = TOK.div;
                return;
            case '.':
                p++;
                if (isdigit(*p))
                {
                    /* Note that we don't allow ._1 and ._ as being
                     * valid floating point numbers.
                     */
                    p--;
                    t.value = inreal(t);
                }
                else if (p[0] == '.')
                {
                    if (p[1] == '.')
                    {
                        p += 2;
                        t.value = TOK.dotDotDot;
                    }
                    else
                    {
                        p++;
                        t.value = TOK.slice;
                    }
                }
                else
                    t.value = TOK.dot;
                return;
            case '&':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.andAssign;
                }
                else if (*p == '&')
                {
                    p++;
                    t.value = TOK.andAnd;
                }
                else
                    t.value = TOK.and;
                return;
            case '|':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.orAssign;
                }
                else if (*p == '|')
                {
                    p++;
                    t.value = TOK.orOr;
                }
                else
                    t.value = TOK.or;
                return;
            case '-':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.minAssign;
                }
                else if (*p == '-')
                {
                    p++;
                    t.value = TOK.minusMinus;
                }
                else
                    t.value = TOK.min;
                return;
            case '+':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.addAssign;
                }
                else if (*p == '+')
                {
                    p++;
                    t.value = TOK.plusPlus;
                }
                else
                    t.value = TOK.add;
                return;
            case '<':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.lessOrEqual; // <=
                }
                else if (*p == '<')
                {
                    p++;
                    if (*p == '=')
                    {
                        p++;
                        t.value = TOK.leftShiftAssign; // <<=
                    }
                    else
                        t.value = TOK.leftShift; // <<
                }
                else
                    t.value = TOK.lessThan; // <
                return;
            case '>':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.greaterOrEqual; // >=
                }
                else if (*p == '>')
                {
                    p++;
                    if (*p == '=')
                    {
                        p++;
                        t.value = TOK.rightShiftAssign; // >>=
                    }
                    else if (*p == '>')
                    {
                        p++;
                        if (*p == '=')
                        {
                            p++;
                            t.value = TOK.unsignedRightShiftAssign; // >>>=
                        }
                        else
                            t.value = TOK.unsignedRightShift; // >>>
                    }
                    else
                        t.value = TOK.rightShift; // >>
                }
                else
                    t.value = TOK.greaterThan; // >
                return;
            case '!':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.notEqual; // !=
                }
                else
                    t.value = TOK.not; // !
                return;
            case '=':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.equal; // ==
                }
                else if (*p == '>')
                {
                    p++;
                    t.value = TOK.goesTo; // =>
                }
                else
                    t.value = TOK.assign; // =
                return;
            case '~':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.concatenateAssign; // ~=
                }
                else
                    t.value = TOK.tilde; // ~
                return;
            case '^':
                p++;
                if (*p == '^')
                {
                    p++;
                    if (*p == '=')
                    {
                        p++;
                        t.value = TOK.powAssign; // ^^=
                    }
                    else
                        t.value = TOK.pow; // ^^
                }
                else if (*p == '=')
                {
                    p++;
                    t.value = TOK.xorAssign; // ^=
                }
                else
                    t.value = TOK.xor; // ^
                return;
            case '(':
                p++;
                t.value = TOK.leftParentheses;
                return;
            case ')':
                p++;
                t.value = TOK.rightParentheses;
                return;
            case '[':
                p++;
                t.value = TOK.leftBracket;
                return;
            case ']':
                p++;
                t.value = TOK.rightBracket;
                return;
            case '{':
                p++;
                t.value = TOK.leftCurly;
                return;
            case '}':
                p++;
                t.value = TOK.rightCurly;
                return;
            case '?':
                p++;
                t.value = TOK.question;
                return;
            case ',':
                p++;
                t.value = TOK.comma;
                return;
            case ';':
                p++;
                t.value = TOK.semicolon;
                return;
            case ':':
                p++;
                t.value = TOK.colon;
                return;
            case '$':
                p++;
                t.value = TOK.dollar;
                return;
            case '@':
                p++;
                t.value = TOK.at;
                return;
            case '*':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.mulAssign;
                }
                else
                    t.value = TOK.mul;
                return;
            case '%':
                p++;
                if (*p == '=')
                {
                    p++;
                    t.value = TOK.modAssign;
                }
                else
                    t.value = TOK.mod;
                return;
            case '#':
                {
                    p++;
                    Token n;
                    scan(&n);
                    if (n.value == TOK.identifier)
                    {
                        if (n.ident == Id.line)
                        {
                            poundLine();
                            continue;
                        }
                        else
                        {
                            const locx = loc();
                            warning(locx, "C preprocessor directive `#%s` is not supported", n.ident.toChars());
                        }
                    }
                    else if (n.value == TOK.if_)
                    {
                        error("C preprocessor directive `#if` is not supported, use `version` or `static if`");
                    }
                    t.value = TOK.pound;
                    return;
                }
            default:
                {
                    dchar c = *p;
                    if (c & 0x80)
                    {
                        c = decodeUTF();
                        // Check for start of unicode identifier
                        if (isUniAlpha(c))
                            goto case_ident;
                        if (c == PS || c == LS)
                        {
                            endOfLine();
                            p++;
                            continue;
                        }
                    }
                    if (c < 0x80 && isprint(c))
                        error("character '%c' is not a valid token", c);
                    else
                        error("character 0x%02x is not a valid token", c);
                    p++;
                    continue;
                }
            }
        }
    }

    final Token* peek(Token* ct)
    {
        Token* t;
        if (ct.next)
            t = ct.next;
        else
        {
            t = allocateToken();
            scan(t);
            ct.next = t;
        }
        return t;
    }

    /*********************************
     * tk is on the opening (.
     * Look ahead and return token that is past the closing ).
     */
    final Token* peekPastParen(Token* tk)
    {
        //printf("peekPastParen()\n");
        int parens = 1;
        int curlynest = 0;
        while (1)
        {
            tk = peek(tk);
            //tk.print();
            switch (tk.value)
            {
            case TOK.leftParentheses:
                parens++;
                continue;
            case TOK.rightParentheses:
                --parens;
                if (parens)
                    continue;
                tk = peek(tk);
                break;
            case TOK.leftCurly:
                curlynest++;
                continue;
            case TOK.rightCurly:
                if (--curlynest >= 0)
                    continue;
                break;
            case TOK.semicolon:
                if (curlynest)
                    continue;
                break;
            case TOK.endOfFile:
                break;
            default:
                continue;
            }
            return tk;
        }
    }

    /*******************************************
     * Parse escape sequence.
     */
    private uint escapeSequence()
    {
        return Lexer.escapeSequence(token.loc, diagnosticReporter, p);
    }

    /**
    Parse the given string literal escape sequence into a single character.
    Params:
        loc = the location of the current token
        handler = the diagnostic reporter object
        sequence = pointer to string with escape sequence to parse. this is a reference
                   variable that is also used to return the position after the sequence
    Returns:
        the escaped sequence as a single character
    */
    private static dchar escapeSequence(const ref Loc loc, DiagnosticReporter handler, ref const(char)* sequence)
    in
    {
        assert(handler !is null);
    }
    body
    {
        const(char)* p = sequence; // cache sequence reference on stack
        scope(exit) sequence = p;

        uint c = *p;
        int ndigits;
        switch (c)
        {
        case '\'':
        case '"':
        case '?':
        case '\\':
        Lconsume:
            p++;
            break;
        case 'a':
            c = 7;
            goto Lconsume;
        case 'b':
            c = 8;
            goto Lconsume;
        case 'f':
            c = 12;
            goto Lconsume;
        case 'n':
            c = 10;
            goto Lconsume;
        case 'r':
            c = 13;
            goto Lconsume;
        case 't':
            c = 9;
            goto Lconsume;
        case 'v':
            c = 11;
            goto Lconsume;
        case 'u':
            ndigits = 4;
            goto Lhex;
        case 'U':
            ndigits = 8;
            goto Lhex;
        case 'x':
            ndigits = 2;
        Lhex:
            p++;
            c = *p;
            if (ishex(cast(char)c))
            {
                uint v = 0;
                int n = 0;
                while (1)
                {
                    if (isdigit(cast(char)c))
                        c -= '0';
                    else if (islower(c))
                        c -= 'a' - 10;
                    else
                        c -= 'A' - 10;
                    v = v * 16 + c;
                    c = *++p;
                    if (++n == ndigits)
                        break;
                    if (!ishex(cast(char)c))
                    {
                        handler.error(loc, "escape hex sequence has %d hex digits instead of %d", n, ndigits);
                        break;
                    }
                }
                if (ndigits != 2 && !utf_isValidDchar(v))
                {
                    handler.error(loc, "invalid UTF character \\U%08x", v);
                    v = '?'; // recover with valid UTF character
                }
                c = v;
            }
            else
            {
                handler.error(loc, "undefined escape hex sequence \\%c%c", sequence[0], c);
                p++;
            }
            break;
        case '&':
            // named character entity
            for (const idstart = ++p; 1; p++)
            {
                switch (*p)
                {
                case ';':
                    c = HtmlNamedEntity(idstart, p - idstart);
                    if (c == ~0)
                    {
                        handler.error(loc, "unnamed character entity &%.*s;", cast(int)(p - idstart), idstart);
                        c = '?';
                    }
                    p++;
                    break;
                default:
                    if (isalpha(*p) || (p != idstart && isdigit(*p)))
                        continue;
                    handler.error(loc, "unterminated named entity &%.*s;", cast(int)(p - idstart + 1), idstart);
                    c = '?';
                    break;
                }
                break;
            }
            break;
        case 0:
        case 0x1A:
            // end of file
            c = '\\';
            break;
        default:
            if (isoctal(cast(char)c))
            {
                uint v = 0;
                int n = 0;
                do
                {
                    v = v * 8 + (c - '0');
                    c = *++p;
                }
                while (++n < 3 && isoctal(cast(char)c));
                c = v;
                if (c > 0xFF)
                    handler.error(loc, "escape octal sequence \\%03o is larger than \\377", c);
            }
            else
            {
                handler.error(loc, "undefined escape sequence \\%c", c);
                p++;
            }
            break;
        }
        return c;
    }

    /**
    Lex a wysiwyg string. `p` must be pointing to the first character before the
    contents of the string literal. The character pointed to by `p` will be used as
    the terminating character (i.e. backtick or double-quote).
    Params:
        result = pointer to the token that accepts the result
    */
    private void wysiwygStringConstant(Token* result)
    {
        result.value = TOK.string_;
        Loc start = loc();
        auto terminator = p[0];
        p++;
        stringbuffer.reset();
        while (1)
        {
            dchar c = p[0];
            p++;
            switch (c)
            {
            case '\n':
                endOfLine();
                break;
            case '\r':
                if (p[0] == '\n')
                    continue; // ignore
                c = '\n'; // treat EndOfLine as \n character
                endOfLine();
                break;
            case 0:
            case 0x1A:
                error("unterminated string constant starting at %s", start.toChars());
                result.setString();
                // rewind `p` so it points to the EOF character
                p--;
                return;
            default:
                if (c == terminator)
                {
                    result.setString(stringbuffer);
                    stringPostfix(result);
                    return;
                }
                else if (c & 0x80)
                {
                    p--;
                    const u = decodeUTF();
                    p++;
                    if (u == PS || u == LS)
                        endOfLine();
                    stringbuffer.writeUTF8(u);
                    continue;
                }
                break;
            }
            stringbuffer.writeByte(c);
        }
    }

    /**************************************
     * Lex hex strings:
     *      x"0A ae 34FE BD"
     */
    private TOK hexStringConstant(Token* t)
    {
        Loc start = loc();
        uint n = 0;
        uint v = ~0; // dead assignment, needed to suppress warning
        p++;
        stringbuffer.reset();
        while (1)
        {
            dchar c = *p++;
            switch (c)
            {
            case ' ':
            case '\t':
            case '\v':
            case '\f':
                continue; // skip white space
            case '\r':
                if (*p == '\n')
                    continue; // ignore '\r' if followed by '\n'
                // Treat isolated '\r' as if it were a '\n'
                goto case '\n';
            case '\n':
                endOfLine();
                continue;
            case 0:
            case 0x1A:
                error("unterminated string constant starting at %s", start.toChars());
                t.setString();
                // decrement `p`, because it needs to point to the next token (the 0 or 0x1A character is the TOK.endOfFile token).
                p--;
                return TOK.hexadecimalString;
            case '"':
                if (n & 1)
                {
                    error("odd number (%d) of hex characters in hex string", n);
                    stringbuffer.writeByte(v);
                }
                t.setString(stringbuffer);
                stringPostfix(t);
                return TOK.hexadecimalString;
            default:
                if (c >= '0' && c <= '9')
                    c -= '0';
                else if (c >= 'a' && c <= 'f')
                    c -= 'a' - 10;
                else if (c >= 'A' && c <= 'F')
                    c -= 'A' - 10;
                else if (c & 0x80)
                {
                    p--;
                    const u = decodeUTF();
                    p++;
                    if (u == PS || u == LS)
                        endOfLine();
                    else
                        error("non-hex character \\u%04x in hex string", u);
                }
                else
                    error("non-hex character '%c' in hex string", c);
                if (n & 1)
                {
                    v = (v << 4) | c;
                    stringbuffer.writeByte(v);
                }
                else
                    v = c;
                n++;
                break;
            }
        }
        assert(0); // see bug 15731
    }

    /**
    Lex a delimited string. Some examples of delimited strings are:
    ---
    q"(foo(xxx))"      // "foo(xxx)"
    q"[foo$(LPAREN)]"  // "foo$(LPAREN)"
    q"/foo]/"          // "foo]"
    q"HERE
    foo
    HERE"              // "foo\n"
    ---
    It is assumed that `p` points to the opening double-quote '"'.
    Params:
        result = pointer to the token that accepts the result
    */
    private void delimitedStringConstant(Token* result)
    {
        result.value = TOK.string_;
        Loc start = loc();
        dchar delimleft = 0;
        dchar delimright = 0;
        uint nest = 1;
        uint nestcount = ~0; // dead assignment, needed to suppress warning
        Identifier hereid = null;
        uint blankrol = 0;
        uint startline = 0;
        p++;
        stringbuffer.reset();
        while (1)
        {
            dchar c = *p++;
            //printf("c = '%c'\n", c);
            switch (c)
            {
            case '\n':
            Lnextline:
                endOfLine();
                startline = 1;
                if (blankrol)
                {
                    blankrol = 0;
                    continue;
                }
                if (hereid)
                {
                    stringbuffer.writeUTF8(c);
                    continue;
                }
                break;
            case '\r':
                if (*p == '\n')
                    continue; // ignore
                c = '\n'; // treat EndOfLine as \n character
                goto Lnextline;
            case 0:
            case 0x1A:
                error("unterminated delimited string constant starting at %s", start.toChars());
                result.setString();
                // decrement `p`, because it needs to point to the next token (the 0 or 0x1A character is the TOK.endOfFile token).
                p--;
                return;
            default:
                if (c & 0x80)
                {
                    p--;
                    c = decodeUTF();
                    p++;
                    if (c == PS || c == LS)
                        goto Lnextline;
                }
                break;
            }
            if (delimleft == 0)
            {
                delimleft = c;
                nest = 1;
                nestcount = 1;
                if (c == '(')
                    delimright = ')';
                else if (c == '{')
                    delimright = '}';
                else if (c == '[')
                    delimright = ']';
                else if (c == '<')
                    delimright = '>';
                else if (isalpha(c) || c == '_' || (c >= 0x80 && isUniAlpha(c)))
                {
                    // Start of identifier; must be a heredoc
                    Token tok;
                    p--;
                    scan(&tok); // read in heredoc identifier
                    if (tok.value != TOK.identifier)
                    {
                        error("identifier expected for heredoc, not %s", tok.toChars());
                        delimright = c;
                    }
                    else
                    {
                        hereid = tok.ident;
                        //printf("hereid = '%s'\n", hereid.toChars());
                        blankrol = 1;
                    }
                    nest = 0;
                }
                else
                {
                    delimright = c;
                    nest = 0;
                    if (isspace(c))
                        error("delimiter cannot be whitespace");
                }
            }
            else
            {
                if (blankrol)
                {
                    error("heredoc rest of line should be blank");
                    blankrol = 0;
                    continue;
                }
                if (nest == 1)
                {
                    if (c == delimleft)
                        nestcount++;
                    else if (c == delimright)
                    {
                        nestcount--;
                        if (nestcount == 0)
                            goto Ldone;
                    }
                }
                else if (c == delimright)
                    goto Ldone;
                if (startline && (isalpha(c) || c == '_' || (c >= 0x80 && isUniAlpha(c))) && hereid)
                {
                    Token tok;
                    auto psave = p;
                    p--;
                    scan(&tok); // read in possible heredoc identifier
                    //printf("endid = '%s'\n", tok.ident.toChars());
                    if (tok.value == TOK.identifier && tok.ident is hereid)
                    {
                        /* should check that rest of line is blank
                         */
                        goto Ldone;
                    }
                    p = psave;
                }
                stringbuffer.writeUTF8(c);
                startline = 0;
            }
        }
    Ldone:
        if (*p == '"')
            p++;
        else if (hereid)
            error("delimited string must end in %s\"", hereid.toChars());
        else
            error("delimited string must end in %c\"", delimright);
        result.setString(stringbuffer);
        stringPostfix(result);
    }

    /**
    Lex a token string. Some examples of token strings are:
    ---
    q{ foo(xxx) }    // " foo(xxx) "
    q{foo$(LPAREN)}  // "foo$(LPAREN)"
    q{{foo}"}"}      // "{foo}"}""
    ---
    It is assumed that `p` points to the opening curly-brace '{'.
    Params:
        result = pointer to the token that accepts the result
    */
    private void tokenStringConstant(Token* result)
    {
        result.value = TOK.string_;

        uint nest = 1;
        const start = loc();
        const pstart = ++p;
        inTokenStringConstant++;
        scope(exit) inTokenStringConstant--;
        while (1)
        {
            Token tok;
            scan(&tok);
            switch (tok.value)
            {
            case TOK.leftCurly:
                nest++;
                continue;
            case TOK.rightCurly:
                if (--nest == 0)
                {
                    result.setString(pstart, p - 1 - pstart);
                    stringPostfix(result);
                    return;
                }
                continue;
            case TOK.endOfFile:
                error("unterminated token string constant starting at %s", start.toChars());
                result.setString();
                return;
            default:
                continue;
            }
        }
    }

    /**
    Scan a double-quoted string while building the processed string value by
    handling escape sequences. The result is returned in the given `t` token.
    This function assumes that `p` currently points to the opening double-quote
    of the string.
    Params:
        t = the token to set the resulting string to
    */
    private void escapeStringConstant(Token* t)
    {
        t.value = TOK.string_;

        const start = loc();
        p++;
        stringbuffer.reset();
        while (1)
        {
            dchar c = *p++;
            switch (c)
            {
            case '\\':
                switch (*p)
                {
                case 'u':
                case 'U':
                case '&':
                    c = escapeSequence();
                    stringbuffer.writeUTF8(c);
                    continue;
                default:
                    c = escapeSequence();
                    break;
                }
                break;
            case '\n':
                endOfLine();
                break;
            case '\r':
                if (*p == '\n')
                    continue; // ignore
                c = '\n'; // treat EndOfLine as \n character
                endOfLine();
                break;
            case '"':
                t.setString(stringbuffer);
                stringPostfix(t);
                return;
            case 0:
            case 0x1A:
                // decrement `p`, because it needs to point to the next token (the 0 or 0x1A character is the TOK.endOfFile token).
                p--;
                error("unterminated string constant starting at %s", start.toChars());
                t.setString();
                return;
            default:
                if (c & 0x80)
                {
                    p--;
                    c = decodeUTF();
                    if (c == LS || c == PS)
                    {
                        c = '\n';
                        endOfLine();
                    }
                    p++;
                    stringbuffer.writeUTF8(c);
                    continue;
                }
                break;
            }
            stringbuffer.writeByte(c);
        }
    }

    /**************************************
     */
    private TOK charConstant(Token* t)
    {
        TOK tk = TOK.charLiteral;
        //printf("Lexer::charConstant\n");
        p++;
        dchar c = *p++;
        switch (c)
        {
        case '\\':
            switch (*p)
            {
            case 'u':
                t.unsvalue = escapeSequence();
                tk = TOK.wcharLiteral;
                break;
            case 'U':
            case '&':
                t.unsvalue = escapeSequence();
                tk = TOK.dcharLiteral;
                break;
            default:
                t.unsvalue = escapeSequence();
                break;
            }
            break;
        case '\n':
        L1:
            endOfLine();
            goto case;
        case '\r':
            goto case '\'';
        case 0:
        case 0x1A:
            // decrement `p`, because it needs to point to the next token (the 0 or 0x1A character is the TOK.endOfFile token).
            p--;
            goto case;
        case '\'':
            error("unterminated character constant");
            t.unsvalue = '?';
            return tk;
        default:
            if (c & 0x80)
            {
                p--;
                c = decodeUTF();
                p++;
                if (c == LS || c == PS)
                    goto L1;
                if (c < 0xD800 || (c >= 0xE000 && c < 0xFFFE))
                    tk = TOK.wcharLiteral;
                else
                    tk = TOK.dcharLiteral;
            }
            t.unsvalue = c;
            break;
        }
        if (*p != '\'')
        {
            error("unterminated character constant");
            t.unsvalue = '?';
            return tk;
        }
        p++;
        return tk;
    }

    /***************************************
     * Get postfix of string literal.
     */
    private void stringPostfix(Token* t) pure @nogc
    {
        switch (*p)
        {
        case 'c':
        case 'w':
        case 'd':
            t.postfix = *p;
            p++;
            break;
        default:
            t.postfix = 0;
            break;
        }
    }

    /**************************************
     * Read in a number.
     * If it's an integer, store it in tok.TKutok.Vlong.
     *      integers can be decimal, octal or hex
     *      Handle the suffixes U, UL, LU, L, etc.
     * If it's double, store it in tok.TKutok.Vdouble.
     * Returns:
     *      TKnum
     *      TKdouble,...
     */
    private TOK number(Token* t)
    {
        int base = 10;
        const start = p;
        uinteger_t n = 0; // unsigned >=64 bit integer type
        int d;
        bool err = false;
        bool overflow = false;
        bool anyBinaryDigitsNoSingleUS = false;
        bool anyHexDigitsNoSingleUS = false;
        dchar c = *p;
        if (c == '0')
        {
            ++p;
            c = *p;
            switch (c)
            {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                base = 8;
                break;
            case 'x':
            case 'X':
                ++p;
                base = 16;
                break;
            case 'b':
            case 'B':
                ++p;
                base = 2;
                break;
            case '.':
                if (p[1] == '.')
                    goto Ldone; // if ".."
                if (isalpha(p[1]) || p[1] == '_' || p[1] & 0x80)
                    goto Ldone; // if ".identifier" or ".unicode"
                goto Lreal; // '.' is part of current token
            case 'i':
            case 'f':
            case 'F':
                goto Lreal;
            case '_':
                ++p;
                base = 8;
                break;
            case 'L':
                if (p[1] == 'i')
                    goto Lreal;
                break;
            default:
                break;
            }
        }
        while (1)
        {
            c = *p;
            switch (c)
            {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                ++p;
                d = c - '0';
                break;
            case 'a':
            case 'b':
            case 'c':
            case 'd':
            case 'e':
            case 'f':
            case 'A':
            case 'B':
            case 'C':
            case 'D':
            case 'E':
            case 'F':
                ++p;
                if (base != 16)
                {
                    if (c == 'e' || c == 'E' || c == 'f' || c == 'F')
                        goto Lreal;
                }
                if (c >= 'a')
                    d = c + 10 - 'a';
                else
                    d = c + 10 - 'A';
                break;
            case 'L':
                if (p[1] == 'i')
                    goto Lreal;
                goto Ldone;
            case '.':
                if (p[1] == '.')
                    goto Ldone; // if ".."
                if (base == 10 && (isalpha(p[1]) || p[1] == '_' || p[1] & 0x80))
                    goto Ldone; // if ".identifier" or ".unicode"
                if (base == 16 && (!ishex(p[1]) || p[1] == '_' || p[1] & 0x80))
                    goto Ldone; // if ".identifier" or ".unicode"
                if (base == 2)
                    goto Ldone; // if ".identifier" or ".unicode"
                goto Lreal; // otherwise as part of a floating point literal
            case 'p':
            case 'P':
            case 'i':
            Lreal:
                p = start;
                return inreal(t);
            case '_':
                ++p;
                continue;
            default:
                goto Ldone;
            }
            // got a digit here, set any necessary flags, check for errors
            anyHexDigitsNoSingleUS = true;
            anyBinaryDigitsNoSingleUS = true;
            if (!err && d >= base)
            {
                error("%s digit expected, not `%c`", base == 2 ? "binary".ptr :
                                                     base == 8 ? "octal".ptr :
                                                     "decimal".ptr, c);
                err = true;
            }
            // Avoid expensive overflow check if we aren't at risk of overflow
            if (n <= 0x0FFF_FFFF_FFFF_FFFFUL)
                n = n * base + d;
            else
            {
                import core.checkedint : mulu, addu;

                n = mulu(n, base, overflow);
                n = addu(n, d, overflow);
            }
        }
    Ldone:
        if (overflow && !err)
        {
            error("integer overflow");
            err = true;
        }
        if ((base == 2 && !anyBinaryDigitsNoSingleUS) ||
            (base == 16 && !anyHexDigitsNoSingleUS))
            error("`%.*s` isn't a valid integer literal, use `%.*s0` instead", cast(int)(p - start), start, 2, start);
        enum FLAGS : int
        {
            none = 0,
            decimal = 1, // decimal
            unsigned = 2, // u or U suffix
            long_ = 4, // L suffix
        }

        FLAGS flags = (base == 10) ? FLAGS.decimal : FLAGS.none;
        // Parse trailing 'u', 'U', 'l' or 'L' in any combination
        const psuffix = p;
        while (1)
        {
            FLAGS f;
            switch (*p)
            {
            case 'U':
            case 'u':
                f = FLAGS.unsigned;
                goto L1;
            case 'l':
                f = FLAGS.long_;
                error("lower case integer suffix 'l' is not allowed. Please use 'L' instead");
                goto L1;
            case 'L':
                f = FLAGS.long_;
            L1:
                p++;
                if ((flags & f) && !err)
                {
                    error("unrecognized token");
                    err = true;
                }
                flags = cast(FLAGS)(flags | f);
                continue;
            default:
                break;
            }
            break;
        }
        if (base == 8 && n >= 8)
        {
            if (err)
                // can't translate invalid octal value, just show a generic message
                error("octal literals larger than 7 are no longer supported");
            else
                error("octal literals `0%llo%.*s` are no longer supported, use `std.conv.octal!%llo%.*s` instead",
                    n, cast(int)(p - psuffix), psuffix, n, cast(int)(p - psuffix), psuffix);
        }
        TOK result;
        switch (flags)
        {
        case FLAGS.none:
            /* Octal or Hexadecimal constant.
             * First that fits: int, uint, long, ulong
             */
            if (n & 0x8000000000000000L)
                result = TOK.uns64Literal;
            else if (n & 0xFFFFFFFF00000000L)
                result = TOK.int64Literal;
            else if (n & 0x80000000)
                result = TOK.uns32Literal;
            else
                result = TOK.int32Literal;
            break;
        case FLAGS.decimal:
            /* First that fits: int, long, long long
             */
            if (n & 0x8000000000000000L)
            {
                if (!err)
                {
                    error("signed integer overflow");
                    err = true;
                }
                result = TOK.uns64Literal;
            }
            else if (n & 0xFFFFFFFF80000000L)
                result = TOK.int64Literal;
            else
                result = TOK.int32Literal;
            break;
        case FLAGS.unsigned:
        case FLAGS.decimal | FLAGS.unsigned:
            /* First that fits: uint, ulong
             */
            if (n & 0xFFFFFFFF00000000L)
                result = TOK.uns64Literal;
            else
                result = TOK.uns32Literal;
            break;
        case FLAGS.decimal | FLAGS.long_:
            if (n & 0x8000000000000000L)
            {
                if (!err)
                {
                    error("signed integer overflow");
                    err = true;
                }
                result = TOK.uns64Literal;
            }
            else
                result = TOK.int64Literal;
            break;
        case FLAGS.long_:
            if (n & 0x8000000000000000L)
                result = TOK.uns64Literal;
            else
                result = TOK.int64Literal;
            break;
        case FLAGS.unsigned | FLAGS.long_:
        case FLAGS.decimal | FLAGS.unsigned | FLAGS.long_:
            result = TOK.uns64Literal;
            break;
        default:
            debug
            {
                printf("%x\n", flags);
            }
            assert(0);
        }
        t.unsvalue = n;
        return result;
    }

    /**************************************
     * Read in characters, converting them to real.
     * Bugs:
     *      Exponent overflow not detected.
     *      Too much requested precision is not detected.
     */
    private TOK inreal(Token* t)
    {
        //printf("Lexer::inreal()\n");
        debug
        {
            assert(*p == '.' || isdigit(*p));
        }
        bool isWellformedString = true;
        stringbuffer.reset();
        auto pstart = p;
        bool hex = false;
        dchar c = *p++;
        // Leading '0x'
        if (c == '0')
        {
            c = *p++;
            if (c == 'x' || c == 'X')
            {
                hex = true;
                c = *p++;
            }
        }
        // Digits to left of '.'
        while (1)
        {
            if (c == '.')
            {
                c = *p++;
                break;
            }
            if (isdigit(c) || (hex && isxdigit(c)) || c == '_')
            {
                c = *p++;
                continue;
            }
            break;
        }
        // Digits to right of '.'
        while (1)
        {
            if (isdigit(c) || (hex && isxdigit(c)) || c == '_')
            {
                c = *p++;
                continue;
            }
            break;
        }
        if (c == 'e' || c == 'E' || (hex && (c == 'p' || c == 'P')))
        {
            c = *p++;
            if (c == '-' || c == '+')
            {
                c = *p++;
            }
            bool anyexp = false;
            while (1)
            {
                if (isdigit(c))
                {
                    anyexp = true;
                    c = *p++;
                    continue;
                }
                if (c == '_')
                {
                    c = *p++;
                    continue;
                }
                if (!anyexp)
                {
                    error("missing exponent");
                    isWellformedString = false;
                }
                break;
            }
        }
        else if (hex)
        {
            error("exponent required for hex float");
            isWellformedString = false;
        }
        --p;
        while (pstart < p)
        {
            if (*pstart != '_')
                stringbuffer.writeByte(*pstart);
            ++pstart;
        }
        stringbuffer.writeByte(0);
        auto sbufptr = cast(const(char)*)stringbuffer[].ptr;
        TOK result;
        bool isOutOfRange = false;
        t.floatvalue = (isWellformedString ? CTFloat.parse(sbufptr, &isOutOfRange) : CTFloat.zero);
        switch (*p)
        {
        case 'F':
        case 'f':
            if (isWellformedString && !isOutOfRange)
                isOutOfRange = Port.isFloat32LiteralOutOfRange(sbufptr);
            result = TOK.float32Literal;
            p++;
            break;
        default:
            if (isWellformedString && !isOutOfRange)
                isOutOfRange = Port.isFloat64LiteralOutOfRange(sbufptr);
            result = TOK.float64Literal;
            break;
        case 'l':
            error("use 'L' suffix instead of 'l'");
            goto case 'L';
        case 'L':
            result = TOK.float80Literal;
            p++;
            break;
        }
        if (*p == 'i' || *p == 'I')
        {
            if (*p == 'I')
                error("use 'i' suffix instead of 'I'");
            p++;
            switch (result)
            {
            case TOK.float32Literal:
                result = TOK.imaginary32Literal;
                break;
            case TOK.float64Literal:
                result = TOK.imaginary64Literal;
                break;
            case TOK.float80Literal:
                result = TOK.imaginary80Literal;
                break;
            default:
                break;
            }
        }
        const isLong = (result == TOK.float80Literal || result == TOK.imaginary80Literal);
        if (isOutOfRange && !isLong)
        {
            const char* suffix = (result == TOK.float32Literal || result == TOK.imaginary32Literal) ? "f" : "";
            error(scanloc, "number `%s%s` is not representable", sbufptr, suffix);
        }
        debug
        {
            switch (result)
            {
            case TOK.float32Literal:
            case TOK.float64Literal:
            case TOK.float80Literal:
            case TOK.imaginary32Literal:
            case TOK.imaginary64Literal:
            case TOK.imaginary80Literal:
                break;
            default:
                assert(0);
            }
        }
        return result;
    }

    final Loc loc() pure @nogc
    {
        scanloc.charnum = cast(uint)(1 + p - line);
        return scanloc;
    }

    final void error(const(char)* format, ...)
    {
        va_list args;
        va_start(args, format);
        diagnosticReporter.error(token.loc, format, args);
        va_end(args);
    }

    final void error(const ref Loc loc, const(char)* format, ...)
    {
        va_list args;
        va_start(args, format);
        diagnosticReporter.error(loc, format, args);
        va_end(args);
    }

    final void errorSupplemental(const ref Loc loc, const(char)* format, ...)
    {
        va_list args;
        va_start(args, format);
        diagnosticReporter.errorSupplemental(loc, format, args);
        va_end(args);
    }

    final void warning(const ref Loc loc, const(char)* format, ...)
    {
        va_list args;
        va_start(args, format);
        diagnosticReporter.warning(loc, format, args);
        va_end(args);
    }

    final void warningSupplemental(const ref Loc loc, const(char)* format, ...)
    {
        va_list args;
        va_start(args, format);
        diagnosticReporter.warningSupplemental(loc, format, args);
        va_end(args);
    }

    final void deprecation(const(char)* format, ...)
    {
        va_list args;
        va_start(args, format);
        diagnosticReporter.deprecation(token.loc, format, args);
        va_end(args);
    }

    final void deprecation(const ref Loc loc, const(char)* format, ...)
    {
        va_list args;
        va_start(args, format);
        diagnosticReporter.deprecation(loc, format, args);
        va_end(args);
    }

    final void deprecationSupplemental(const ref Loc loc, const(char)* format, ...)
    {
        va_list args;
        va_start(args, format);
        diagnosticReporter.deprecationSupplemental(loc, format, args);
        va_end(args);
    }

    /*********************************************
     * parse:
     *      #line linnum [filespec]
     * also allow __LINE__ for linnum, and __FILE__ for filespec
     */
    private void poundLine()
    {
        auto linnum = this.scanloc.linnum;
        const(char)* filespec = null;
        const loc = this.loc();
        Token tok;
        scan(&tok);
        if (tok.value == TOK.int32Literal || tok.value == TOK.int64Literal)
        {
            const lin = cast(int)(tok.unsvalue - 1);
            if (lin != tok.unsvalue - 1)
                error("line number `%lld` out of range", cast(ulong)tok.unsvalue);
            else
                linnum = lin;
        }
        else if (tok.value == TOK.line)
        {
        }
        else
            goto Lerr;
        while (1)
        {
            switch (*p)
            {
            case 0:
            case 0x1A:
            case '\n':
            Lnewline:
                if (!inTokenStringConstant)
                {
                    this.scanloc.linnum = linnum;
                    if (filespec)
                        this.scanloc.filename = filespec;
                }
                return;
            case '\r':
                p++;
                if (*p != '\n')
                {
                    p--;
                    goto Lnewline;
                }
                continue;
            case ' ':
            case '\t':
            case '\v':
            case '\f':
                p++;
                continue; // skip white space
            case '_':
                if (memcmp(p, "__FILE__".ptr, 8) == 0)
                {
                    p += 8;
                    filespec = mem.xstrdup(scanloc.filename);
                    continue;
                }
                goto Lerr;
            case '"':
                if (filespec)
                    goto Lerr;
                stringbuffer.reset();
                p++;
                while (1)
                {
                    uint c;
                    c = *p;
                    switch (c)
                    {
                    case '\n':
                    case '\r':
                    case 0:
                    case 0x1A:
                        goto Lerr;
                    case '"':
                        stringbuffer.writeByte(0);
                        filespec = mem.xstrdup(cast(const(char)*)stringbuffer[].ptr);
                        p++;
                        break;
                    default:
                        if (c & 0x80)
                        {
                            uint u = decodeUTF();
                            if (u == PS || u == LS)
                                goto Lerr;
                        }
                        stringbuffer.writeByte(c);
                        p++;
                        continue;
                    }
                    break;
                }
                continue;
            default:
                if (*p & 0x80)
                {
                    uint u = decodeUTF();
                    if (u == PS || u == LS)
                        goto Lnewline;
                }
                goto Lerr;
            }
        }
    Lerr:
        error(loc, "#line integer [\"filespec\"]\\n expected");
    }

    /********************************************
     * Decode UTF character.
     * Issue error messages for invalid sequences.
     * Return decoded character, advance p to last character in UTF sequence.
     */
    private uint decodeUTF()
    {
        const s = p;
        assert(*s & 0x80);
        // Check length of remaining string up to 4 UTF-8 characters
        size_t len;
        for (len = 1; len < 4 && s[len]; len++)
        {
        }
        size_t idx = 0;
        dchar u;
        const msg = utf_decodeChar(s, len, idx, u);
        p += idx - 1;
        if (msg)
        {
            error("%s", msg);
        }
        return u;
    }

    /***************************************************
     * Parse doc comment embedded between t.ptr and p.
     * Remove trailing blanks and tabs from lines.
     * Replace all newlines with \n.
     * Remove leading comment character from each line.
     * Decide if it's a lineComment or a blockComment.
     * Append to previous one for this token.
     *
     * If newParagraph is true, an extra newline will be
     * added between adjoining doc comments.
     */
    private void getDocComment(Token* t, uint lineComment, bool newParagraph) pure
    {
        /* ct tells us which kind of comment it is: '/', '*', or '+'
         */
        const ct = t.ptr[2];
        /* Start of comment text skips over / * *, / + +, or / / /
         */
        const(char)* q = t.ptr + 3; // start of comment text
        const(char)* qend = p;
        if (ct == '*' || ct == '+')
            qend -= 2;
        /* Scan over initial row of ****'s or ++++'s or ////'s
         */
        for (; q < qend; q++)
        {
            if (*q != ct)
                break;
        }
        /* Remove leading spaces until start of the comment
         */
        int linestart = 0;
        if (ct == '/')
        {
            while (q < qend && (*q == ' ' || *q == '\t'))
                ++q;
        }
        else if (q < qend)
        {
            if (*q == '\r')
            {
                ++q;
                if (q < qend && *q == '\n')
                    ++q;
                linestart = 1;
            }
            else if (*q == '\n')
            {
                ++q;
                linestart = 1;
            }
        }
        /* Remove trailing row of ****'s or ++++'s
         */
        if (ct != '/')
        {
            for (; q < qend; qend--)
            {
                if (qend[-1] != ct)
                    break;
            }
        }
        /* Comment is now [q .. qend].
         * Canonicalize it into buf[].
         */
        OutBuffer buf;

        void trimTrailingWhitespace() nothrow
        {
            const s = buf[];
            auto len = s.length;
            while (len && (s[len - 1] == ' ' || s[len - 1] == '\t'))
                --len;
            buf.setsize(len);
        }

        for (; q < qend; q++)
        {
            char c = *q;
            switch (c)
            {
            case '*':
            case '+':
                if (linestart && c == ct)
                {
                    linestart = 0;
                    /* Trim preceding whitespace up to preceding \n
                     */
                    trimTrailingWhitespace();
                    continue;
                }
                break;
            case ' ':
            case '\t':
                break;
            case '\r':
                if (q[1] == '\n')
                    continue; // skip the \r
                goto Lnewline;
            default:
                if (c == 226)
                {
                    // If LS or PS
                    if (q[1] == 128 && (q[2] == 168 || q[2] == 169))
                    {
                        q += 2;
                        goto Lnewline;
                    }
                }
                linestart = 0;
                break;
            Lnewline:
                c = '\n'; // replace all newlines with \n
                goto case;
            case '\n':
                linestart = 1;
                /* Trim trailing whitespace
                 */
                trimTrailingWhitespace();
                break;
            }
            buf.writeByte(c);
        }
        /* Trim trailing whitespace (if the last line does not have newline)
         */
        trimTrailingWhitespace();

        // Always end with a newline
        const s = buf[];
        if (s.length == 0 || s[$ - 1] != '\n')
            buf.writeByte('\n');

        // It's a line comment if the start of the doc comment comes
        // after other non-whitespace on the same line.
        auto dc = (lineComment && anyToken) ? &t.lineComment : &t.blockComment;
        // Combine with previous doc comment, if any
        if (*dc)
            *dc = combineComments(*dc, buf[], newParagraph).toDString();
        else
            *dc = buf.extractSlice(true);
    }

    /********************************************
     * Combine two document comments into one,
     * separated by an extra newline if newParagraph is true.
     */
    static const(char)* combineComments(const(char)[] c1, const(char)[] c2, bool newParagraph) pure
    {
        //printf("Lexer::combineComments('%s', '%s', '%i')\n", c1, c2, newParagraph);
        const(int) newParagraphSize = newParagraph ? 1 : 0; // Size of the combining '\n'
        if (!c1)
            return c2.ptr;
        if (!c2)
            return c1.ptr;

        int insertNewLine = 0;
        if (c1.length && c1[$ - 1] != '\n')
            insertNewLine = 1;
        const retSize = c1.length + insertNewLine + newParagraphSize + c2.length;
        auto p = cast(char*)mem.xmalloc_noscan(retSize + 1);
        p[0 .. c1.length] = c1[];
        if (insertNewLine)
            p[c1.length] = '\n';
        if (newParagraph)
            p[c1.length + insertNewLine] = '\n';
        p[retSize - c2.length .. retSize] = c2[];
        p[retSize] = 0;
        return p;
    }

private:
    void endOfLine() pure @nogc @safe
    {
        scanloc.linnum++;
        line = p;
    }
}

unittest
{
    static final class AssertDiagnosticReporter : DiagnosticReporter
    {
        override int errorCount() { assert(0); }
        override int warningCount() { assert(0); }
        override int deprecationCount() { assert(0); }
        override void error(const ref Loc, const(char)*, va_list) { assert(0); }
        override void errorSupplemental(const ref Loc, const(char)*, va_list) { assert(0); }
        override void warning(const ref Loc, const(char)*, va_list) { assert(0); }
        override void warningSupplemental(const ref Loc, const(char)*, va_list) { assert(0); }
        override void deprecation(const ref Loc, const(char)*, va_list) { assert(0); }
        override void deprecationSupplemental(const ref Loc, const(char)*, va_list) { assert(0); }
    }
    static void test(T)(string sequence, T expected)
    {
        scope assertOnError = new AssertDiagnosticReporter();
        auto p = cast(const(char)*)sequence.ptr;
        assert(expected == Lexer.escapeSequence(Loc.initial, assertOnError, p));
        assert(p == sequence.ptr + sequence.length);
    }

    test(`'`, '\'');
    test(`"`, '"');
    test(`?`, '?');
    test(`\`, '\\');
    test(`0`, '\0');
    test(`a`, '\a');
    test(`b`, '\b');
    test(`f`, '\f');
    test(`n`, '\n');
    test(`r`, '\r');
    test(`t`, '\t');
    test(`v`, '\v');

    test(`x00`, 0x00);
    test(`xff`, 0xff);
    test(`xFF`, 0xff);
    test(`xa7`, 0xa7);
    test(`x3c`, 0x3c);
    test(`xe2`, 0xe2);

    test(`1`, '\1');
    test(`42`, '\42');
    test(`357`, '\357');

    test(`u1234`, '\u1234');
    test(`uf0e4`, '\uf0e4');

    test(`U0001f603`, '\U0001f603');

    test(`&quot;`, '"');
    test(`&lt;`, '<');
    test(`&gt;`, '>');
}
unittest
{
    static final class ExpectDiagnosticReporter : DiagnosticReporter
    {
        string expected;
        bool gotError;

      nothrow:

        this(string expected) { this.expected = expected; }

        override int errorCount() { assert(0); }
        override int warningCount() { assert(0); }
        override int deprecationCount() { assert(0); }

        override void error(const ref Loc loc, const(char)* format, va_list args)
        {
            gotError = true;
            char[100] buffer = void;
            auto actual = buffer[0 .. vsprintf(buffer.ptr, format, args)];
            assert(expected == actual);
        }

        override void errorSupplemental(const ref Loc, const(char)*, va_list) { assert(0); }
        override void warning(const ref Loc, const(char)*, va_list) { assert(0); }
        override void warningSupplemental(const ref Loc, const(char)*, va_list) { assert(0); }
        override void deprecation(const ref Loc, const(char)*, va_list) { assert(0); }
        override void deprecationSupplemental(const ref Loc, const(char)*, va_list) { assert(0); }
    }
    static void test(string sequence, string expectedError, dchar expectedReturnValue, uint expectedScanLength) nothrow
    {
        scope handler = new ExpectDiagnosticReporter(expectedError);
        auto p = cast(const(char)*)sequence.ptr;
        auto actualReturnValue = Lexer.escapeSequence(Loc.initial, handler, p);
        assert(handler.gotError);
        assert(expectedReturnValue == actualReturnValue);

        auto actualScanLength = p - sequence.ptr;
        assert(expectedScanLength == actualScanLength);
    }

    test("c", `undefined escape sequence \c`, 'c', 1);
    test("!", `undefined escape sequence \!`, '!', 1);

    test("x1", `escape hex sequence has 1 hex digits instead of 2`, '\x01', 2);

    test("u1"  , `escape hex sequence has 1 hex digits instead of 4`,   0x1, 2);
    test("u12" , `escape hex sequence has 2 hex digits instead of 4`,  0x12, 3);
    test("u123", `escape hex sequence has 3 hex digits instead of 4`, 0x123, 4);

    test("U0"      , `escape hex sequence has 1 hex digits instead of 8`,       0x0, 2);
    test("U00"     , `escape hex sequence has 2 hex digits instead of 8`,      0x00, 3);
    test("U000"    , `escape hex sequence has 3 hex digits instead of 8`,     0x000, 4);
    test("U0000"   , `escape hex sequence has 4 hex digits instead of 8`,    0x0000, 5);
    test("U0001f"  , `escape hex sequence has 5 hex digits instead of 8`,   0x0001f, 6);
    test("U0001f6" , `escape hex sequence has 6 hex digits instead of 8`,  0x0001f6, 7);
    test("U0001f60", `escape hex sequence has 7 hex digits instead of 8`, 0x0001f60, 8);

    test("ud800"    , `invalid UTF character \U0000d800`, '?', 5);
    test("udfff"    , `invalid UTF character \U0000dfff`, '?', 5);
    test("U00110000", `invalid UTF character \U00110000`, '?', 9);

    test("xg0"      , `undefined escape hex sequence \xg`, 'g', 2);
    test("ug000"    , `undefined escape hex sequence \ug`, 'g', 2);
    test("Ug0000000", `undefined escape hex sequence \Ug`, 'g', 2);

    test("&BAD;", `unnamed character entity &BAD;`  , '?', 5);
    test("&quot", `unterminated named entity &quot;`, '?', 5);

    test("400", `escape octal sequence \400 is larger than \377`, 0x100, 3);
}
