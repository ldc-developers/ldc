/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/tokens.d, _tokens.d)
 * Documentation:  https://dlang.org/phobos/dmd_tokens.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/tokens.d
 */

module dmd.tokens;

import core.stdc.ctype;
import core.stdc.stdio;
import core.stdc.string;
import dmd.globals;
import dmd.identifier;
import dmd.root.ctfloat;
import dmd.root.outbuffer;
import dmd.root.rmem;
import dmd.utf;

// IN_LLVM: ubyte -> int due to https://issues.dlang.org/show_bug.cgi?id=19658
enum TOK : int
{
    reserved,

    // Other
    leftParentheses,
    rightParentheses,
    leftBracket,
    rightBracket,
    leftCurly,
    rightCurly,
    colon,
    negate,
    semicolon,
    dotDotDot,
    endOfFile,
    cast_,
    null_,
    assert_,
    true_,
    false_,
    array,
    call,
    address,
    type,
    throw_,
    new_,
    delete_,
    star,
    symbolOffset,
    variable,
    dotVariable,
    dotIdentifier,
    dotTemplateInstance,
    dotType,
    slice,
    arrayLength,
    version_,
    module_,
    dollar,
    template_,
    dotTemplateDeclaration,
    declaration,
    typeof_,
    pragma_,
    dSymbol,
    typeid_,
    uadd,
    remove,
    newAnonymousClass,
    comment,
    arrayLiteral,
    assocArrayLiteral,
    structLiteral,
    classReference,
    thrownException,
    delegatePointer,
    delegateFunctionPointer,

    // Operators
    lessThan = 54,
    greaterThan,
    lessOrEqual,
    greaterOrEqual,
    equal,
    notEqual,
    identity,
    notIdentity,
    index,
    is_,

    leftShift = 64,
    rightShift,
    leftShiftAssign,
    rightShiftAssign,
    unsignedRightShift,
    unsignedRightShiftAssign,
    concatenate,
    concatenateAssign, // ~=
    concatenateElemAssign,
    concatenateDcharAssign,
    add,
    min,
    addAssign,
    minAssign,
    mul,
    div,
    mod,
    mulAssign,
    divAssign,
    modAssign,
    and,
    or,
    xor,
    andAssign,
    orAssign,
    xorAssign,
    assign,
    not,
    tilde,
    plusPlus,
    minusMinus,
    construct,
    blit,
    dot,
    arrow,
    comma,
    question,
    andAnd,
    orOr,
    prePlusPlus,
    preMinusMinus,

    // Numeric literals
    int32Literal = 105,
    uns32Literal,
    int64Literal,
    uns64Literal,
    int128Literal,
    uns128Literal,
    float32Literal,
    float64Literal,
    float80Literal,
    imaginary32Literal,
    imaginary64Literal,
    imaginary80Literal,

    // Char constants
    charLiteral = 117,
    wcharLiteral,
    dcharLiteral,

    // Leaf operators
    identifier = 120,
    string_,
    hexadecimalString,
    this_,
    super_,
    halt,
    tuple,
    error,

    // Basic types
    void_ = 128,
    int8,
    uns8,
    int16,
    uns16,
    int32,
    uns32,
    int64,
    uns64,
    int128,
    uns128,
    float32,
    float64,
    float80,
    imaginary32,
    imaginary64,
    imaginary80,
    complex32,
    complex64,
    complex80,
    char_,
    wchar_,
    dchar_,
    bool_,

    // Aggregates
    struct_ = 152,
    class_,
    interface_,
    union_,
    enum_,
    import_,
    alias_,
    override_,
    delegate_,
    function_,
    mixin_,
    align_,
    extern_,
    private_,
    protected_,
    public_,
    export_,
    static_,
    final_,
    const_,
    abstract_,
    debug_,
    deprecated_,
    in_,
    out_,
    inout_,
    lazy_,
    auto_,
    package_,
    immutable_,

    // Statements
    if_ = 182,
    else_,
    while_,
    for_,
    do_,
    switch_,
    case_,
    default_,
    break_,
    continue_,
    with_,
    synchronized_,
    return_,
    goto_,
    try_,
    catch_,
    finally_,
    asm_,
    foreach_,
    foreach_reverse_,
    scope_,
    onScopeExit,
    onScopeFailure,
    onScopeSuccess,

    // Contracts
    invariant_ = 206,

    // Testing
    unittest_,

    // Added after 1.0
    argumentTypes,
    ref_,
    macro_,

    parameters = 211,
    traits,
    overloadSet,
    pure_,
    nothrow_,
    gshared,
    line,
    file,
    fileFullPath,
    moduleString,
    functionString,
    prettyFunction,
    shared_,
    at,
    pow,
    powAssign,
    goesTo,
    vector,
    pound,

    interval = 230,
    voidExpression,
    cantExpression,
    showCtfeContext,

    objcClassReference,
    vectorArray,

    max_,
}

// Assert that all token enum members have consecutive values and
// that none of them overlap
static assert(() {
    foreach (idx, enumName; __traits(allMembers, TOK)) {
       static if (idx != __traits(getMember, TOK, enumName)) {
           pragma(msg, "Error: Expected TOK.", enumName, " to be ", idx, " but is ", __traits(getMember, TOK, enumName));
           static assert(0);
       }
    }
    return true;
}());


/****************************************
 */

private immutable TOK[] keywords =
[
    TOK.this_,
    TOK.super_,
    TOK.assert_,
    TOK.null_,
    TOK.true_,
    TOK.false_,
    TOK.cast_,
    TOK.new_,
    TOK.delete_,
    TOK.throw_,
    TOK.module_,
    TOK.pragma_,
    TOK.typeof_,
    TOK.typeid_,
    TOK.template_,
    TOK.void_,
    TOK.int8,
    TOK.uns8,
    TOK.int16,
    TOK.uns16,
    TOK.int32,
    TOK.uns32,
    TOK.int64,
    TOK.uns64,
    TOK.int128,
    TOK.uns128,
    TOK.float32,
    TOK.float64,
    TOK.float80,
    TOK.bool_,
    TOK.char_,
    TOK.wchar_,
    TOK.dchar_,
    TOK.imaginary32,
    TOK.imaginary64,
    TOK.imaginary80,
    TOK.complex32,
    TOK.complex64,
    TOK.complex80,
    TOK.delegate_,
    TOK.function_,
    TOK.is_,
    TOK.if_,
    TOK.else_,
    TOK.while_,
    TOK.for_,
    TOK.do_,
    TOK.switch_,
    TOK.case_,
    TOK.default_,
    TOK.break_,
    TOK.continue_,
    TOK.synchronized_,
    TOK.return_,
    TOK.goto_,
    TOK.try_,
    TOK.catch_,
    TOK.finally_,
    TOK.with_,
    TOK.asm_,
    TOK.foreach_,
    TOK.foreach_reverse_,
    TOK.scope_,
    TOK.struct_,
    TOK.class_,
    TOK.interface_,
    TOK.union_,
    TOK.enum_,
    TOK.import_,
    TOK.mixin_,
    TOK.static_,
    TOK.final_,
    TOK.const_,
    TOK.alias_,
    TOK.override_,
    TOK.abstract_,
    TOK.debug_,
    TOK.deprecated_,
    TOK.in_,
    TOK.out_,
    TOK.inout_,
    TOK.lazy_,
    TOK.auto_,
    TOK.align_,
    TOK.extern_,
    TOK.private_,
    TOK.package_,
    TOK.protected_,
    TOK.public_,
    TOK.export_,
    TOK.invariant_,
    TOK.unittest_,
    TOK.version_,
    TOK.argumentTypes,
    TOK.parameters,
    TOK.ref_,
    TOK.macro_,
    TOK.pure_,
    TOK.nothrow_,
    TOK.gshared,
    TOK.traits,
    TOK.vector,
    TOK.overloadSet,
    TOK.file,
    TOK.fileFullPath,
    TOK.line,
    TOK.moduleString,
    TOK.functionString,
    TOK.prettyFunction,
    TOK.shared_,
    TOK.immutable_,
];

/***********************************************************
 */
extern (C++) struct Token
{
    Token* next;
    Loc loc;
    const(char)* ptr; // pointer to first character of this token within buffer
    TOK value;
    const(char)* blockComment; // doc comment string prior to this token
    const(char)* lineComment; // doc comment for previous token

    union
    {
        // Integers
        sinteger_t intvalue;
        uinteger_t unsvalue;
        // Floats
        real_t floatvalue;

        struct
        {
            const(char)* ustring; // UTF8 string
            uint len;
            ubyte postfix; // 'c', 'w', 'd'
        }

        Identifier ident;
    }

    extern (D) private __gshared immutable string[TOK.max_] tochars =
    [
        // Keywords
        TOK.this_: "this",
        TOK.super_: "super",
        TOK.assert_: "assert",
        TOK.null_: "null",
        TOK.true_: "true",
        TOK.false_: "false",
        TOK.cast_: "cast",
        TOK.new_: "new",
        TOK.delete_: "delete",
        TOK.throw_: "throw",
        TOK.module_: "module",
        TOK.pragma_: "pragma",
        TOK.typeof_: "typeof",
        TOK.typeid_: "typeid",
        TOK.template_: "template",
        TOK.void_: "void",
        TOK.int8: "byte",
        TOK.uns8: "ubyte",
        TOK.int16: "short",
        TOK.uns16: "ushort",
        TOK.int32: "int",
        TOK.uns32: "uint",
        TOK.int64: "long",
        TOK.uns64: "ulong",
        TOK.int128: "cent",
        TOK.uns128: "ucent",
        TOK.float32: "float",
        TOK.float64: "double",
        TOK.float80: "real",
        TOK.bool_: "bool",
        TOK.char_: "char",
        TOK.wchar_: "wchar",
        TOK.dchar_: "dchar",
        TOK.imaginary32: "ifloat",
        TOK.imaginary64: "idouble",
        TOK.imaginary80: "ireal",
        TOK.complex32: "cfloat",
        TOK.complex64: "cdouble",
        TOK.complex80: "creal",
        TOK.delegate_: "delegate",
        TOK.function_: "function",
        TOK.is_: "is",
        TOK.if_: "if",
        TOK.else_: "else",
        TOK.while_: "while",
        TOK.for_: "for",
        TOK.do_: "do",
        TOK.switch_: "switch",
        TOK.case_: "case",
        TOK.default_: "default",
        TOK.break_: "break",
        TOK.continue_: "continue",
        TOK.synchronized_: "synchronized",
        TOK.return_: "return",
        TOK.goto_: "goto",
        TOK.try_: "try",
        TOK.catch_: "catch",
        TOK.finally_: "finally",
        TOK.with_: "with",
        TOK.asm_: "asm",
        TOK.foreach_: "foreach",
        TOK.foreach_reverse_: "foreach_reverse",
        TOK.scope_: "scope",
        TOK.struct_: "struct",
        TOK.class_: "class",
        TOK.interface_: "interface",
        TOK.union_: "union",
        TOK.enum_: "enum",
        TOK.import_: "import",
        TOK.mixin_: "mixin",
        TOK.static_: "static",
        TOK.final_: "final",
        TOK.const_: "const",
        TOK.alias_: "alias",
        TOK.override_: "override",
        TOK.abstract_: "abstract",
        TOK.debug_: "debug",
        TOK.deprecated_: "deprecated",
        TOK.in_: "in",
        TOK.out_: "out",
        TOK.inout_: "inout",
        TOK.lazy_: "lazy",
        TOK.auto_: "auto",
        TOK.align_: "align",
        TOK.extern_: "extern",
        TOK.private_: "private",
        TOK.package_: "package",
        TOK.protected_: "protected",
        TOK.public_: "public",
        TOK.export_: "export",
        TOK.invariant_: "invariant",
        TOK.unittest_: "unittest",
        TOK.version_: "version",
        TOK.argumentTypes: "__argTypes",
        TOK.parameters: "__parameters",
        TOK.ref_: "ref",
        TOK.macro_: "macro",
        TOK.pure_: "pure",
        TOK.nothrow_: "nothrow",
        TOK.gshared: "__gshared",
        TOK.traits: "__traits",
        TOK.vector: "__vector",
        TOK.overloadSet: "__overloadset",
        TOK.file: "__FILE__",
        TOK.fileFullPath: "__FILE_FULL_PATH__",
        TOK.line: "__LINE__",
        TOK.moduleString: "__MODULE__",
        TOK.functionString: "__FUNCTION__",
        TOK.prettyFunction: "__PRETTY_FUNCTION__",
        TOK.shared_: "shared",
        TOK.immutable_: "immutable",

        TOK.endOfFile: "End of File",
        TOK.leftCurly: "{",
        TOK.rightCurly: "}",
        TOK.leftParentheses: "(",
        TOK.rightParentheses: ")",
        TOK.leftBracket: "[",
        TOK.rightBracket: "]",
        TOK.semicolon: ";",
        TOK.colon: ":",
        TOK.comma: ",",
        TOK.dot: ".",
        TOK.xor: "^",
        TOK.xorAssign: "^=",
        TOK.assign: "=",
        TOK.construct: "=",
        TOK.blit: "=",
        TOK.lessThan: "<",
        TOK.greaterThan: ">",
        TOK.lessOrEqual: "<=",
        TOK.greaterOrEqual: ">=",
        TOK.equal: "==",
        TOK.notEqual: "!=",
        TOK.not: "!",
        TOK.leftShift: "<<",
        TOK.rightShift: ">>",
        TOK.unsignedRightShift: ">>>",
        TOK.add: "+",
        TOK.min: "-",
        TOK.mul: "*",
        TOK.div: "/",
        TOK.mod: "%",
        TOK.slice: "..",
        TOK.dotDotDot: "...",
        TOK.and: "&",
        TOK.andAnd: "&&",
        TOK.or: "|",
        TOK.orOr: "||",
        TOK.array: "[]",
        TOK.index: "[i]",
        TOK.address: "&",
        TOK.star: "*",
        TOK.tilde: "~",
        TOK.dollar: "$",
        TOK.plusPlus: "++",
        TOK.minusMinus: "--",
        TOK.prePlusPlus: "++",
        TOK.preMinusMinus: "--",
        TOK.type: "type",
        TOK.question: "?",
        TOK.negate: "-",
        TOK.uadd: "+",
        TOK.variable: "var",
        TOK.addAssign: "+=",
        TOK.minAssign: "-=",
        TOK.mulAssign: "*=",
        TOK.divAssign: "/=",
        TOK.modAssign: "%=",
        TOK.leftShiftAssign: "<<=",
        TOK.rightShiftAssign: ">>=",
        TOK.unsignedRightShiftAssign: ">>>=",
        TOK.andAssign: "&=",
        TOK.orAssign: "|=",
        TOK.concatenateAssign: "~=",
        TOK.concatenateElemAssign: "~=",
        TOK.concatenateDcharAssign: "~=",
        TOK.concatenate: "~",
        TOK.call: "call",
        TOK.identity: "is",
        TOK.notIdentity: "!is",
        TOK.identifier: "identifier",
        TOK.at: "@",
        TOK.pow: "^^",
        TOK.powAssign: "^^=",
        TOK.goesTo: "=>",
        TOK.pound: "#",

        // For debugging
        TOK.error: "error",
        TOK.dotIdentifier: "dotid",
        TOK.dotTemplateDeclaration: "dottd",
        TOK.dotTemplateInstance: "dotti",
        TOK.dotVariable: "dotvar",
        TOK.dotType: "dottype",
        TOK.symbolOffset: "symoff",
        TOK.arrayLength: "arraylength",
        TOK.arrayLiteral: "arrayliteral",
        TOK.assocArrayLiteral: "assocarrayliteral",
        TOK.structLiteral: "structliteral",
        TOK.string_: "string",
        TOK.dSymbol: "symbol",
        TOK.tuple: "tuple",
        TOK.declaration: "declaration",
        TOK.onScopeExit: "scope(exit)",
        TOK.onScopeSuccess: "scope(success)",
        TOK.onScopeFailure: "scope(failure)",
        TOK.delegatePointer: "delegateptr",

        // Finish up
        TOK.reserved: "reserved",
        TOK.remove: "remove",
        TOK.newAnonymousClass: "newanonclass",
        TOK.comment: "comment",
        TOK.classReference: "classreference",
        TOK.thrownException: "thrownexception",
        TOK.delegateFunctionPointer: "delegatefuncptr",
        TOK.arrow: "arrow",
        TOK.int32Literal: "int32v",
        TOK.uns32Literal: "uns32v",
        TOK.int64Literal: "int64v",
        TOK.uns64Literal: "uns64v",
        TOK.int128Literal: "int128v",
        TOK.uns128Literal: "uns128v",
        TOK.float32Literal: "float32v",
        TOK.float64Literal: "float64v",
        TOK.float80Literal: "float80v",
        TOK.imaginary32Literal: "imaginary32v",
        TOK.imaginary64Literal: "imaginary64v",
        TOK.imaginary80Literal: "imaginary80v",
        TOK.charLiteral: "charv",
        TOK.wcharLiteral: "wcharv",
        TOK.dcharLiteral: "dcharv",

        TOK.halt: "halt",
        TOK.hexadecimalString: "xstring",

        TOK.interval: "interval",
        TOK.voidExpression: "voidexp",
        TOK.cantExpression: "cantexp",
        TOK.showCtfeContext : "showCtfeContext",

        TOK.objcClassReference: "class",
        TOK.vectorArray: "vectorarray",
    ];

    static assert(() {
        foreach (s; tochars)
            assert(s.length);
        return true;
    }());

nothrow:

    shared static this()
    {
        Identifier.initTable();
        foreach (kw; keywords)
        {
            //printf("keyword[%d] = '%s'\n",kw, tochars[kw].ptr);
            Identifier.idPool(tochars[kw].ptr, tochars[kw].length, cast(uint)kw);
        }
    }

    int isKeyword() const
    {
        foreach (kw; keywords)
        {
            if (kw == value)
                return 1;
        }
        return 0;
    }

    /****
     * Set to contents of ptr[0..length]
     * Params:
     *  ptr = pointer to string
     *  length = length of string
     */
    void setString(const(char)* ptr, size_t length)
    {
        auto s = cast(char*)mem.xmalloc(length + 1);
        memcpy(s, ptr, length);
        s[length] = 0;
        ustring = s;
        len = cast(uint)length;
        postfix = 0;
    }

    /****
     * Set to contents of buf
     * Params:
     *  buf = string (not zero terminated)
     */
    void setString(const ref OutBuffer buf)
    {
        setString(cast(const(char)*)buf.data, buf.offset);
    }

    /****
     * Set to empty string
     */
    void setString()
    {
        ustring = "";
        len = 0;
        postfix = 0;
    }

    extern (C++) const(char)* toChars() const
    {
        __gshared char[3 + 3 * floatvalue.sizeof + 1] buffer;
        const(char)* p = &buffer[0];
        switch (value)
        {
        case TOK.int32Literal:
            sprintf(&buffer[0], "%d", cast(d_int32)intvalue);
            break;
        case TOK.uns32Literal:
        case TOK.charLiteral:
        case TOK.wcharLiteral:
        case TOK.dcharLiteral:
            sprintf(&buffer[0], "%uU", cast(d_uns32)unsvalue);
            break;
        case TOK.int64Literal:
            sprintf(&buffer[0], "%lldL", cast(long)intvalue);
            break;
        case TOK.uns64Literal:
            sprintf(&buffer[0], "%lluUL", cast(ulong)unsvalue);
            break;
        case TOK.float32Literal:
            CTFloat.sprint(&buffer[0], 'g', floatvalue);
            strcat(&buffer[0], "f");
            break;
        case TOK.float64Literal:
            CTFloat.sprint(&buffer[0], 'g', floatvalue);
            break;
        case TOK.float80Literal:
            CTFloat.sprint(&buffer[0], 'g', floatvalue);
            strcat(&buffer[0], "L");
            break;
        case TOK.imaginary32Literal:
            CTFloat.sprint(&buffer[0], 'g', floatvalue);
            strcat(&buffer[0], "fi");
            break;
        case TOK.imaginary64Literal:
            CTFloat.sprint(&buffer[0], 'g', floatvalue);
            strcat(&buffer[0], "i");
            break;
        case TOK.imaginary80Literal:
            CTFloat.sprint(&buffer[0], 'g', floatvalue);
            strcat(&buffer[0], "Li");
            break;
        case TOK.string_:
            {
                OutBuffer buf;
                buf.writeByte('"');
                for (size_t i = 0; i < len;)
                {
                    dchar c;
                    utf_decodeChar(ustring, len, i, c);
                    switch (c)
                    {
                    case 0:
                        break;
                    case '"':
                    case '\\':
                        buf.writeByte('\\');
                        goto default;
                    default:
                        if (c <= 0x7F)
                        {
                            if (isprint(c))
                                buf.writeByte(c);
                            else
                                buf.printf("\\x%02x", c);
                        }
                        else if (c <= 0xFFFF)
                            buf.printf("\\u%04x", c);
                        else
                            buf.printf("\\U%08x", c);
                        continue;
                    }
                    break;
                }
                buf.writeByte('"');
                if (postfix)
                    buf.writeByte(postfix);
                p = buf.extractChars();
            }
            break;
        case TOK.hexadecimalString:
            {
                OutBuffer buf;
                buf.writeByte('x');
                buf.writeByte('"');
                foreach (size_t i; 0 .. len)
                {
                    if (i)
                        buf.writeByte(' ');
                    buf.printf("%02x", ustring[i]);
                }
                buf.writeByte('"');
                if (postfix)
                    buf.writeByte(postfix);
                buf.writeByte(0);
                p = buf.extractData();
                break;
            }
        case TOK.identifier:
        case TOK.enum_:
        case TOK.struct_:
        case TOK.import_:
        case TOK.wchar_:
        case TOK.dchar_:
        case TOK.bool_:
        case TOK.char_:
        case TOK.int8:
        case TOK.uns8:
        case TOK.int16:
        case TOK.uns16:
        case TOK.int32:
        case TOK.uns32:
        case TOK.int64:
        case TOK.uns64:
        case TOK.int128:
        case TOK.uns128:
        case TOK.float32:
        case TOK.float64:
        case TOK.float80:
        case TOK.imaginary32:
        case TOK.imaginary64:
        case TOK.imaginary80:
        case TOK.complex32:
        case TOK.complex64:
        case TOK.complex80:
        case TOK.void_:
            p = ident.toChars();
            break;
        default:
            p = toChars(value);
            break;
        }
        return p;
    }

    static const(char)* toChars(TOK value)
    {
        return toString(value).ptr;
    }

    extern (D) static string toString(TOK value) pure nothrow @nogc @safe
    {
        return tochars[value];
    }
}
