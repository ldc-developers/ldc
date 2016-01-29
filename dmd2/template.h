
/* Compiler implementation of the D programming language
 * Copyright (c) 1999-2014 by Digital Mars
 * All Rights Reserved
 * written by Walter Bright
 * http://www.digitalmars.com
 * Distributed under the Boost Software License, Version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 * https://github.com/D-Programming-Language/dmd/blob/master/src/template.h
 */

#ifndef DMD_TEMPLATE_H
#define DMD_TEMPLATE_H

#ifdef __DMC__
#pragma once
#endif /* __DMC__ */

#if IN_LLVM
#include <string>
#endif
#include "root.h"
#include "arraytypes.h"
#include "dsymbol.h"


struct OutBuffer;
class Identifier;
class TemplateInstance;
class TemplateParameter;
class TemplateTypeParameter;
class TemplateThisParameter;
class TemplateValueParameter;
class TemplateAliasParameter;
class TemplateTupleParameter;
class Type;
class TypeQualified;
class TypeTypeof;
struct Scope;
class Expression;
class AliasDeclaration;
class FuncDeclaration;
class Parameter;
enum MATCH;
enum PASS;

class Tuple : public RootObject
{
public:
    Objects objects;

    // kludge for template.isType()
    int dyncast() { return DYNCAST_TUPLE; }

    char *toChars() { return objects.toChars(); }
};

struct TemplatePrevious
{
    TemplatePrevious *prev;
    Scope *sc;
    Objects *dedargs;
};

class TemplateDeclaration : public ScopeDsymbol
{
public:
    TemplateParameters *parameters;     // array of TemplateParameter's

    TemplateParameters *origParameters; // originals for Ddoc
    Expression *constraint;

    // Hash table to look up TemplateInstance's of this TemplateDeclaration
    Array<TemplateInstances *> buckets;
    size_t numinstances;                // number of instances in the hash table

    TemplateDeclaration *overnext;      // next overloaded TemplateDeclaration
    TemplateDeclaration *overroot;      // first in overnext list
    FuncDeclaration *funcroot;          // first function in unified overload list

    Dsymbol *onemember;         // if !=NULL then one member of this template

    bool literal;               // this template declaration is a literal
    bool ismixin;               // template declaration is only to be used as a mixin
    bool isstatic;              // this is static template declaration
    Prot protection;

    TemplatePrevious *previous;         // threaded list of previous instantiation attempts on stack

    TemplateDeclaration(Loc loc, Identifier *id, TemplateParameters *parameters,
        Expression *constraint, Dsymbols *decldefs, bool ismixin = false, bool literal = false);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc);
    bool overloadInsert(Dsymbol *s);
    bool hasStaticCtorOrDtor();
    const char *kind();
    char *toChars();

    Prot prot();

    bool evaluateConstraint(TemplateInstance *ti, Scope *sc, Scope *paramscope, Objects *dedtypes, FuncDeclaration *fd);

    MATCH matchWithInstance(Scope *sc, TemplateInstance *ti, Objects *atypes, Expressions *fargs, int flag);
    MATCH leastAsSpecialized(Scope *sc, TemplateDeclaration *td2, Expressions *fargs);

    MATCH deduceFunctionTemplateMatch(TemplateInstance *ti, Scope *sc, FuncDeclaration *&fd, Type *tthis, Expressions *fargs);
    RootObject *declareParameter(Scope *sc, TemplateParameter *tp, RootObject *o);
    FuncDeclaration *doHeaderInstantiation(TemplateInstance *ti, Scope *sc, FuncDeclaration *fd, Type *tthis, Expressions *fargs);
    TemplateInstance *findExistingInstance(TemplateInstance *tithis, Expressions *fargs);
    TemplateInstance *addInstance(TemplateInstance *ti);
    void removeInstance(TemplateInstance *handle);

    TemplateDeclaration *isTemplateDeclaration() { return this; }

    TemplateTupleParameter *isVariadic();
    bool isOverloadable();

    void accept(Visitor *v) { v->visit(this); }

#if IN_LLVM
    // LDC
    std::string intrinsicName;
#endif
};

/* For type-parameter:
 *  template Foo(ident)             // specType is set to NULL
 *  template Foo(ident : specType)
 * For value-parameter:
 *  template Foo(valType ident)     // specValue is set to NULL
 *  template Foo(valType ident : specValue)
 * For alias-parameter:
 *  template Foo(alias ident)
 * For this-parameter:
 *  template Foo(this ident)
 */
class TemplateParameter
{
public:
    Loc loc;
    Identifier *ident;

    /* True if this is a part of precedent parameter specialization pattern.
     *
     *  template A(T : X!TL, alias X, TL...) {}
     *  // X and TL are dependent template parameter
     *
     * A dependent template parameter should return MATCHexact in matchArg()
     * to respect the match level of the corresponding precedent parameter.
     */
    bool dependent;

    TemplateParameter(Loc loc, Identifier *ident);

    virtual TemplateTypeParameter  *isTemplateTypeParameter();
    virtual TemplateValueParameter *isTemplateValueParameter();
    virtual TemplateAliasParameter *isTemplateAliasParameter();
    virtual TemplateThisParameter *isTemplateThisParameter();
    virtual TemplateTupleParameter *isTemplateTupleParameter();

    virtual TemplateParameter *syntaxCopy() = 0;
    virtual bool declareParameter(Scope *sc) = 0;
    virtual bool semantic(Scope *sc, TemplateParameters *parameters) = 0;
    virtual void print(RootObject *oarg, RootObject *oded) = 0;
    virtual RootObject *specialization() = 0;
    virtual RootObject *defaultArg(Loc instLoc, Scope *sc) = 0;
    virtual bool hasDefaultArg() = 0;

    /* Match actual argument against parameter.
     */
    virtual MATCH matchArg(Loc instLoc, Scope *sc, Objects *tiargs, size_t i, TemplateParameters *parameters, Objects *dedtypes, Declaration **psparam);
    virtual MATCH matchArg(Scope *sc, RootObject *oarg, size_t i, TemplateParameters *parameters, Objects *dedtypes, Declaration **psparam) = 0;

    /* Create dummy argument based on parameter.
     */
    virtual void *dummyArg() = 0;
    virtual void accept(Visitor *v) { v->visit(this); }
};

/* Syntax:
 *  ident : specType = defaultType
 */
class TemplateTypeParameter : public TemplateParameter
{
public:
    Type *specType;     // type parameter: if !=NULL, this is the type specialization
    Type *defaultType;

    static Type *tdummy;

    TemplateTypeParameter(Loc loc, Identifier *ident, Type *specType, Type *defaultType);

    TemplateTypeParameter *isTemplateTypeParameter();
    TemplateParameter *syntaxCopy();
    bool declareParameter(Scope *sc);
    bool semantic(Scope *sc, TemplateParameters *parameters);
    void print(RootObject *oarg, RootObject *oded);
    RootObject *specialization();
    RootObject *defaultArg(Loc instLoc, Scope *sc);
    bool hasDefaultArg();
    MATCH matchArg(Scope *sc, RootObject *oarg, size_t i, TemplateParameters *parameters, Objects *dedtypes, Declaration **psparam);
    void *dummyArg();
    void accept(Visitor *v) { v->visit(this); }
};

/* Syntax:
 *  this ident : specType = defaultType
 */
class TemplateThisParameter : public TemplateTypeParameter
{
public:
    TemplateThisParameter(Loc loc, Identifier *ident, Type *specType, Type *defaultType);

    TemplateThisParameter *isTemplateThisParameter();
    TemplateParameter *syntaxCopy();
    void accept(Visitor *v) { v->visit(this); }
};

/* Syntax:
 *  valType ident : specValue = defaultValue
 */
class TemplateValueParameter : public TemplateParameter
{
public:
    Type *valType;
    Expression *specValue;
    Expression *defaultValue;

    static AA *edummies;

    TemplateValueParameter(Loc loc, Identifier *ident, Type *valType, Expression *specValue, Expression *defaultValue);

    TemplateValueParameter *isTemplateValueParameter();
    TemplateParameter *syntaxCopy();
    bool declareParameter(Scope *sc);
    bool semantic(Scope *sc, TemplateParameters *parameters);
    void print(RootObject *oarg, RootObject *oded);
    RootObject *specialization();
    RootObject *defaultArg(Loc instLoc, Scope *sc);
    bool hasDefaultArg();
    MATCH matchArg(Scope *sc, RootObject *oarg, size_t i, TemplateParameters *parameters, Objects *dedtypes, Declaration **psparam);
    void *dummyArg();
    void accept(Visitor *v) { v->visit(this); }
};

/* Syntax:
 *  specType ident : specAlias = defaultAlias
 */
class TemplateAliasParameter : public TemplateParameter
{
public:
    Type *specType;
    RootObject *specAlias;
    RootObject *defaultAlias;

    static Dsymbol *sdummy;

    TemplateAliasParameter(Loc loc, Identifier *ident, Type *specType, RootObject *specAlias, RootObject *defaultAlias);

    TemplateAliasParameter *isTemplateAliasParameter();
    TemplateParameter *syntaxCopy();
    bool declareParameter(Scope *sc);
    bool semantic(Scope *sc, TemplateParameters *parameters);
    void print(RootObject *oarg, RootObject *oded);
    RootObject *specialization();
    RootObject *defaultArg(Loc instLoc, Scope *sc);
    bool hasDefaultArg();
    MATCH matchArg(Scope *sc, RootObject *oarg, size_t i, TemplateParameters *parameters, Objects *dedtypes, Declaration **psparam);
    void *dummyArg();
    void accept(Visitor *v) { v->visit(this); }
};

/* Syntax:
 *  ident ...
 */
class TemplateTupleParameter : public TemplateParameter
{
public:
    TemplateTupleParameter(Loc loc, Identifier *ident);

    TemplateTupleParameter *isTemplateTupleParameter();
    TemplateParameter *syntaxCopy();
    bool declareParameter(Scope *sc);
    bool semantic(Scope *sc, TemplateParameters *parameters);
    void print(RootObject *oarg, RootObject *oded);
    RootObject *specialization();
    RootObject *defaultArg(Loc instLoc, Scope *sc);
    bool hasDefaultArg();
    MATCH matchArg(Loc loc, Scope *sc, Objects *tiargs, size_t i, TemplateParameters *parameters, Objects *dedtypes, Declaration **psparam);
    MATCH matchArg(Scope *sc, RootObject *oarg, size_t i, TemplateParameters *parameters, Objects *dedtypes, Declaration **psparam);
    void *dummyArg();
    void accept(Visitor *v) { v->visit(this); }
};

/* Given:
 *  foo!(args) =>
 *      name = foo
 *      tiargs = args
 */
class TemplateInstance : public ScopeDsymbol
{
public:
    Identifier *name;

    // Array of Types/Expressions of template
    // instance arguments [int*, char, 10*10]
    Objects *tiargs;

    // Array of Types/Expressions corresponding
    // to TemplateDeclaration.parameters
    // [int, char, 100]
    Objects tdtypes;

    Dsymbol *tempdecl;                  // referenced by foo.bar.abc
    Dsymbol *enclosing;                 // if referencing local symbols, this is the context
    Dsymbol *aliasdecl;                 // !=NULL if instance is an alias for its sole member
    TemplateInstance *inst;             // refer to existing instance
    ScopeDsymbol *argsym;               // argument symbol table
    int nest;                           // for recursion detection
    bool semantictiargsdone;            // has semanticTiargs() been done?
    bool havetempdecl;                  // if used second constructor
    bool gagged;                        // if the instantiation is done with error gagging
    hash_t hash;                        // cached result of hashCode()
    Expressions *fargs;                 // for function template, these are the function arguments

    TemplateInstances* deferred;

    // Used to determine the instance needs code generation.
    // Note that these are inaccurate until semantic analysis phase completed.
    TemplateInstance *tinst;            // enclosing template instance
    TemplateInstance *tnext;            // non-first instantiated instances
    Module *minst;                      // the top module that instantiated this instance

    TemplateInstance(Loc loc, Identifier *temp_id);
    TemplateInstance(Loc loc, TemplateDeclaration *tempdecl, Objects *tiargs);
    static Objects *arraySyntaxCopy(Objects *objs);
    Dsymbol *syntaxCopy(Dsymbol *);
    void semantic(Scope *sc, Expressions *fargs);
    void semantic(Scope *sc);
    void semantic2(Scope *sc);
    void semantic3(Scope *sc);
    Dsymbol *toAlias();                 // resolve real symbol
    const char *kind();
    bool oneMember(Dsymbol **ps, Identifier *ident);
    char *toChars();
    char* toPrettyCharsHelper();
    void printInstantiationTrace();
    Identifier *getIdent();
    int compare(RootObject *o);
    hash_t hashCode();

    bool needsCodegen();

    // Internal
    bool findTempDecl(Scope *sc, WithScopeSymbol **pwithsym);
    bool updateTempDecl(Scope *sc, Dsymbol *s);
    static bool semanticTiargs(Loc loc, Scope *sc, Objects *tiargs, int flags);
    bool semanticTiargs(Scope *sc);
    bool findBestMatch(Scope *sc, Expressions *fargs);
    bool needsTypeInference(Scope *sc, int flag = 0);
    bool hasNestedArgs(Objects *tiargs, bool isstatic);
    Dsymbols *appendToModuleMember();
    void declareParameters(Scope *sc);
    Identifier *genIdent(Objects *args);
    void expandMembers(Scope *sc);
    void tryExpandMembers(Scope *sc);
    void trySemantic3(Scope *sc2);

    TemplateInstance *isTemplateInstance() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

class TemplateMixin : public TemplateInstance
{
public:
    TypeQualified *tqual;

    TemplateMixin(Loc loc, Identifier *ident, TypeQualified *tqual, Objects *tiargs);
    Dsymbol *syntaxCopy(Dsymbol *s);
    void semantic(Scope *sc);
    void semantic2(Scope *sc);
    void semantic3(Scope *sc);
    const char *kind();
    bool oneMember(Dsymbol **ps, Identifier *ident);
    int apply(Dsymbol_apply_ft_t fp, void *param);
    bool hasPointers();
    void setFieldOffset(AggregateDeclaration *ad, unsigned *poffset, bool isunion);
    char *toChars();

    bool findTempDecl(Scope *sc);

    TemplateMixin *isTemplateMixin() { return this; }
    void accept(Visitor *v) { v->visit(this); }
};

Expression *isExpression(RootObject *o);
Dsymbol *isDsymbol(RootObject *o);
Type *isType(RootObject *o);
Tuple *isTuple(RootObject *o);
Parameter *isParameter(RootObject *o);
bool arrayObjectIsError(Objects *args);
bool isError(RootObject *o);
Type *getType(RootObject *o);
Dsymbol *getDsymbol(RootObject *o);

RootObject *objectSyntaxCopy(RootObject *o);

#endif /* DMD_TEMPLATE_H */
