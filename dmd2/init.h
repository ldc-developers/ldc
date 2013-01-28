
// Compiler implementation of the D programming language
// Copyright (c) 1999-2007 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef INIT_H
#define INIT_H

#include "root.h"

#include "mars.h"
#include "arraytypes.h"

struct Identifier;
struct Expression;
struct Scope;
struct Type;
struct dt_t;
struct AggregateDeclaration;
struct VoidInitializer;
struct StructInitializer;
struct ArrayInitializer;
struct ExpInitializer;
struct HdrGenState;

#if IN_LLVM
namespace llvm {
    class StructType;
}
#endif

enum NeedInterpret { INITnointerpret, INITinterpret };

struct Initializer : Object
{
    Loc loc;

    Initializer(Loc loc);
    virtual Initializer *syntaxCopy();
    // needInterpret is INITinterpret if must be a manifest constant, 0 if not.
    virtual Initializer *semantic(Scope *sc, Type *t, NeedInterpret needInterpret);
    virtual Type *inferType(Scope *sc);
    virtual Expression *toExpression() = 0;
    virtual void toCBuffer(OutBuffer *buf, HdrGenState *hgs) = 0;
    char *toChars();

    static Initializers *arraySyntaxCopy(Initializers *ai);

#if IN_DMD
    virtual dt_t *toDt();
#endif

    virtual VoidInitializer *isVoidInitializer() { return NULL; }
    virtual StructInitializer  *isStructInitializer()  { return NULL; }
    virtual ArrayInitializer  *isArrayInitializer()  { return NULL; }
    virtual ExpInitializer  *isExpInitializer()  { return NULL; }
};

struct VoidInitializer : Initializer
{
    Type *type;         // type that this will initialize to

    VoidInitializer(Loc loc);
    Initializer *syntaxCopy();
    Initializer *semantic(Scope *sc, Type *t, NeedInterpret needInterpret);
    Expression *toExpression();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

#if IN_DMD
    dt_t *toDt();
#endif

    virtual VoidInitializer *isVoidInitializer() { return this; }
};

struct StructInitializer : Initializer
{
    Identifiers field;  // of Identifier *'s
    Initializers value; // parallel array of Initializer *'s

    VarDeclarations vars;       // parallel array of VarDeclaration *'s
    AggregateDeclaration *ad;   // which aggregate this is for

    StructInitializer(Loc loc);
    Initializer *syntaxCopy();
    void addInit(Identifier *field, Initializer *value);
    Initializer *semantic(Scope *sc, Type *t, NeedInterpret needInterpret);
    Expression *toExpression();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

#if IN_DMD
    dt_t *toDt();
#endif

    StructInitializer *isStructInitializer() { return this; }
#if IN_LLVM
    llvm::StructType *ltype;
#endif
};

struct ArrayInitializer : Initializer
{
    Expressions index;  // indices
    Initializers value; // of Initializer *'s
    size_t dim;         // length of array being initialized
    Type *type;         // type that array will be used to initialize
    int sem;            // !=0 if semantic() is run

    ArrayInitializer(Loc loc);
    Initializer *syntaxCopy();
    void addInit(Expression *index, Initializer *value);
    Initializer *semantic(Scope *sc, Type *t, NeedInterpret needInterpret);
    int isAssociativeArray();
    Type *inferType(Scope *sc);
    Expression *toExpression();
    Expression *toAssocArrayLiteral();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

#if IN_DMD
    dt_t *toDt();
    dt_t *toDtBit();    // for bit arrays
#endif

    ArrayInitializer *isArrayInitializer() { return this; }
};

struct ExpInitializer : Initializer
{
    Expression *exp;
    int expandTuples;

    ExpInitializer(Loc loc, Expression *exp);
    Initializer *syntaxCopy();
    Initializer *semantic(Scope *sc, Type *t, NeedInterpret needInterpret);
    Type *inferType(Scope *sc);
    Expression *toExpression();
    void toCBuffer(OutBuffer *buf, HdrGenState *hgs);

#if IN_DMD
    dt_t *toDt();
#endif

    virtual ExpInitializer *isExpInitializer() { return this; }
};

#endif
