// Compiler implementation of the D programming language
// Copyright (c) 1999-2012 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef DMD_CTFE_H
#define DMD_CTFE_H

#ifdef __DMC__
#pragma once
#endif /* __DMC__ */


/**
   Global status of the CTFE engine. Mostly used for performance diagnostics
 */
struct CtfeStatus
{
    static int callDepth; // current number of recursive calls
    static int stackTraceCallsToSuppress; /* When printing a stack trace,
                                           * suppress this number of calls
                                           */
    static int maxCallDepth; // highest number of recursive calls
    static int numArrayAllocs; // Number of allocated arrays
    static int numAssignments; // total number of assignments executed
};


/** Expression subclasses which only exist in CTFE */

#define TOKclassreference ((TOK)(TOKMAX+1))
#define TOKthrownexception ((TOK)(TOKMAX+2))

/**
  A reference to a class, or an interface. We need this when we
  point to a base class (we must record what the type is).
 */
struct ClassReferenceExp : Expression
{
    StructLiteralExp *value;
    ClassReferenceExp(Loc loc, StructLiteralExp *lit, Type *type);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    char *toChars();
    ClassDeclaration *originalClass();
    VarDeclaration *getFieldAt(unsigned index);

    /// Return index of the field, or -1 if not found
    int getFieldIndex(Type *fieldtype, unsigned fieldoffset);
    /// Return index of the field, or -1 if not found
    /// Same as getFieldIndex, but checks for a direct match with the VarDeclaration
    int findFieldIndexByName(VarDeclaration *v);
};

/// Return index of the field, or -1 if not found
/// Same as getFieldIndex, but checks for a direct match with the VarDeclaration
int findFieldIndexByName(StructDeclaration *sd, VarDeclaration *v);


/** An uninitialized value
 */
struct VoidInitExp : Expression
{
    VarDeclaration *var;

    VoidInitExp(VarDeclaration *var, Type *type);
    char *toChars();
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
};


/** Fake class which holds the thrown exception.
    Used for implementing exception handling.
*/
struct ThrownExceptionExp : Expression
{
    ClassReferenceExp *thrown; // the thing being tossed
    ThrownExceptionExp(Loc loc, ClassReferenceExp *victim);
    Expression *interpret(InterState *istate, CtfeGoal goal = ctfeNeedRvalue);
    char *toChars();
    /// Generate an error message when this exception is not caught
    void generateUncaughtError();
};


/// True if 'e' is EXP_CANT_INTERPRET, or an exception
bool exceptionOrCantInterpret(Expression *e);

// Used for debugging only
void showCtfeExpr(Expression *e, int level = 0);

/// Return true if this is a valid CTFE expression
bool isCtfeValueValid(Expression *newval);


/// Given expr, which evaluates to an array/AA/string literal,
/// return true if it needs to be copied
bool needToCopyLiteral(Expression *expr);

/// Make a copy of the ArrayLiteral, AALiteral, String, or StructLiteral.
/// This value will be used for in-place modification.
Expression *copyLiteral(Expression *e);

/// Set this literal to the given type, copying it if necessary
Expression *paintTypeOntoLiteral(Type *type, Expression *lit);

/// Convert from a CTFE-internal slice, into a normal Expression
Expression *resolveSlice(Expression *e);

/// Determine the array length, without interpreting the expression.
uinteger_t resolveArrayLength(Expression *e);

/// Create an array literal consisting of 'elem' duplicated 'dim' times.
ArrayLiteralExp *createBlockDuplicatedArrayLiteral(Loc loc, Type *type,
        Expression *elem, size_t dim);

/// Create a string literal consisting of 'value' duplicated 'dim' times.
StringExp *createBlockDuplicatedStringLiteral(Loc loc, Type *type,
        unsigned value, size_t dim, int sz);


/* Set dest = src, where both dest and src are container value literals
 * (ie, struct literals, or static arrays (can be an array literal or a string)
 * Assignment is recursively in-place.
 * Purpose: any reference to a member of 'dest' will remain valid after the
 * assignment.
 */
void assignInPlace(Expression *dest, Expression *src);

/// Set all elements of 'ae' to 'val'. ae may be a multidimensional array.
/// If 'wantRef', all elements of ae will hold references to the same val.
void recursiveBlockAssign(ArrayLiteralExp *ae, Expression *val, bool wantRef);

/// Duplicate the elements array, then set field 'indexToChange' = newelem.
Expressions *changeOneElement(Expressions *oldelems, size_t indexToChange, Expression *newelem);

/// Create a new struct literal, which is the same as se except that se.field[offset] = elem
Expression * modifyStructField(Type *type, StructLiteralExp *se, size_t offset, Expression *newval);

/// Given an AA literal aae,  set arr[index] = newval and return the new array.
Expression *assignAssocArrayElement(Loc loc, AssocArrayLiteralExp *aae,
    Expression *index, Expression *newval);

/// Given array literal oldval of type ArrayLiteralExp or StringExp, of length
/// oldlen, change its length to newlen. If the newlen is longer than oldlen,
/// all new elements will be set to the default initializer for the element type.
Expression *changeArrayLiteralLength(Loc loc, TypeArray *arrayType,
    Expression *oldval,  size_t oldlen, size_t newlen);



/// Return true if t is a pointer (not a function pointer)
bool isPointer(Type *t);

// For CTFE only. Returns true if 'e' is TRUE or a non-null pointer.
int isTrueBool(Expression *e);

/// Is it safe to convert from srcPointee* to destPointee* ?
///  srcPointee is the genuine type (never void).
///  destPointee may be void.
bool isSafePointerCast(Type *srcPointee, Type *destPointee);

/// Given pointer e, return the memory block expression it points to,
/// and set ofs to the offset within that memory block.
Expression *getAggregateFromPointer(Expression *e, dinteger_t *ofs);

/// Return true if agg1 and agg2 are pointers to the same memory block
bool pointToSameMemoryBlock(Expression *agg1, Expression *agg2);

// return e1 - e2 as an integer, or error if not possible
Expression *pointerDifference(Loc loc, Type *type, Expression *e1, Expression *e2);

/// Return 1 if true, 0 if false
/// -1 if comparison is illegal because they point to non-comparable memory blocks
int comparePointers(Loc loc, enum TOK op, Type *type, Expression *agg1, dinteger_t ofs1, Expression *agg2, dinteger_t ofs2);

// Return eptr op e2, where eptr is a pointer, e2 is an integer,
// and op is TOKadd or TOKmin
Expression *pointerArithmetic(Loc loc, enum TOK op, Type *type,
    Expression *eptr, Expression *e2);

// True if conversion from type 'from' to 'to' involves a reinterpret_cast
// floating point -> integer or integer -> floating point
bool isFloatIntPaint(Type *to, Type *from);

// Reinterpret float/int value 'fromVal' as a float/integer of type 'to'.
Expression *paintFloatInt(Expression *fromVal, Type *to);

/// Return true if t is an AA, or AssociativeArray!(key, value)
bool isAssocArray(Type *t);

/// Given a template AA type, extract the corresponding built-in AA type
TypeAArray *toBuiltinAAType(Type *t);

/*  Given an AA literal 'ae', and a key 'e2':
 *  Return ae[e2] if present, or NULL if not found.
 *  Return EXP_CANT_INTERPRET on error.
 */
Expression *findKeyInAA(Loc loc, AssocArrayLiteralExp *ae, Expression *e2);

/***********************************************
      In-place integer operations
***********************************************/

/// e = OP e
void intUnary(TOK op, IntegerExp *e);

/// dest = e1 OP e2;
void intBinary(TOK op, IntegerExp *dest, Type *type, IntegerExp *e1, IntegerExp *e2);


/***********************************************
      COW const-folding operations
***********************************************/

/// Return true if non-pointer expression e can be compared
/// with >,is, ==, etc, using ctfeCmp, ctfeEquals, ctfeIdentity
bool isCtfeComparable(Expression *e);

/// Evaluate ==, !=.  Resolves slices before comparing. Returns 0 or 1
int ctfeEqual(Loc loc, enum TOK op, Expression *e1, Expression *e2);

/// Evaluate is, !is.  Resolves slices before comparing. Returns 0 or 1
int ctfeIdentity(Loc loc, enum TOK op, Expression *e1, Expression *e2);

/// Evaluate >,<=, etc. Resolves slices before comparing. Returns 0 or 1
int ctfeCmp(Loc loc, enum TOK op, Expression *e1, Expression *e2);

/// Returns e1 ~ e2. Resolves slices before concatenation.
Expression *ctfeCat(Type *type, Expression *e1, Expression *e2);

/// Same as for constfold.Index, except that it only works for static arrays,
/// dynamic arrays, and strings.
Expression *ctfeIndex(Loc loc, Type *type, Expression *e1, uinteger_t indx);

/// Cast 'e' of type 'type' to type 'to'.
Expression *ctfeCast(Loc loc, Type *type, Type *to, Expression *e);


#endif /* DMD_CTFE_H */
