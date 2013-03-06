
// Compiler implementation of the D programming language
// Copyright (c) 1999-2006 by Digital Mars
// All Rights Reserved
// initial header generation implementation by Dave Fladebo
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#include <string.h>                     // memset()

struct HdrGenState
{
    int hdrgen;         // 1 if generating header file
    int ddoc;           // 1 if generating Ddoc file
    int console;        // 1 if writing to console
    int tpltMember;
    int inCallExp;
    int inPtrExp;
    int inSlcExp;
    int inDotExp;
    int inBinExp;
    int inArrExp;
    int emitInst;
    int autoMember;

    struct
    {
        int init;
        int decl;
    } FLinit;
    Scope* scope;       // Scope when generating ddoc

    HdrGenState() { memset(this, 0, sizeof(HdrGenState)); }
};