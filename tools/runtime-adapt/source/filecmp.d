//===-- tools/runtime-adapt/source/filecmp.d ----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Three-way full-file compare (stock / generated / goal). This is how LDC
// reviews a runtime change: look at the hunk, not a token scan.
//
//   match        generated == goal
//   ldc-emit     ldc/* we generate from compiler sources
//   adapt-delta  we changed a stock file the goal left alone
//   missed-ldc   stock==generated, goal has an LDC patch we did not apply
//
//===----------------------------------------------------------------------===//

module filecmp;

import std.array : split;

struct FileCmp
{
    string rel;
    string klass; /// match | ldc-emit | adapt-delta | missed-ldc | extra | missing | text-diff
    bool textsEqual;
    bool stockEqualsGoal;
    bool stockEqualsGen;
    size_t genLen;
    size_t goalLen;
    size_t stockLen;
    int firstDiffLine;
    string genLine;
    string goalLine;
}

FileCmp classifyFile(string rel, string genTxt, string goalTxt, string stockTxt)
{
    FileCmp c;
    c.rel = rel;
    c.genLen = genTxt.length;
    c.goalLen = goalTxt.length;
    c.stockLen = stockTxt.length;
    c.textsEqual = genTxt.length && genTxt == goalTxt;
    c.stockEqualsGoal = stockTxt.length && stockTxt == goalTxt;
    c.stockEqualsGen = stockTxt.length && stockTxt == genTxt;
    firstHunk(genTxt, goalTxt, c.firstDiffLine, c.genLine, c.goalLine);
    immutable isLdc = rel.length >= 4 && rel[0 .. 4] == "ldc/";
    if (c.textsEqual)
        c.klass = "match";
    else if (!genTxt.length)
        c.klass = "missing";
    else if (!goalTxt.length)
        c.klass = "extra";
    else if (isLdc)
        c.klass = "ldc-emit";
    else if (c.stockEqualsGoal && !c.stockEqualsGen)
        c.klass = "adapt-delta";
    else if (c.stockEqualsGen && !c.stockEqualsGoal)
        c.klass = "missed-ldc";
    else
        c.klass = "text-diff";
    return c;
}

void firstHunk(string a, string b, ref int line, ref string al, ref string bl)
{
    auto la = a.split("\n");
    auto lb = b.split("\n");
    auto n = la.length < lb.length ? la.length : lb.length;
    foreach (i; 0 .. n)
    {
        if (la[i] != lb[i])
        {
            line = cast(int)(i + 1);
            al = clip(la[i], 80);
            bl = clip(lb[i], 80);
            return;
        }
    }
    if (la.length != lb.length)
    {
        line = cast(int)(n + 1);
        al = la.length > n ? clip(la[n], 80) : "(eof)";
        bl = lb.length > n ? clip(lb[n], 80) : "(eof)";
    }
}

private string clip(string s, size_t n)
{
    if (s.length <= n)
        return s;
    return s[0 .. n] ~ "…";
}
