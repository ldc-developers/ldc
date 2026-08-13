//===-- tools/runtime-adapt/source/versions.d ---------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Version ladder is `consecutiveTags`. Constraints are inferred: one
// closed interval per minor (v1.N.0 applies only to itself).
//
//===----------------------------------------------------------------------===//

module versions;

import std.array : split;
import std.conv : to;
import std.string : strip, startsWith;

/// One overlay pinned to a closed interval of LDC reference tags.
struct VersionConstraint
{
    string id;              /// overlay id, e.g. wasm-host-136
    string reference;       /// exact tag this overlay was taken from (v1.36.0)
    string referenceMin;    /// inclusive, tag or dotted version
    string referenceMax;    /// inclusive
    string adaptedRel;      /// path under tools/runtime-adapt/
    string note;
}

/// Consecutive minor tags (git refs in this LDC repo). The known ladder;
/// `--all-versions` uses `tagWindow` of the last `defaultVersionWindow`
/// tags ending at `latestMinorTag()` (slides when this list grows).
immutable string[] consecutiveTags = [
    "v1.30.0", "v1.31.0", "v1.32.0", "v1.33.0", "v1.34.0", "v1.35.0",
    "v1.36.0", "v1.37.0", "v1.38.0", "v1.39.0", "v1.40.0", "v1.41.0", "v1.42.0",
];

/// One closed interval per minor, inferred from `consecutiveTags`.
immutable VersionConstraint[] minorLadder = inferMinorLadder();

immutable VersionConstraint[] builtinConstraints = minorLadder;

private VersionConstraint[] inferMinorLadder()
{
    VersionConstraint[] cs;
    foreach (tag; consecutiveTags)
        cs ~= VersionConstraint("adapt-" ~ tag, tag, tag, tag, "", "inferred " ~ tag);
    return cs;
}

/// Default `--all-versions` / `--prefetch-refs` span (LDC-style last-N minors).
enum defaultVersionWindow = 12;

string latestMinorTag()
{
    return consecutiveTags[$ - 1];
}

/// Last `window` tags on the ladder, ending at `latest` (or the ladder tip).
string[] tagWindow(size_t window = defaultVersionWindow, string latest = "")
{
    if (!window)
        return null;
    auto end = latest.length ? latest : latestMinorTag();
    size_t hi = size_t.max;
    foreach (i, t; consecutiveTags)
        if (t == end || versionKey(t) == versionKey(end))
            hi = i;
    if (hi == size_t.max)
        return [end];
    auto lo = (hi + 1 >= window) ? hi + 1 - window : 0;
    return consecutiveTags[lo .. hi + 1].dup;
}

/// Inclusive range on the ladder (`from`/`to` may be outside; clipped).
string[] tagRange(string from, string to)
{
    if (!from.length && !to.length)
        return tagWindow();
    if (!from.length)
        from = consecutiveTags[0];
    if (!to.length)
        to = latestMinorTag();
    auto a = versionKey(from);
    auto b = versionKey(to);
    if (b < a)
    {
        auto tmp = a;
        a = b;
        b = tmp;
    }
    string[] outp;
    foreach (t; consecutiveTags)
    {
        auto k = versionKey(t);
        if (k >= a && k <= b)
            outp ~= t;
    }
    return outp;
}

/// `v1.36.0`, `v1.36.0..v1.42.0`, `latest`, `window`, `window:8`.
string[] parseTagSpec(string spec)
{
    auto s = spec.strip;
    if (!s.length || s == "latest")
        return [latestMinorTag()];
    if (s == "window")
        return tagWindow();
    if (s.length > 7 && s[0 .. 7] == "window:")
    {
        auto n = parseLeadingInt(s[7 .. $]);
        return tagWindow(n > 0 ? n : defaultVersionWindow);
    }
    foreach (sep; ["..", "..."])
    {
        auto i = indexOfSep(s, sep);
        if (i != size_t.max)
            return tagRange(s[0 .. i], s[i + sep.length .. $]);
    }
    return [s];
}

private size_t indexOfSep(string s, string sep)
{
    if (s.length < sep.length)
        return size_t.max;
    foreach (i; 0 .. s.length - sep.length + 1)
        if (s[i .. i + sep.length] == sep)
            return i;
    return size_t.max;
}

int versionKey(string tag)
{
    auto t = tag.strip;
    if (t.startsWith("v") || t.startsWith("V"))
        t = t[1 .. $];
    // v1.36.0 / 1.36.0-beta1 → 1_36_00
    int maj, mino, pat;
    auto parts = t.split(".");
    if (parts.length > 0)
        maj = parseLeadingInt(parts[0]);
    if (parts.length > 1)
        mino = parseLeadingInt(parts[1]);
    if (parts.length > 2)
        pat = parseLeadingInt(parts[2]);
    return maj * 10000 + mino * 100 + pat;
}

private int parseLeadingInt(string s)
{
    size_t i;
    while (i < s.length && s[i] >= '0' && s[i] <= '9')
        ++i;
    if (i == 0)
        return 0;
    return to!int(s[0 .. i]);
}

bool appliesTo(const VersionConstraint c, string referenceTag)
{
    auto k = versionKey(referenceTag);
    auto lo = versionKey(c.referenceMin.length ? c.referenceMin : c.reference);
    auto hi = versionKey(c.referenceMax.length ? c.referenceMax : c.reference);
    if (hi < lo)
    {
        auto tmp = lo;
        lo = hi;
        hi = tmp;
    }
    return k >= lo && k <= hi;
}

/// Constraints are inferred from `consecutiveTags`; `toolDir` is unused.
VersionConstraint[] loadConstraints(string = "")
{
    return minorLadder.dup;
}

const(VersionConstraint)* matchConstraint(const VersionConstraint[] cs, string referenceTag)
{
    const(VersionConstraint)* exact;
    const(VersionConstraint)* range;
    foreach (ref c; cs)
    {
        if (c.reference == referenceTag)
            exact = &c;
        else if (appliesTo(c, referenceTag))
            range = &c;
    }
    if (exact !is null)
        return exact;
    return range;
}

unittest
{
    assert(versionKey("v1.36.0") == 1 * 10000 + 36 * 100);
    assert(versionKey("1.42.0") > versionKey("v1.36.0"));
    assert(versionKey("v1.36.0-beta1") == versionKey("v1.36.0"));
    auto c = VersionConstraint("x", "v1.36.0", "v1.36.0", "v1.36.99", "overlays/x", "");
    assert(appliesTo(c, "v1.36.0"));
    assert(appliesTo(c, "v1.36.1"));
    assert(!appliesTo(c, "v1.42.0"));
    assert(!appliesTo(c, "v1.35.0"));
    assert(consecutiveTags.length == 13);
    assert(consecutiveTags[0] == "v1.30.0");
    assert(consecutiveTags[$ - 1] == "v1.42.0");
    assert(latestMinorTag() == "v1.42.0");
    auto w = tagWindow();
    assert(w.length == defaultVersionWindow);
    assert(w[0] == "v1.31.0");
    assert(w[$ - 1] == "v1.42.0");
    assert(tagWindow(3) == ["v1.40.0", "v1.41.0", "v1.42.0"]);
    auto w40 = tagWindow(12, "v1.40.0");
    assert(w40[$ - 1] == "v1.40.0");
    assert(w40[0] == "v1.30.0");
    assert(w40.length == 11); // ladder starts at v1.30.0
    assert(tagRange("v1.40.0", "v1.42.0") == ["v1.40.0", "v1.41.0", "v1.42.0"]);
    assert(parseTagSpec("latest") == ["v1.42.0"]);
    assert(parseTagSpec("window:3") == ["v1.40.0", "v1.41.0", "v1.42.0"]);
    assert(parseTagSpec("v1.36.0..v1.38.0") == ["v1.36.0", "v1.37.0", "v1.38.0"]);
    foreach (i, tag; consecutiveTags)
    {
        auto m = matchConstraint(minorLadder, tag);
        assert(m !is null && m.reference == tag);
        if (i + 1 < consecutiveTags.length)
            assert(!appliesTo(*m, consecutiveTags[i + 1]));
    }
}
