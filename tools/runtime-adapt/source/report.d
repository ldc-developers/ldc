//===-- tools/runtime-adapt/source/report.d -----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module report;

import classify;
import justify;
import kernel;
import versions;

import std.algorithm : sort;
import std.array : appender, array;
import std.format : format;
import std.json;

string renderMarkdown(const Inventory inv, const UpgradeRow[] upgrade,
    string reference, string adapted, string target,
    const VersionConstraint* constraint)
{
    auto c = countKinds(inv.rows);
    auto buf = appender!string();
    buf.put("# Adapted runtime vs LDC reference\n\n");
    buf.put("Parser: **libdparse** (serve-d / D-Scanner).\n\n");
    buf.put(format("- reference: `%s`\n- adapted: `%s`\n- target: `%s`\n",
        reference, adapted, target));
    if (constraint !is null)
        buf.put(format("- constraint: **%s** (`%s` … `%s`) — %s\n",
            constraint.id, constraint.referenceMin, constraint.referenceMax,
            constraint.note));
    buf.put("\n## Reconstruct counts\n\n");
    buf.put("| kind | n | generate |\n|---|---:|---|\n");
    buf.put(format("| identical | %s | copy-reference |\n", c.identical));
    buf.put(format("| adapted | %s | copy-adapted |\n", c.adapted));
    buf.put(format("| stub-adapt | %s | copy-adapted |\n", c.stubAdapt));
    buf.put(format("| extra | %s | copy-adapted |\n", c.extra));
    buf.put(format("| missing | %s | omit |\n", c.missing));
    buf.put(format("| **total** | **%s** | |\n\n", inv.rows.length));

    buf.put("## LDC concepts that prompt adaptations\n\n");
    foreach (j; catalog)
    {
        buf.put(format("### %s — %s\n\n", groupName(j.group), j.conceptName));
        buf.put(j.summary ~ "\n\n");
        buf.put(justifyLociMarkdown(j.group));
        buf.put("\n");
    }

    buf.put("## Overlay files (adapted / stub-adapt / extra)\n\n");
    buf.put("| path | kind | ref B | ours B | prompt |\n|---|---|---:|---:|---|\n");
    foreach (r; inv.rows)
    {
        if (r.kind == Kind.missing || r.kind == Kind.identical)
            continue;
        buf.put(format("| `%s` | %s | %s | %s | %s |\n",
            r.rel, kindName(r.kind), r.refBytes, r.adaptedBytes,
            r.prompt.replacePipes));
    }

    if (upgrade.length)
    {
        buf.put("\n## Upgrade map (reference → target)\n\n");
        int[string] vc;
        foreach (u; upgrade)
            vc[u.verdict] = vc.get(u.verdict, 0) + 1;
        buf.put("| verdict | n |\n|---|---:|\n");
        auto ks = vc.byKey.array;
        ks.sort;
        foreach (k; ks)
            buf.put(format("| %s | %s |\n", k, vc[k]));
        buf.put("\n");
        int n;
        foreach (u; upgrade)
        {
            if (u.verdict != "reapply-adapt" && u.verdict != "rebase-from-target"
                && u.verdict != "still-omit")
                continue;
            buf.put(format("- `%s` — **%s** — %s\n", u.rel, u.verdict, u.note));
            if (++n >= 60)
            {
                buf.put("\n_(truncated)_\n");
                break;
            }
        }
    }

    if (inv.parseFailures.length)
    {
        buf.put("\n## Parse failures (libdparse)\n\n");
        foreach (e; inv.parseFailures)
            buf.put("- " ~ e ~ "\n");
    }
    return buf.data;
}

private string replacePipes(string s)
{
    import std.array : replace;
    return s.replace("|", "/");
}

string renderJson(const Inventory inv, const UpgradeRow[] upgrade,
    const VersionConstraint* constraint)
{
    JSONValue root;
    auto c = countKinds(inv.rows);
    root["counts"] = JSONValue([
        "identical": JSONValue(c.identical),
        "adapted": JSONValue(c.adapted),
        "stubAdapt": JSONValue(c.stubAdapt),
        "extra": JSONValue(c.extra),
        "missing": JSONValue(c.missing),
        "total": JSONValue(cast(int) inv.rows.length),
    ]);
    if (constraint !is null)
    {
        JSONValue cj;
        cj["id"] = constraint.id;
        cj["reference"] = constraint.reference;
        cj["referenceMin"] = constraint.referenceMin;
        cj["referenceMax"] = constraint.referenceMax;
        root["constraint"] = cj;
    }
    JSONValue[] files;
    foreach (r; inv.rows)
    {
        JSONValue j;
        j["rel"] = r.rel;
        j["kind"] = kindName(r.kind);
        j["action"] = actionName(r.action);
        j["prompt"] = r.prompt;
        files ~= j;
    }
    root["files"] = JSONValue(files);
    JSONValue[] up;
    foreach (u; upgrade)
    {
        JSONValue j;
        j["rel"] = u.rel;
        j["verdict"] = u.verdict;
        j["note"] = u.note;
        up ~= j;
    }
    root["upgrade"] = JSONValue(up);
    return root.toPrettyString;
}
