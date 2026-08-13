//===-- tools/runtime-adapt/tests/iterate.d -----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module tests.iterate;

import adapt;
import emit;
import iterate;
import kernel;
import ldcmods;
import parseutil;
import paths;
import principles;
import resolve;
import versions;
import walk;

import std.algorithm : canFind;
import std.file : exists, readText, tempDir, rmdirRecurse;
import std.path : buildPath;

private string thisLdc()
{
    auto ldc = findLdcRoot();
    if (isLdcCheckout(ldc))
        return ldc;
    return "";
}

unittest
{
    assert(principleTable.length >= 4);
    assert(selectAction("core/thread/osthread.d", KernelGroup.thread,
        ParseFacts.init, ParseFacts.init) == EmitAction.takeReference);
    assert(selectAction("object.d", KernelGroup.none,
        ParseFacts.init, ParseFacts.init) == EmitAction.takeReference);
    assert(selectAction("core/thread/osthread.d", KernelGroup.thread,
        ParseFacts.init, ParseFacts.init, EmitProfile.strip) == EmitAction.omit);
    assert(selectAction("object.d", KernelGroup.none,
        ParseFacts.init, ParseFacts.init, EmitProfile.strip) == EmitAction.astOverlay);
    assert(remapRel("core/thread/fiber.d") == "core/thread/fiber/package.d");
    assert(inverseRemap("core/thread/fiber/package.d") == "core/thread/fiber.d");
}

unittest
{
    assert(versioningHolds("v1.36.0"));
    assert(!appliesTo(*matchConstraint(minorLadder, "v1.36.0"), "v1.37.0"));
    assert(tagRange("v1.40.0", "v1.42.0") == ["v1.40.0", "v1.41.0", "v1.42.0"]);
    assert(parseTagSpec("v1.36.0..v1.38.0").length == 3);
    auto win = tagWindow(defaultVersionWindow);
    assert(win.length == defaultVersionWindow);
    assert(win[$ - 1] == latestMinorTag());
}

unittest
{
    // Product path: emit this checkout (stock walk + compiler ldc/*).
    auto ldc = thisLdc();
    if (!ldc.length)
        return;
    auto outDir = buildPath(tempDir, "ldc-runtime-adapt-generate-head");
    if (exists(outDir))
        rmdirRecurse(outDir);
    auto stock = walkStockRuntime(ldc);
    assert(stock.length > 4);
    auto rep = iterateVersion(ldc, "HEAD", ldc, outDir, 1);
    assert(rep.emit.omitted == 0, renderIterate(rep));
    assert(rep.emit.sourcedFromRef == stock.length, renderIterate(rep));
    assert(rep.emit.sourcedFromCompiler >= 4, renderIterate(rep));
    assert(rep.emit.productComplete, renderIterate(rep));
    assert(rep.emit.parseFail == 0, renderIterate(rep));
    assert(rep.equivalent, renderIterate(rep));
    assert(exists(buildPath(outDir, "object.d")));
    assert(exists(buildPath(outDir, "ldc", "intrinsics.di")));
    assert(exists(buildPath(outDir, "ldc", "llvmasm.di")));
    assert(exists(buildPath(outDir, "ldc", "simd.di")));
    assert(exists(buildPath(outDir, "ldc", "attributes.d")));
    assert(!exists(buildPath(outDir, "ldc", "attribute.d")));
    assert(!exists(buildPath(outDir, "ldc", "dylib.d")));
    auto attr = readText(buildPath(outDir, "ldc", "attributes.d"));
    assert(attr.canFind("immutable weak") || attr.canFind("enum weak"));
    assert(!attr.canFind("struct weak"));
    assert(attr.canFind("sizeArgIdx"));
    assert(attr.canFind("numArgIdx = int.min"));
    auto inn = readText(buildPath(outDir, "ldc", "intrinsics.di"));
    assert(!inn.canFind("llvm.metadata"));
    assert(!inn.canFind("llvm.ldc.classinfo"));
    assert(inn.canFind("llvm_memcpy(") || inn.canFind("llvm_memcpy(T)"));
    assert(!inn.canFind("llvm_memcpy_p0"));
    auto asmSrc = readText(buildPath(outDir, "ldc", "llvmasm.di"));
    assert(asmSrc.canFind("__irEx"));
    auto eh = readText(buildPath(outDir, "ldc", "eh_msvc.d"));
    assert(eh.canFind("Throwable _d_eh_enter_catch"));
}

unittest
{
    auto ldc = thisLdc();
    if (!ldc.length)
        return;
    auto files = generateFromCompiler(ldc, "HEAD");
    assert(files.length >= 4);
    bool sawInn, sawAttr, sawJunk;
    foreach (f; files)
    {
        if (f.rel == "ldc/intrinsics.di")
        {
            sawInn = true;
            assert(!f.body.canFind("llvm.metadata"));
        }
        if (f.rel == "ldc/attributes.d")
            sawAttr = true;
        if (f.rel == "ldc/attribute.d" || f.rel == "ldc/dylib.d")
            sawJunk = true;
    }
    assert(sawInn && sawAttr);
    assert(!sawJunk);
}
