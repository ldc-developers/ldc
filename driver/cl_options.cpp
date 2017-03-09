//===-- cl_options.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/cl_options.h"
#include "mars.h"
#include "gen/cl_helpers.h"
#include "gen/logger.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Operator.h"

namespace opts {

// This vector is filled by parseCommandLine in main.cpp.
llvm::SmallVector<const char *, 32> allArguments;

/* Option parser that defaults to zero when no explicit number is given.
 * i.e.:  -cov    --> value = 0
 *        -cov=9  --> value = 9
 *        -cov=101 --> error, value must be in range [0..100]
 */
struct CoverageParser : public cl::parser<unsigned char> {
#if LDC_LLVM_VER >= 307
  explicit CoverageParser(cl::Option &O) : cl::parser<unsigned char>(O) {}
#endif

  bool parse(cl::Option &O, llvm::StringRef /*ArgName*/, llvm::StringRef Arg,
             unsigned char &Val) {
    if (Arg == "") {
      Val = 0;
      return false;
    }

    if (Arg.getAsInteger(0, Val)) {
      return O.error("'" + Arg +
                     "' value invalid for required coverage percentage");
    }

    if (Val > 100) {
      return O.error("Required coverage percentage must be <= 100");
    }
    return false;
  }
};

// Positional options first, in order:
cl::list<std::string> fileList(cl::Positional, cl::desc("files"));

cl::list<std::string> runargs(
    "run",
    cl::desc(
        "Runs the resulting program, passing the remaining arguments to it"),
    cl::Positional, cl::PositionalEatsArgs);

cl::opt<bool> invokedByLDMD("ldmd", cl::desc("Invoked by LDMD?"),
                            cl::ZeroOrMore, cl::ReallyHidden);

static cl::opt<ubyte, true> useDeprecated(
    cl::desc("Allow deprecated code/language features:"), cl::ZeroOrMore,
    clEnumValues(clEnumValN(0, "de", "Do not allow deprecated features"),
                 clEnumValN(1, "d", "Silently allow deprecated features"),
                 clEnumValN(2, "dw",
                            "Warn about the use of deprecated features")),
    cl::location(global.params.useDeprecated), cl::init(2));

cl::opt<bool, true>
    enforcePropertySyntax("property", cl::desc("Enforce property syntax"),
                          cl::ZeroOrMore, cl::ReallyHidden,
                          cl::location(global.params.enforcePropertySyntax));

cl::opt<bool> compileOnly("c", cl::desc("Do not link"), cl::ZeroOrMore);

static cl::opt<bool, true> createStaticLib("lib",
                                           cl::desc("Create static library"),
                                           cl::ZeroOrMore,
                                           cl::location(global.params.lib));

static cl::opt<bool, true>
    createSharedLib("shared", cl::desc("Create shared library (DLL)"),
                    cl::ZeroOrMore, cl::location(global.params.dll));

static cl::opt<bool, true> verbose("v", cl::desc("Verbose"), cl::ZeroOrMore,
                                   cl::location(global.params.verbose));

static cl::opt<bool, true>
    vcolumns("vcolumns",
             cl::desc("Print character (column) numbers in diagnostics"),
             cl::ZeroOrMore, cl::location(global.params.showColumns));

static cl::opt<bool, true>
    vgc("vgc", cl::desc("List all gc allocations including hidden ones"),
        cl::ZeroOrMore, cl::location(global.params.vgc));

static cl::opt<bool, true> verbose_cg("v-cg", cl::desc("Verbose codegen"),
                                      cl::ZeroOrMore,
                                      cl::location(global.params.verbose_cg));

static cl::opt<unsigned, true> errorLimit(
    "verrors", cl::ZeroOrMore,
    cl::desc("Limit the number of error messages (0 means unlimited)"),
    cl::location(global.errorLimit));

static cl::opt<bool, true>
    showGaggedErrors("verrors-spec", cl::ZeroOrMore,
                     cl::location(global.params.showGaggedErrors),
                     cl::desc("Show errors from speculative compiles such as "
                              "__traits(compiles,...)"));

static cl::opt<ubyte, true> warnings(
    cl::desc("Warnings:"), cl::ZeroOrMore,
    clEnumValues(
        clEnumValN(1, "w", "Enable warnings as errors (compilation will halt)"),
        clEnumValN(2, "wi",
                   "Enable warnings as messages (compilation will continue)")),
    cl::location(global.params.warnings), cl::init(0));

static cl::opt<bool, true> ignoreUnsupportedPragmas(
    "ignore", cl::desc("Ignore unsupported pragmas"), cl::ZeroOrMore,
    cl::location(global.params.ignoreUnsupportedPragmas));

static cl::opt<ubyte, true> debugInfo(
    cl::desc("Generating debug information:"), cl::ZeroOrMore,
    clEnumValues(
        clEnumValN(1, "g", "Add symbolic debug info"),
        clEnumValN(2, "gc",
                   "Add symbolic debug info, optimize for non D debuggers"),
        clEnumValN(3, "gline-tables-only", "Add line tables only")),
    cl::location(global.params.symdebug), cl::init(0));

static cl::opt<unsigned, true>
    dwarfVersion("dwarf-version", cl::desc("Dwarf version"),
                 cl::location(global.params.dwarfVersion), cl::init(0),
                 cl::Hidden);

cl::opt<bool> noAsm("noasm", cl::desc("Disallow use of inline assembler"));

// Output file options
cl::opt<bool> dontWriteObj("o-", cl::desc("Do not write object file"),
                           cl::ZeroOrMore);

cl::opt<std::string> objectFile("of", cl::value_desc("filename"), cl::Prefix,
                                cl::desc("Use <filename> as output file name"),
                                cl::ZeroOrMore);

cl::opt<std::string> objectDir("od", cl::value_desc("directory"), cl::Prefix,
                               cl::ZeroOrMore,
                               cl::desc("Write object files to <directory>"));

cl::opt<std::string>
    soname("soname", cl::value_desc("soname"), cl::Hidden, cl::Prefix,
           cl::desc("Use <soname> as output shared library soname"));

// Output format options
cl::opt<bool> output_bc("output-bc", cl::desc("Write LLVM bitcode"),
                        cl::ZeroOrMore);

cl::opt<bool> output_ll("output-ll", cl::desc("Write LLVM IR"), cl::ZeroOrMore);

cl::opt<bool> output_s("output-s", cl::desc("Write native assembly"),
                       cl::ZeroOrMore);

cl::opt<cl::boolOrDefault> output_o("output-o", cl::ZeroOrMore,
                                    cl::desc("Write native object"));

static cl::opt<bool, true>
    cleanupObjectFiles("cleanup-obj",
                       cl::desc("Remove generated object files on success"),
                       cl::ZeroOrMore, cl::ReallyHidden,
                       cl::location(global.params.cleanupObjectFiles));

// Disabling Red Zone
cl::opt<bool, true>
    disableRedZone("disable-red-zone",
                   cl::desc("Do not emit code that uses the red zone."),
                   cl::location(global.params.disableRedZone), cl::init(false));

// DDoc options
static cl::opt<bool, true> doDdoc("D", cl::desc("Generate documentation"),
                                  cl::location(global.params.doDocComments),
                                  cl::ZeroOrMore);

cl::opt<std::string>
    ddocDir("Dd", cl::desc("Write documentation file to <directory>"),
            cl::value_desc("directory"), cl::Prefix, cl::ZeroOrMore);

cl::opt<std::string>
    ddocFile("Df", cl::desc("Write documentation file to <filename>"),
             cl::value_desc("filename"), cl::Prefix, cl::ZeroOrMore);

// Json options
static cl::opt<bool, true> doJson("X", cl::desc("Generate JSON file"),
                                  cl::ZeroOrMore,
                                  cl::location(global.params.doJsonGeneration));

cl::opt<std::string> jsonFile("Xf", cl::desc("Write JSON file to <filename>"),
                              cl::value_desc("filename"), cl::Prefix,
                              cl::ZeroOrMore);

// Header generation options
static cl::opt<bool, true>
    doHdrGen("H", cl::desc("Generate 'header' file"), cl::ZeroOrMore,
             cl::location(global.params.doHdrGeneration));

cl::opt<std::string> hdrDir("Hd",
                            cl::desc("Write 'header' file to <directory>"),
                            cl::value_desc("directory"), cl::Prefix,
                            cl::ZeroOrMore);

cl::opt<std::string> hdrFile("Hf", cl::ZeroOrMore,
                             cl::desc("Write 'header' file to <filename>"),
                             cl::value_desc("filename"), cl::Prefix);

cl::opt<bool>
    hdrKeepAllBodies("Hkeep-all-bodies",
                     cl::desc("Keep all function bodies in .di files"),
                     cl::ZeroOrMore);

static cl::opt<bool, true> unittest("unittest", cl::ZeroOrMore,
                                    cl::desc("Compile in unit tests"),
                                    cl::location(global.params.useUnitTests));

cl::opt<std::string>
    cacheDir("cache", cl::desc("Enable compilation cache, using <cache dir> to "
                               "store cache files (experimental)"),
             cl::value_desc("cache dir"));

static StringsAdapter strImpPathStore("J", global.params.fileImppath);
static cl::list<std::string, StringsAdapter> stringImportPaths(
    "J", cl::desc("Look for string imports also in <directory>"),
    cl::value_desc("directory"), cl::location(strImpPathStore), cl::Prefix);

static cl::opt<bool, true>
    addMain("main", cl::desc("Add default main() (e.g. for unittesting)"),
            cl::ZeroOrMore, cl::location(global.params.addMain));

// -d-debug is a bit messy, it has 3 modes:
// -d-debug=ident, -d-debug=level and -d-debug (without argument)
// That last of these must be acted upon immediately to ensure proper
// interaction with other options, so it needs some special handling:
std::vector<std::string> debugArgs;

struct D_DebugStorage {
  void push_back(const std::string &str) {
    if (str.empty()) {
      // Bare "-d-debug" has a special meaning.
      global.params.useAssert = true;
      global.params.useArrayBounds = BOUNDSCHECKon;
      global.params.useInvariants = true;
      global.params.useIn = true;
      global.params.useOut = true;
      debugArgs.push_back("1");
    } else {
      debugArgs.push_back(str);
    }
  }
};

static D_DebugStorage dds;

// -debug is already declared in LLVM (at least, in debug builds),
// so we need to be a bit more verbose.
static cl::list<std::string, D_DebugStorage> debugVersionsOption(
    "d-debug",
    cl::desc("Compile in debug code >= <level> or identified by <idents>"),
    cl::value_desc("level/idents"), cl::location(dds), cl::CommaSeparated,
    cl::ValueOptional);

// -version is also declared in LLVM, so again we need to be a bit more verbose.
cl::list<std::string> versions(
    "d-version",
    cl::desc("Compile in version code >= <level> or identified by <idents>"),
    cl::value_desc("level/idents"), cl::CommaSeparated);

cl::list<std::string> transitions(
    "transition",
    cl::desc(
        "Help with language change identified by <idents>, use ? for list"),
    cl::value_desc("idents"), cl::CommaSeparated);

static StringsAdapter linkSwitchStore("L", global.params.linkswitches);
static cl::list<std::string, StringsAdapter>
    linkerSwitches("L", cl::desc("Pass <linkerflag> to the linker"),
                   cl::value_desc("linkerflag"), cl::location(linkSwitchStore),
                   cl::Prefix);

cl::opt<std::string>
    moduleDeps("deps",
               cl::desc("Write module dependencies to filename (only imports). "
                        "'-deps' alone prints module dependencies "
                        "(imports/file/version/debug/lib)"),
               cl::value_desc("filename"), cl::ValueOptional);

cl::opt<std::string> mArch("march",
                           cl::desc("Architecture to generate code for:"));

cl::opt<bool> m32bits("m32", cl::desc("32 bit target"), cl::ZeroOrMore);

cl::opt<bool> m64bits("m64", cl::desc("64 bit target"), cl::ZeroOrMore);

cl::opt<std::string>
    mCPU("mcpu",
         cl::desc("Target a specific cpu type (-mcpu=help for details)"),
         cl::value_desc("cpu-name"), cl::init(""));

cl::list<std::string>
    mAttrs("mattr", cl::CommaSeparated,
           cl::desc("Target specific attributes (-mattr=help for details)"),
           cl::value_desc("a1,+a2,-a3,..."));

cl::opt<std::string> mTargetTriple("mtriple",
                                   cl::desc("Override target triple"));

#if LDC_LLVM_VER >= 307
cl::opt<std::string>
    mABI("mabi",
         cl::desc("The name of the ABI to be targeted from the backend"),
         cl::Hidden, cl::init(""));
#endif

cl::opt<llvm::Reloc::Model> mRelocModel(
    "relocation-model", cl::desc("Relocation model"), cl::ZeroOrMore,
#if LDC_LLVM_VER < 309
    cl::init(llvm::Reloc::Default),
#endif
    clEnumValues(
#if LDC_LLVM_VER < 309
        clEnumValN(llvm::Reloc::Default, "default",
                   "Target default relocation model"),
#endif
        clEnumValN(llvm::Reloc::Static, "static", "Non-relocatable code"),
        clEnumValN(llvm::Reloc::PIC_, "pic",
                   "Fully relocatable, position independent code"),
        clEnumValN(llvm::Reloc::DynamicNoPIC, "dynamic-no-pic",
                   "Relocatable external references, non-relocatable code")));

cl::opt<llvm::CodeModel::Model> mCodeModel(
    "code-model", cl::desc("Code model"), cl::init(llvm::CodeModel::Default),
    clEnumValues(
        clEnumValN(llvm::CodeModel::Default, "default",
                   "Target default code model"),
        clEnumValN(llvm::CodeModel::Small, "small", "Small code model"),
        clEnumValN(llvm::CodeModel::Kernel, "kernel", "Kernel code model"),
        clEnumValN(llvm::CodeModel::Medium, "medium", "Medium code model"),
        clEnumValN(llvm::CodeModel::Large, "large", "Large code model")));

cl::opt<FloatABI::Type> mFloatABI(
    "float-abi", cl::desc("ABI/operations to use for floating-point types:"),
    cl::init(FloatABI::Default),
    clEnumValues(
        clEnumValN(FloatABI::Default, "default",
                   "Target default floating-point ABI"),
        clEnumValN(FloatABI::Soft, "soft",
                   "Software floating-point ABI and operations"),
        clEnumValN(FloatABI::SoftFP, "softfp",
                   "Soft-float ABI, but hardware floating-point instructions"),
        clEnumValN(FloatABI::Hard, "hard",
                   "Hardware floating-point ABI and instructions")));

cl::opt<bool>
    disableFpElim("disable-fp-elim", cl::ZeroOrMore,
                  cl::desc("Disable frame pointer elimination optimization"),
                  cl::init(false));

static cl::opt<bool, true, FlagParser<bool>>
    asserts("asserts", cl::desc("(*) Enable assertions"),
            cl::value_desc("bool"), cl::location(global.params.useAssert),
            cl::init(true));

cl::opt<BOUNDSCHECK> boundsCheck(
    "boundscheck", cl::desc("Array bounds check"),
    clEnumValues(clEnumValN(BOUNDSCHECKoff, "off", "Disabled"),
                 clEnumValN(BOUNDSCHECKsafeonly, "safeonly",
                            "Enabled for @safe functions only"),
                 clEnumValN(BOUNDSCHECKon, "on", "Enabled for all functions")),
    cl::init(BOUNDSCHECKdefault));

static cl::opt<bool, true, FlagParser<bool>>
    invariants("invariants", cl::desc("(*) Enable invariants"),
               cl::location(global.params.useInvariants), cl::init(true));

static cl::opt<bool, true, FlagParser<bool>>
    preconditions("preconditions",
                  cl::desc("(*) Enable function preconditions"),
                  cl::location(global.params.useIn), cl::init(true));

static cl::opt<bool, true, FlagParser<bool>>
    postconditions("postconditions",
                   cl::desc("(*) Enable function postconditions"),
                   cl::location(global.params.useOut), cl::init(true));

static MultiSetter ContractsSetter(false, &global.params.useIn,
                                   &global.params.useOut, NULL);
static cl::opt<MultiSetter, true, FlagParser<bool>>
    contracts("contracts",
              cl::desc("(*) Enable function pre- and post-conditions"),
              cl::location(ContractsSetter));

bool nonSafeBoundsChecks = true;
static MultiSetter ReleaseSetter(true, &global.params.useAssert,
                                 &nonSafeBoundsChecks,
                                 &global.params.useInvariants,
                                 &global.params.useOut, &global.params.useIn,
                                 NULL);
static cl::opt<MultiSetter, true, cl::parser<bool>>
    release("release",
            cl::desc("Disables asserts, invariants, contracts and boundscheck"),
            cl::location(ReleaseSetter), cl::ValueDisallowed);

cl::opt<bool, true>
    singleObj("singleobj", cl::desc("Create only a single output object file"),
              cl::location(global.params.oneobj));

cl::opt<uint32_t, true> hashThreshold(
    "hash-threshold",
    cl::desc("Hash symbol names longer than this threshold (experimental)"),
    cl::location(global.params.hashThreshold), cl::init(0));

cl::opt<bool> linkonceTemplates(
    "linkonce-templates",
    cl::desc(
        "Use linkonce_odr linkage for template symbols instead of weak_odr"),
    cl::ZeroOrMore);

cl::opt<bool> disableLinkerStripDead(
    "disable-linker-strip-dead",
    cl::desc("Do not try to remove unused symbols during linking"),
    cl::init(false));

// Math options
bool fFastMath; // Storage for the dynamically created ffast-math option.
llvm::FastMathFlags defaultFMF;
void setDefaultMathOptions(llvm::TargetMachine &target) {
  if (fFastMath) {
    defaultFMF.setUnsafeAlgebra();

    llvm::TargetOptions &TO = target.Options;
    TO.UnsafeFPMath = true;
  }
}

cl::opt<bool, true>
    allinst("allinst", cl::ZeroOrMore,
            cl::desc("Generate code for all template instantiations"),
            cl::location(global.params.allInst));

cl::opt<unsigned, true> nestedTemplateDepth(
    "template-depth", cl::location(global.params.nestedTmpl), cl::init(500),
    cl::desc(
        "Set maximum number of nested template instantiations (experimental)"));

cl::opt<bool, true>
    useDIP25("dip25", cl::ZeroOrMore,
             cl::desc("Implement http://wiki.dlang.org/DIP25 (experimental)"),
             cl::location(global.params.useDIP25));

cl::opt<bool, true> betterC(
    "betterC", cl::ZeroOrMore,
    cl::desc("Omit generating some runtime information and helper functions"),
    cl::location(global.params.betterC));

cl::opt<unsigned char, true, CoverageParser> coverageAnalysis(
    "cov", cl::desc("Compile-in code coverage analysis\n(use -cov=n for n% "
                    "minimum required coverage)"),
    cl::location(global.params.covPercent), cl::ValueOptional, cl::init(127));

#if LDC_WITH_PGO
cl::opt<std::string>
    genfileInstrProf("fprofile-instr-generate", cl::value_desc("filename"),
                     cl::desc("Generate instrumented code to collect a runtime "
                              "profile into default.profraw (overriden by "
                              "'=<filename>' or LLVM_PROFILE_FILE env var)"),
                     cl::ValueOptional);

cl::opt<std::string> usefileInstrProf(
    "fprofile-instr-use", cl::value_desc("filename"),
    cl::desc("Use instrumentation data for profile-guided optimization"),
    cl::ValueRequired);
#endif

cl::opt<bool>
    instrumentFunctions("finstrument-functions",
                        cl::desc("Instrument function entry and exit with "
                                 "GCC-compatible profiling calls"));

#if LDC_LLVM_VER >= 309
cl::opt<LTOKind> ltoMode(
    "flto", cl::desc("Set LTO mode, requires linker support"),
    cl::init(LTO_None),
    clEnumValues(
        clEnumValN(LTO_Full, "full", "Merges all input into a single module"),
        clEnumValN(LTO_Thin, "thin",
                   "Parallel importing and codegen (faster than 'full')")));
#endif

static cl::extrahelp footer(
    "\n"
    "-d-debug can also be specified without options, in which case it enables "
    "all\n"
    "debug checks (i.e. (asserts, boundschecks, contracts and invariants) as "
    "well\n"
    "as acting as -d-debug=1\n\n"
    "Options marked with (*) also have a -disable-FOO variant with inverted\n"
    "meaning.\n");

/// Create commandline options that may clash with LLVM's options (depending on
/// LLVM version and on LLVM configuration), and that thus cannot be created
/// using static construction.
/// The clashing LLVM options are suffixed with "llvm-" and hidden from the
/// -help output.
void createClashingOptions() {
#if LDC_LLVM_VER >= 307
  llvm::StringMap<cl::Option *> &map = cl::getRegisteredOptions();
#else
  llvm::StringMap<cl::Option *> map;
  cl::getRegisteredOptions(map);
#endif

  auto renameAndHide = [&map](const char *from, const char *to) {
    auto i = map.find(from);
    if (i != map.end()) {
      cl::Option *opt = i->getValue();
      map.erase(i);
      opt->setArgStr(to);
      opt->setHiddenFlag(cl::Hidden);
      map[to] = opt;
    }
  };

  // Step 1. Hide the clashing LLVM options.
  // LLVM 3.7 introduces compiling as shared library. The result
  // is a clash in the command line options.
  renameAndHide("color", "llvm-color");
  renameAndHide("ffast-math", "llvm-ffast-math");

  // Step 2. Add the LDC options.
  new cl::opt<bool, true, FlagParser<bool>>(
      "color", cl::desc("Force colored console output"),
      cl::location(global.params.color));
  new cl::opt<bool, true>(
      "ffast-math", cl::desc("Set @fastmath for all functions."),
      cl::location(fFastMath), cl::init(false), cl::ZeroOrMore);
}

/// Hides command line options exposed from within LLVM that are unlikely
/// to be useful for end users from the -help output.
void hideLLVMOptions() {
  static const char *const hiddenOptions[] = {
      "bounds-checking-single-trap", "disable-debug-info-verifier",
      "disable-objc-arc-checkforcfghazards", "disable-spill-fusing", "cppfname",
      "cppfor", "cppgen", "enable-correct-eh-support", "enable-load-pre",
      "enable-implicit-null-checks", "enable-misched",
      "enable-objc-arc-annotations", "enable-objc-arc-opts",
      "enable-scoped-noalias", "enable-tbaa", "exhaustive-register-search",
      "fatal-assembler-warnings", "gpsize", "imp-null-check-page-size",
      "internalize-public-api-file", "internalize-public-api-list",
      "join-liveintervals", "limit-float-precision",
      "mc-x86-disable-arith-relaxation", "merror-missing-parenthesis",
      "merror-noncontigious-register", "mfuture-regs", "mips-compact-branches",
      "mips16-constant-islands", "mips16-hard-float", "mlsm", "mno-compound",
      "mno-fixup", "mno-ldc1-sdc1", "mno-pairing", "mwarn-missing-parenthesis",
      "mwarn-noncontigious-register", "mwarn-sign-mismatch", "nvptx-sched4reg",
      "no-discriminators", "objc-arc-annotation-target-identifier",
      "pre-RA-sched", "print-after-all", "print-before-all",
      "print-machineinstrs", "profile-estimator-loop-weight",
      "profile-estimator-loop-weight", "profile-file", "profile-info-file",
      "profile-verifier-noassert", "r600-ir-structurize", "rdf-dump",
      "rdf-limit", "regalloc", "rewrite-map-file", "rng-seed",
      "sample-profile-max-propagate-iterations", "shrink-wrap", "spiller",
      "stackmap-version", "stats", "strip-debug", "struct-path-tbaa",
      "time-passes", "unit-at-a-time", "verify-debug-info", "verify-dom-info",
      "verify-loop-info", "verify-machine-dom-info", "verify-regalloc",
      "verify-region-info", "verify-scev", "verify-scev-maps",
      "x86-early-ifcvt", "x86-use-vzeroupper", "x86-recip-refinement-steps",

      // We enable -fdata-sections/-ffunction-sections by default where it makes
      // sense for reducing code size, so hide them to avoid confusion.
      //
      // We need our own switch as these two are defined by LLVM and linked to
      // static TargetMachine members, but the default we want to use depends
      // on the target triple (and thus we do not know it until after the
      // command
      // line has been parsed).
      "fdata-sections", "ffunction-sections"};

#if LDC_LLVM_VER >= 307
  llvm::StringMap<cl::Option *> &map = cl::getRegisteredOptions();
#else
  llvm::StringMap<cl::Option *> map;
  cl::getRegisteredOptions(map);
#endif

  for (const auto name : hiddenOptions) {
    // Check if option exists first for resilience against LLVM changes
    // between versions.
    auto it = map.find(name);
    if (it != map.end()) {
      it->second->setHiddenFlag(cl::Hidden);
    }
  }
}

} // namespace opts
