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

cl::OptionCategory linkingCategory("Linking options");

/* Option parser that defaults to zero when no explicit number is given.
 * i.e.:  -cov    --> value = 0
 *        -cov=9  --> value = 9
 *        -cov=101 --> error, value must be in range [0..100]
 */
struct CoverageParser : public cl::parser<unsigned char> {
  explicit CoverageParser(cl::Option &O) : cl::parser<unsigned char>(O) {}

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

cl::opt<bool> compileOnly("c", cl::desc("Do not link"), cl::ZeroOrMore);

static cl::opt<bool, true> createStaticLib("lib", cl::ZeroOrMore,
                                           cl::desc("Create static library"),
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

static cl::opt<bool, true> verbose_cg_ast("vcg-ast", cl::ZeroOrMore, cl::Hidden,
                                          cl::desc("Write AST to .cg file"),
                                          cl::location(global.params.vcg_ast));

static cl::opt<unsigned, true> errorLimit(
    "verrors", cl::ZeroOrMore, cl::location(global.params.errorLimit),
    cl::desc("Limit the number of error messages (0 means unlimited)"));

static cl::opt<bool, true>
    showGaggedErrors("verrors-spec", cl::ZeroOrMore,
                     cl::location(global.params.showGaggedErrors),
                     cl::desc("Show errors from speculative compiles such as "
                              "__traits(compiles,...)"));

static cl::opt<ubyte, true> warnings(
    cl::desc("Warnings:"), cl::ZeroOrMore, cl::location(global.params.warnings),
    clEnumValues(
        clEnumValN(1, "w", "Enable warnings as errors (compilation will halt)"),
        clEnumValN(2, "wi",
                   "Enable warnings as messages (compilation will continue)")),
    cl::init(0));

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

cl::opt<bool> noAsm("noasm", cl::desc("Disallow use of inline assembler"),
                    cl::ZeroOrMore);

// Output file options
cl::opt<bool> dontWriteObj("o-", cl::desc("Do not write object file"),
                           cl::ZeroOrMore);

cl::opt<std::string> objectFile("of", cl::value_desc("filename"), cl::Prefix,
                                cl::desc("Use <filename> as output file name"),
                                cl::ZeroOrMore);

cl::opt<std::string> objectDir("od", cl::value_desc("directory"), cl::Prefix,
                               cl::desc("Write object files to <directory>"),
                               cl::ZeroOrMore);

cl::opt<std::string>
    soname("soname", cl::value_desc("soname"), cl::Hidden, cl::Prefix,
           cl::desc("Use <soname> as output shared library soname"),
           cl::ZeroOrMore);

// Output format options
cl::opt<bool> output_bc("output-bc", cl::desc("Write LLVM bitcode"),
                        cl::ZeroOrMore);

cl::opt<bool> output_ll("output-ll", cl::desc("Write LLVM IR"), cl::ZeroOrMore);

cl::opt<bool> output_s("output-s", cl::desc("Write native assembly"),
                       cl::ZeroOrMore);

cl::opt<cl::boolOrDefault> output_o("output-o", cl::ZeroOrMore,
                                    cl::desc("Write native object"));

static cl::opt<bool, true>
    cleanupObjectFiles("cleanup-obj", cl::ZeroOrMore, cl::ReallyHidden,
                       cl::desc("Remove generated object files on success"),
                       cl::location(global.params.cleanupObjectFiles));

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

// supported by DMD, but still undocumented
cl::list<std::string> jsonFields("Xi", cl::ReallyHidden, cl::value_desc("field"));

// Header generation options
static cl::opt<bool, true>
    doHdrGen("H", cl::desc("Generate 'header' file"), cl::ZeroOrMore,
             cl::location(global.params.doHdrGeneration));

cl::opt<std::string> hdrDir("Hd", cl::ZeroOrMore, cl::Prefix,
                            cl::desc("Write 'header' file to <directory>"),
                            cl::value_desc("directory"));

cl::opt<std::string> hdrFile("Hf", cl::ZeroOrMore, cl::Prefix,
                             cl::desc("Write 'header' file to <filename>"),
                             cl::value_desc("filename"));

cl::opt<bool>
    hdrKeepAllBodies("Hkeep-all-bodies", cl::ZeroOrMore,
                     cl::desc("Keep all function bodies in .di files"));

static cl::opt<bool, true> unittest("unittest", cl::ZeroOrMore,
                                    cl::desc("Compile in unit tests"),
                                    cl::location(global.params.useUnitTests));

cl::opt<std::string>
    cacheDir("cache",
             cl::desc("Enable compilation cache, using <cache dir> to "
                      "store cache files (experimental)"),
             cl::value_desc("cache dir"), cl::ZeroOrMore);

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
      global.params.useAssert = CHECKENABLEon;
      global.params.useArrayBounds = CHECKENABLEon;
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
    "d-debug", cl::location(dds), cl::CommaSeparated, cl::ValueOptional,
    cl::desc("Compile in debug code >= <level> or identified by <idents>"),
    cl::value_desc("level/idents"));

// -version is also declared in LLVM, so again we need to be a bit more verbose.
cl::list<std::string> versions(
    "d-version", cl::CommaSeparated, cl::value_desc("level/idents"),
    cl::desc("Compile in version code >= <level> or identified by <idents>"));

cl::list<std::string> transitions(
    "transition", cl::CommaSeparated, cl::value_desc("idents"),
    cl::desc(
        "Help with language change identified by <idents>, use ? for list"));

cl::list<std::string>
    linkerSwitches("L", cl::desc("Pass <linkerflag> to the linker"),
                   cl::value_desc("linkerflag"), cl::cat(linkingCategory),
                   cl::Prefix);

cl::list<std::string>
    ccSwitches("Xcc", cl::CommaSeparated,
               cl::desc("Pass <ccflag> to GCC/Clang for linking"),
               cl::value_desc("ccflag"), cl::cat(linkingCategory));

cl::opt<std::string>
    moduleDeps("deps", cl::ValueOptional, cl::ZeroOrMore,
               cl::value_desc("filename"),
               cl::desc("Write module dependencies to filename (only imports). "
                        "'-deps' alone prints module dependencies "
                        "(imports/file/version/debug/lib)"));

cl::opt<bool> m32bits("m32", cl::desc("32 bit target"), cl::ZeroOrMore);

cl::opt<bool> m64bits("m64", cl::desc("64 bit target"), cl::ZeroOrMore);

cl::opt<std::string> mTargetTriple("mtriple", cl::ZeroOrMore,
                                   cl::desc("Override target triple"));

cl::opt<std::string>
    mABI("mabi", cl::ZeroOrMore, cl::Hidden, cl::init(""),
         cl::desc("The name of the ABI to be targeted from the backend"));

static StringsAdapter
    modFileAliasStringsStore("mv", global.params.modFileAliasStrings);
static cl::list<std::string, StringsAdapter> modFileAliasStrings(
    "mv", cl::desc("Use <filespec> as source file for <package.module>"),
    cl::value_desc("<package.module>=<filespec>"),
    cl::location(modFileAliasStringsStore));

cl::list<std::string> includeModulePatterns(
    "i", cl::desc("Include imported modules in the compilation"),
    cl::value_desc("pattern"),
    cl::ValueOptional); // DMD allows omitting a value with special meaning

// Storage for the dynamically created float-abi option.
FloatABI::Type floatABI;

static cl::opt<CHECKENABLE, true, FlagParser<CHECKENABLE>>
    asserts("asserts", cl::ZeroOrMore, cl::desc("(*) Enable assertions"),
            cl::value_desc("bool"), cl::location(global.params.useAssert),
            cl::init(CHECKENABLEdefault));

static cl::opt<CHECKENABLE, true> boundsCheck(
    "boundscheck", cl::ZeroOrMore, cl::desc("Array bounds check"),
    cl::location(global.params.useArrayBounds), cl::init(CHECKENABLEdefault),
    clEnumValues(clEnumValN(CHECKENABLEoff, "off", "Disabled"),
                 clEnumValN(CHECKENABLEsafeonly, "safeonly",
                            "Enabled for @safe functions only"),
                 clEnumValN(CHECKENABLEon, "on", "Enabled for all functions")));

static cl::opt<bool, true, FlagParser<bool>>
    invariants("invariants", cl::ZeroOrMore, cl::desc("(*) Enable invariants"),
               cl::location(global.params.useInvariants), cl::init(true));

static cl::opt<bool, true, FlagParser<bool>> preconditions(
    "preconditions", cl::ZeroOrMore, cl::location(global.params.useIn),
    cl::desc("(*) Enable function preconditions"), cl::init(true));

static cl::opt<bool, true, FlagParser<bool>>
    postconditions("postconditions", cl::ZeroOrMore,
                   cl::location(global.params.useOut), cl::init(true),
                   cl::desc("(*) Enable function postconditions"));

static MultiSetter ContractsSetter(false, &global.params.useIn,
                                   &global.params.useOut, nullptr);
static cl::opt<MultiSetter, true, FlagParser<bool>>
    contracts("contracts", cl::ZeroOrMore, cl::location(ContractsSetter),
              cl::desc("(*) Enable function pre- and post-conditions"));

bool invReleaseMode = true;
static MultiSetter ReleaseSetter(true, &invReleaseMode,
                                 &global.params.useInvariants,
                                 &global.params.useOut, &global.params.useIn,
                                 nullptr);
static cl::opt<MultiSetter, true, cl::parser<bool>>
    release("release", cl::ZeroOrMore, cl::location(ReleaseSetter),
            cl::desc("Disables asserts, invariants, contracts and boundscheck"),
            cl::ValueDisallowed);

cl::opt<bool, true>
    singleObj("singleobj", cl::desc("Create only a single output object file"),
              cl::ZeroOrMore, cl::location(global.params.oneobj));

cl::opt<uint32_t, true> hashThreshold(
    "hash-threshold", cl::ZeroOrMore, cl::location(global.params.hashThreshold),
    cl::desc("Hash symbol names longer than this threshold (experimental)"));

cl::opt<bool> linkonceTemplates(
    "linkonce-templates", cl::ZeroOrMore,
    cl::desc(
        "Use linkonce_odr linkage for template symbols instead of weak_odr"));

cl::opt<bool> disableLinkerStripDead(
    "disable-linker-strip-dead", cl::ZeroOrMore,
    cl::desc("Do not try to remove unused symbols during linking"),
    cl::cat(linkingCategory));

// Math options
bool fFastMath; // Storage for the dynamically created ffast-math option.
llvm::FastMathFlags defaultFMF;
void setDefaultMathOptions(llvm::TargetOptions &targetOptions) {
  if (fFastMath) {
#if LDC_LLVM_VER >= 600
    defaultFMF.setFast();
#else
    defaultFMF.setUnsafeAlgebra();
#endif
    targetOptions.UnsafeFPMath = true;
  }
}

cl::opt<bool, true>
    allinst("allinst", cl::ZeroOrMore, cl::location(global.params.allInst),
            cl::desc("Generate code for all template instantiations"));

cl::opt<unsigned, true> nestedTemplateDepth(
    "template-depth", cl::ZeroOrMore, cl::location(global.params.nestedTmpl),
    cl::init(500),
    cl::desc(
        "Set maximum number of nested template instantiations (experimental)"));

cl::opt<bool, true>
    useDIP25("dip25", cl::ZeroOrMore, cl::location(global.params.useDIP25),
             cl::desc("Implement http://wiki.dlang.org/DIP25 (experimental)"));

cl::opt<bool> useDIP1000(
    "dip1000", cl::ZeroOrMore,
    cl::desc("Implement http://wiki.dlang.org/DIP1000 (experimental)"));

cl::opt<bool, true> useDIP1008("dip1008", cl::ZeroOrMore,
                               cl::location(global.params.ehnogc),
                               cl::desc("Implement DIP1008 (experimental)"));

cl::opt<bool, true> betterC(
    "betterC", cl::ZeroOrMore, cl::location(global.params.betterC),
    cl::desc("Omit generating some runtime information and helper functions"));

cl::opt<unsigned char, true, CoverageParser> coverageAnalysis(
    "cov", cl::ZeroOrMore, cl::location(global.params.covPercent),
    cl::desc("Compile-in code coverage analysis\n(use -cov=n for n% "
             "minimum required coverage)"),
    cl::ValueOptional, cl::init(127));

#if LDC_LLVM_VER >= 309
cl::opt<LTOKind> ltoMode(
    "flto", cl::ZeroOrMore, cl::desc("Set LTO mode, requires linker support"),
    cl::init(LTO_None),
    clEnumValues(
        clEnumValN(LTO_Full, "full", "Merges all input into a single module"),
        clEnumValN(LTO_Thin, "thin",
                   "Parallel importing and codegen (faster than 'full')")));
#endif

#if LDC_LLVM_VER >= 400
cl::opt<std::string>
    saveOptimizationRecord("fsave-optimization-record",
                           cl::value_desc("filename"),
                           cl::desc("Generate a YAML optimization record file "
                                    "of optimizations performed by LLVM"),
                           cl::ValueOptional);
#endif

#if LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX
cl::list<std::string>
    dcomputeTargets("mdcompute-targets", cl::CommaSeparated,
                    cl::desc("Generates code for the specified DCompute target"
                             " list. Use 'ocl-xy0' for OpenCL x.y, and "
                             "'cuda-xy0' for CUDA CC x.y"),
                     cl::value_desc("targets"));
cl::opt<std::string>
    dcomputeFilePrefix("mdcompute-file-prefix",
                       cl::desc("Prefix to prepend to the generated kernel files."),
                       cl::init("kernels"),
                       cl::value_desc("prefix"));
#endif

#if defined(LDC_DYNAMIC_COMPILE)
cl::opt<bool> enableDynamicCompile(
    "enable-dynamic-compile",
    cl::desc("Enable dynamic compilation"),
    cl::init(false));

cl::opt<bool> dynamicCompileTlsWorkaround(
    "dynamic-compile-tls-workaround",
    cl::desc("Enable dynamic compilation TLS workaround"),
    cl::init(true),
    cl::Hidden);
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
  llvm::StringMap<cl::Option *> &map = cl::getRegisteredOptions();

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
  renameAndHide("float-abi", "llvm-float-abi");

  // Step 2. Add the LDC options.
  new cl::opt<bool, true, FlagParser<bool>>(
      "color", cl::ZeroOrMore, cl::location(global.params.color),
      cl::desc("(*) Force colored console output"));
  new cl::opt<bool, true>("ffast-math", cl::ZeroOrMore, cl::location(fFastMath),
                          cl::desc("Set @fastmath for all functions."));
  new cl::opt<FloatABI::Type, true>(
      "float-abi", cl::desc("ABI/operations to use for floating-point types:"),
      cl::ZeroOrMore, cl::location(floatABI), cl::init(FloatABI::Default),
      clEnumValues(
          clEnumValN(FloatABI::Default, "default",
                     "Target default floating-point ABI"),
          clEnumValN(FloatABI::Soft, "soft",
                     "Software floating-point ABI and operations"),
          clEnumValN(
              FloatABI::SoftFP, "softfp",
              "Soft-float ABI, but hardware floating-point instructions"),
          clEnumValN(FloatABI::Hard, "hard",
                     "Hardware floating-point ABI and instructions")));
}

/// Hides command line options exposed from within LLVM that are unlikely
/// to be useful for end users from the -help output.
void hideLLVMOptions() {
  static const char *const hiddenOptions[] = {
      "aarch64-neon-syntax", "addrsig", "arm-add-build-attributes",
      "arm-implicit-it", "asm-instrumentation", "asm-show-inst",
      "atomic-counter-update-promoted", "bounds-checking-single-trap",
      "code-model", "cost-kind", "cppfname", "cppfor", "cppgen",
      "cvp-dont-process-adds", "debug-counter", "debugger-tune",
      "denormal-fp-math", "disable-debug-info-verifier",
      "disable-objc-arc-checkforcfghazards", "disable-spill-fusing",
      "do-counter-promotion", "emulated-tls", "enable-correct-eh-support",
      "enable-fp-mad", "enable-implicit-null-checks", "enable-load-pre",
      "enable-misched", "enable-name-compression", "enable-no-infs-fp-math",
      "enable-no-nans-fp-math", "enable-no-signed-zeros-fp-math",
      "enable-no-trapping-fp-math", "enable-objc-arc-annotations",
      "enable-objc-arc-opts", "enable-pie", "enable-scoped-noalias",
      "enable-tbaa", "enable-unsafe-fp-math", "exception-model",
      "exhaustive-register-search", "expensive-combines",
      "fatal-assembler-warnings", "filter-print-funcs", "gpsize",
      "imp-null-check-page-size", "imp-null-max-insts-to-consider",
      "import-all-index", "incremental-linker-compatible",
      "instcombine-guard-widening-window", "instcombine-max-num-phis",
      "instcombine-maxarray-size", "internalize-public-api-file",
      "internalize-public-api-list", "iterative-counter-promotion",
      "join-liveintervals", "jump-table-type", "limit-float-precision",
      "max-counter-promotions", "max-counter-promotions-per-loop",
      "mc-relax-all", "mc-x86-disable-arith-relaxation", "meabi",
      "memop-size-large", "memop-size-range", "merror-missing-parenthesis",
      "merror-noncontigious-register", "mfuture-regs", "mips-compact-branches",
      "mips16-constant-islands", "mips16-hard-float", "mlsm", "mno-compound",
      "mno-fixup", "mno-ldc1-sdc1", "mno-pairing", "mwarn-missing-parenthesis",
      "mwarn-noncontigious-register", "mwarn-sign-mismatch",
      "no-discriminators", "nozero-initialized-in-bss", "nvptx-sched4reg",
      "objc-arc-annotation-target-identifier", "pie-copy-relocations",
      "polly-dump-after", "polly-dump-after-file", "polly-dump-before",
      "polly-dump-before-file", "pre-RA-sched", "print-after-all",
      "print-before-all", "print-machineinstrs", "print-module-scope",
      "profile-estimator-loop-weight", "profile-estimator-loop-weight",
      "profile-file", "profile-info-file", "profile-verifier-noassert",
      "r600-ir-structurize", "rdf-dump", "rdf-limit", "recip", "regalloc",
      "relax-elf-relocations", "rewrite-map-file", "rng-seed",
      "safepoint-ir-verifier-print-only",
      "sample-profile-check-record-coverage",
      "sample-profile-check-sample-coverage",
      "sample-profile-inline-hot-threshold",
      "sample-profile-max-propagate-iterations", "shrink-wrap", "simplify-mir",
      "speculative-counter-promotion-max-exiting",
      "speculative-counter-promotion-to-loop", "spiller", "spirv-debug",
      "spirv-erase-cl-md", "spirv-mem2reg", "spvbool-validate",
      "stack-alignment", "stack-size-section", "stack-symbol-ordering",
      "stackmap-version", "static-func-full-module-prefix",
      "static-func-strip-dirname-prefix", "stats", "stats-json", "strip-debug",
      "struct-path-tbaa", "summary-file", "tailcallopt", "thread-model",
      "time-passes", "unfold-element-atomic-memcpy-max-elements",
      "unique-section-names", "unit-at-a-time", "use-ctors",
      "verify-debug-info", "verify-dom-info", "verify-loop-info",
      "verify-loop-lcssa", "verify-machine-dom-info", "verify-regalloc",
      "verify-region-info", "verify-scev", "verify-scev-maps",
      "vp-counters-per-site", "vp-static-alloc", "x86-early-ifcvt",
      "x86-recip-refinement-steps", "x86-use-vzeroupper",

      // We enable -fdata-sections/-ffunction-sections by default where it makes
      // sense for reducing code size, so hide them to avoid confusion.
      //
      // We need our own switch as these two are defined by LLVM and linked to
      // static TargetMachine members, but the default we want to use depends
      // on the target triple (and thus we do not know it until after the
      // command
      // line has been parsed).
      "fdata-sections", "ffunction-sections", "data-sections",
      "function-sections"};

  // pulled in from shared LLVM headers, but unused or not desired in LDC
  static const char *const removedOptions[] = {"disable-tail-calls",
                                               "fatal-warnings",
                                               "filetype",
                                               "no-deprecated-warn",
                                               "no-warn",
                                               "stackrealign",
                                               "start-after",
                                               "stop-after",
                                               "trap-func",
                                               "W"};

  llvm::StringMap<cl::Option *> &map = cl::getRegisteredOptions();
  for (const auto name : hiddenOptions) {
    // Check if option exists first for resilience against LLVM changes
    // between versions.
    auto it = map.find(name);
    if (it != map.end()) {
      it->second->setHiddenFlag(cl::Hidden);
    }
  }

  for (const auto name : removedOptions) {
    auto it = map.find(name);
    if (it != map.end()) {
      map.erase(it);
    }
  }
}

} // namespace opts
