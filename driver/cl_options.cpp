//===-- cl_options.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/cl_options.h"

#include "gen/logger.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"

namespace opts {

// This vector is filled by parseCommandLine in main.cpp.
llvm::SmallVector<const char *, 32> allArguments;

cl::OptionCategory linkingCategory("Linking options");

// Positional options first, in order:
cl::list<std::string> fileList(cl::Positional, cl::desc("files"));

cl::list<std::string> runargs(
    "run",
    cl::desc(
        "Runs the resulting program, passing the remaining arguments to it"),
    cl::Positional, cl::PositionalEatsArgs);

cl::opt<bool> invokedByLDMD("ldmd", cl::desc("Invoked by LDMD?"),
                            cl::ZeroOrMore, cl::ReallyHidden);

static cl::opt<Diagnostic, true> useDeprecated(
    cl::desc("Allow deprecated language features and symbols:"), cl::ZeroOrMore,
    cl::location(global.params.useDeprecated), cl::init(DIAGNOSTICinform),
    cl::values(
        clEnumValN(DIAGNOSTICoff, "d",
                   "Silently allow deprecated features and symbols"),
        clEnumValN(DIAGNOSTICinform, "dw",
                   "Issue a message when deprecated features or "
                   "symbols are used (default)"),
        clEnumValN(
            DIAGNOSTICerror, "de",
            "Issue an error when deprecated features or symbols are used "
            "(halt compilation)")));

cl::opt<bool> compileOnly("c", cl::desc("Compile only, do not link"),
                          cl::ZeroOrMore);

static cl::opt<bool, true> createStaticLib("lib", cl::ZeroOrMore,
                                           cl::desc("Create static library"),
                                           cl::location(global.params.lib));

static cl::opt<bool, true>
    createSharedLib("shared", cl::desc("Create shared library (DLL)"),
                    cl::ZeroOrMore, cl::location(global.params.dll));

cl::opt<SymbolVisibility> symbolVisibility(
    "fvisibility", cl::ZeroOrMore, cl::desc("Default visibility of symbols"),
    cl::init(SymbolVisibility::default_),
    cl::values(
        clEnumValN(
            SymbolVisibility::default_, "default",
            "Hidden for Windows targets without -shared, otherwise public"),
        clEnumValN(SymbolVisibility::hidden, "hidden",
                   "Only export symbols marked with 'export'"),
        clEnumValN(SymbolVisibility::public_, "public", "Export all symbols")));

cl::opt<DLLImport, true> dllimport(
    "dllimport", cl::ZeroOrMore, cl::location(global.params.dllimport),
    cl::desc("Windows only: which extern(D) global variables to dllimport "
             "implicitly if not defined in a root module"),
    cl::values(
        clEnumValN(DLLImport::none, "none",
                   "None (default with -link-defaultlib-shared=false)"),
        clEnumValN(DLLImport::defaultLibsOnly, "defaultLibsOnly",
                   "Only druntime/Phobos symbols (default with "
                   "-link-defaultlib-shared and -fvisibility=hidden)."),
        clEnumValN(DLLImport::all, "all",
                   "All (default with -link-defaultlib-shared and "
                   "-fvisibility=public)")));

static cl::opt<bool, true> verbose("v", cl::desc("Verbose"), cl::ZeroOrMore,
                                   cl::location(global.params.verbose));

static cl::opt<bool, true>
    vcolumns("vcolumns",
             cl::desc("Print character (column) numbers in diagnostics"),
             cl::ZeroOrMore, cl::location(global.params.showColumns));

static cl::opt<bool, true>
    vgc("vgc", cl::desc("List all gc allocations including hidden ones"),
        cl::ZeroOrMore, cl::location(global.params.vgc));

// Dummy data type for custom parsers where the help output shouldn't display
// any value.
using DummyDataType = bool;

// `-vtemplates[=list-instances]` parser.
struct VTemplatesParser : public cl::parser<DummyDataType> {
  explicit VTemplatesParser(cl::Option &O) : cl::parser<DummyDataType>(O) {}

  bool parse(cl::Option &O, llvm::StringRef /*ArgName*/, llvm::StringRef Arg,
             DummyDataType & /*Val*/) {
    global.params.vtemplates = true;

    if (Arg.empty()) {
      return false;
    }

    if (Arg == "list-instances") {
      global.params.vtemplatesListInstances = true;
      return false;
    }

    return O.error("unsupported value '" + Arg + "'");
  }
};

static cl::opt<DummyDataType, false, VTemplatesParser> vtemplates(
    "vtemplates", cl::ZeroOrMore, cl::ValueOptional,
    cl::desc("List statistics on template instantiations\n"
             "Use -vtemplates=list-instances to additionally show all "
             "instantiation contexts for each template"));

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

static cl::opt<bool, true> printErrorContext(
    "verrors-context", cl::ZeroOrMore,
    cl::location(global.params.printErrorContext),
    cl::desc(
        "Show error messages with the context of the erroring source line"));

static cl::opt<MessageStyle, true> verrorStyle(
    "verror-style", cl::ZeroOrMore, cl::location(global.params.messageStyle),
    cl::desc(
        "Set the style for file/line number annotations on compiler messages"),
    cl::values(
        clEnumValN(MessageStyle::digitalmars, "digitalmars",
                   "'file(line[,column]): message' (default)"),
        clEnumValN(MessageStyle::gnu, "gnu",
                   "'file:line[:column]: message', conforming to the GNU "
                   "standard used by gcc and clang")),
    cl::init(MessageStyle::digitalmars));

static cl::opt<unsigned, true>
    verrorSupplements("verror-supplements", cl::ZeroOrMore,
                      cl::location(global.params.errorSupplementLimit),
                      cl::desc("Limit the number of supplemental messages for "
                               "each error (0 means unlimited)"));

static cl::opt<Diagnostic, true> warnings(
    cl::desc("Warnings:"), cl::ZeroOrMore, cl::location(global.params.warnings),
    cl::values(
        clEnumValN(DIAGNOSTICerror, "w",
                   "Enable warnings as errors (compilation will halt)"),
        clEnumValN(DIAGNOSTICinform, "wi",
                   "Enable warnings as messages (compilation will continue)")),
    cl::init(DIAGNOSTICoff));

static cl::opt<bool, true> warningsObsolete(
    "wo", cl::ZeroOrMore, cl::location(global.params.obsolete),
    cl::desc("Enable warnings about use of obsolete features"));

static cl::opt<bool, true> ignoreUnsupportedPragmas(
    "ignore", cl::desc("Ignore unsupported pragmas"), cl::ZeroOrMore,
    cl::location(global.params.ignoreUnsupportedPragmas));

static cl::opt<CppStdRevision, true> cplusplus(
    "extern-std", cl::ZeroOrMore,
    cl::desc("C++ standard for name mangling compatibility"),
    cl::location(global.params.cplusplus),
    cl::values(
        clEnumValN(CppStdRevisionCpp98, "c++98",
                   "Sets `__traits(getTargetInfo, \"cppStd\")` to `199711`"),
        clEnumValN(
            CppStdRevisionCpp11, "c++11",
            "Sets `__traits(getTargetInfo, \"cppStd\")` to `201103` (default)"),
        clEnumValN(CppStdRevisionCpp14, "c++14",
                   "Sets `__traits(getTargetInfo, \"cppStd\")` to `201402`"),
        clEnumValN(CppStdRevisionCpp17, "c++17",
                   "Sets `__traits(getTargetInfo, \"cppStd\")` to `201703`"),
        clEnumValN(CppStdRevisionCpp20, "c++20",
                   "Sets `__traits(getTargetInfo, \"cppStd\")` to `202002`")));

static cl::opt<unsigned char, true> debugInfo(
    cl::desc("Generating debug information:"), cl::ZeroOrMore,
    cl::values(
        clEnumValN(1, "g", "Add symbolic debug info"),
        clEnumValN(2, "gc",
                   "Add symbolic debug info, optimize for non D debuggers"),
        clEnumValN(3, "gline-tables-only", "Add line tables only")),
    cl::location(global.params.symdebug), cl::init(0));

cl::opt<bool> emitDwarfDebugInfo(
    "gdwarf", cl::ZeroOrMore,
    cl::desc("Emit DWARF debuginfo (instead of CodeView) for MSVC targets"));

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

cl::opt<bool> output_mlir("output-mlir", cl::desc("Write MLIR"),
    cl::ZeroOrMore);

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
                                  cl::location(global.params.ddoc.doOutput),
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
                                  cl::location(global.params.json.doOutput));

cl::opt<std::string> jsonFile("Xf", cl::desc("Write JSON file to <filename>"),
                              cl::value_desc("filename"), cl::Prefix,
                              cl::ZeroOrMore);

// supported by DMD, but still undocumented
cl::list<std::string> jsonFields("Xi", cl::ReallyHidden, cl::value_desc("field"));

// Header generation options
static cl::opt<bool, true>
    doHdrGen("H", cl::desc("Generate 'header' file"), cl::ZeroOrMore,
             cl::location(global.params.dihdr.doOutput));

cl::opt<std::string> hdrDir("Hd", cl::ZeroOrMore, cl::Prefix,
                            cl::desc("Write 'header' file to <directory>"),
                            cl::value_desc("directory"));

cl::opt<std::string> hdrFile("Hf", cl::ZeroOrMore, cl::Prefix,
                             cl::desc("Write 'header' file to <filename>"),
                             cl::value_desc("filename"));

cl::opt<bool>
    hdrKeepAllBodies("Hkeep-all-bodies", cl::ZeroOrMore,
                     cl::desc("Keep all function bodies in .di files"));

// C++ header generation options

// `-HC[=silent|verbose]` parser. Required for defaulting to `silent`.
struct HCParser : public cl::parser<DummyDataType> {
  explicit HCParser(cl::Option &O) : cl::parser<DummyDataType>(O) {}

  bool parse(cl::Option &O, llvm::StringRef /*ArgName*/, llvm::StringRef Arg,
             DummyDataType & /*Val*/) {
    global.params.cxxhdr.doOutput = true;

    if (Arg.empty() || Arg == "silent") {
      return false;
    }

    if (Arg == "verbose") {
      global.params.cxxhdr.fullOutput = true;
      return false;
    }

    return O.error("unsupported value '" + Arg + "'");
  }
};

static cl::opt<DummyDataType, false, HCParser>
    doCxxHdrGen("HC", cl::ZeroOrMore, cl::ValueOptional,
                cl::desc("Generate C++ header file\n"
                         "Use -HC=verbose to add comments for ignored "
                         "declarations (e.g. extern(D))"));

cl::opt<std::string>
    cxxHdrDir("HCd", cl::ZeroOrMore, cl::Prefix,
              cl::desc("Write C++ 'header' file to <directory>"),
              cl::value_desc("directory"));

cl::opt<std::string>
    cxxHdrFile("HCf", cl::ZeroOrMore, cl::Prefix,
               cl::desc("Write C++ 'header' file to <filename>"),
               cl::value_desc("filename"));

cl::opt<std::string> mixinFile("mixin", cl::ZeroOrMore,
                               cl::desc("Expand and save mixins to <filename>"),
                               cl::value_desc("filename"));

static cl::opt<bool, true> unittest("unittest", cl::ZeroOrMore,
                                    cl::desc("Compile in unit tests"),
                                    cl::location(global.params.useUnitTests));

cl::opt<std::string>
    cacheDir("cache",
             cl::desc("Enable compilation cache, using <cache dir> to "
                      "store cache files"),
             cl::value_desc("cache dir"), cl::ZeroOrMore);

static StringsAdapter strImpPathStore("J", global.params.fileImppath);
static cl::list<std::string, StringsAdapter> stringImportPaths(
    "J", cl::desc("Look for string imports also in <directory>"),
    cl::value_desc("directory"), cl::location(strImpPathStore), cl::Prefix);

static cl::opt<bool, true> addMain(
    "main", cl::ZeroOrMore, cl::location(global.params.addMain),
    cl::desc(
        "Add default main() if not present already (e.g. for unittesting)"));

// -d-debug is a bit messy, it has 3 modes:
// -d-debug=ident, -d-debug=level and -d-debug (without argument)
// The last one represents `-d-debug=1`, so it needs some special handling:
std::vector<std::string> debugArgs;

struct D_DebugStorage {
  void push_back(const std::string &str) {
    debugArgs.push_back(str.empty() ? "1" : str);
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
    "transition", cl::CommaSeparated, cl::value_desc("name"),
    cl::desc("Help with language change identified by <name>, use ? for list"));

cl::list<std::string>
    previews("preview", cl::CommaSeparated, cl::value_desc("name"),
             cl::desc("Enable an upcoming language change "
                      "identified by <name>, use ? for list"));

cl::list<std::string> reverts(
    "revert", cl::CommaSeparated, cl::value_desc("name"),
    cl::desc("Revert language change identified by <name>, use ? for list"));

cl::list<std::string>
    linkerSwitches("L", cl::desc("Pass <linkerflag> to the linker"),
                   cl::value_desc("linkerflag"), cl::cat(linkingCategory),
                   cl::Prefix);

cl::list<std::string> ccSwitches(
    "Xcc", cl::value_desc("ccflag"), cl::cat(linkingCategory),
    cl::desc("Pass <ccflag> to GCC/Clang for linking/preprocessing"));

cl::list<std::string> cppSwitches("P", cl::value_desc("cppflag"), cl::Prefix,
                                  cl::desc("Pass <cppflag> to C preprocessor"));

cl::opt<std::string> moduleDeps(
    "deps", cl::ValueOptional, cl::ZeroOrMore, cl::value_desc("filename"),
    cl::desc("Write module dependencies to <filename> (only imports). "
             "'-deps' alone prints module dependencies "
             "(imports/file/version/debug/lib)"));

cl::opt<std::string>
    makeDeps("makedeps", cl::ValueOptional, cl::ZeroOrMore,
             cl::value_desc("filename"),
             cl::desc("Write module dependencies in Makefile compatible format "
                      "to <filename>/stdout (only imports)"));

cl::opt<bool> m32bits("m32", cl::desc("32 bit target"), cl::ZeroOrMore);

cl::opt<bool> m64bits("m64", cl::desc("64 bit target"), cl::ZeroOrMore);

cl::opt<std::string> mTargetTriple("mtriple", cl::ZeroOrMore,
                                   cl::desc("Override target triple"));

cl::opt<std::string>
    mABI("mabi", cl::ZeroOrMore, cl::init(""),
         cl::desc("The name of the ABI to be targeted from the backend"));

static Strings *pModFileAliasStrings = &global.params.modFileAliasStrings;
static StringsAdapter
    modFileAliasStringsStore("mv", pModFileAliasStrings);
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
    cl::values(clEnumValN(CHECKENABLEoff, "off", "Disabled"),
               clEnumValN(CHECKENABLEsafeonly, "safeonly",
                          "Enabled for @safe functions only"),
               clEnumValN(CHECKENABLEon, "on", "Enabled for all functions")));

static cl::opt<CHECKENABLE, true, FlagParser<CHECKENABLE>> switchErrors(
    "switch-errors", cl::ZeroOrMore,
    cl::desc("(*) Enable runtime errors for unhandled switch cases"),
    cl::location(global.params.useSwitchError), cl::init(CHECKENABLEdefault));

static cl::opt<CHECKENABLE, true, FlagParser<CHECKENABLE>>
    invariants("invariants", cl::ZeroOrMore, cl::desc("(*) Enable invariants"),
               cl::location(global.params.useInvariants),
               cl::init(CHECKENABLEdefault));

static cl::opt<CHECKENABLE, true, FlagParser<CHECKENABLE>>
    preconditions("preconditions", cl::ZeroOrMore,
                  cl::location(global.params.useIn),
                  cl::desc("(*) Enable function preconditions"),
                  cl::init(CHECKENABLEdefault));

static cl::opt<CHECKENABLE, true, FlagParser<CHECKENABLE>>
    postconditions("postconditions", cl::ZeroOrMore,
                   cl::location(global.params.useOut),
                   cl::init(CHECKENABLEdefault),
                   cl::desc("(*) Enable function postconditions"));

static MultiSetter ContractsSetter(false, &global.params.useIn,
                                   &global.params.useOut, nullptr);
static cl::opt<MultiSetter, true, FlagParser<bool>>
    contracts("contracts", cl::ZeroOrMore, cl::location(ContractsSetter),
              cl::desc("(*) Enable function pre- and post-conditions"));

static cl::opt<CHECKACTION, true> checkAction(
    "checkaction", cl::ZeroOrMore, cl::location(global.params.checkAction),
    cl::desc("Action to take when an assert/boundscheck/final-switch fails"),
    cl::init(CHECKACTION_D),
    cl::values(
        clEnumValN(CHECKACTION_D, "D",
                   "Usual D behavior of throwing an AssertError"),
        clEnumValN(CHECKACTION_C, "C",
                   "Call the C runtime library assert failure function"),
        clEnumValN(CHECKACTION_halt, "halt",
                   "Halt the program execution (very lightweight)"),
        clEnumValN(CHECKACTION_context, "context",
                   "Use D assert with context information (when available)")));

static cl::opt<bool, true>
    release("release", cl::ZeroOrMore, cl::location(global.params.release),
            cl::desc("Compile release version, defaulting to disabled "
                     "asserts/contracts/invariants, and bounds checks in @safe "
                     "functions only"),
            cl::ValueDisallowed);

cl::opt<bool, true>
    singleObj("singleobj", cl::desc("Create only a single output object file"),
              cl::ZeroOrMore, cl::location(global.params.oneobj));

cl::opt<uint32_t, true> hashThreshold(
    "hash-threshold", cl::ZeroOrMore, cl::location(global.params.hashThreshold),
    cl::desc("Hash symbol names longer than this threshold (experimental)"));

static cl::opt<LinkonceTemplates, true> linkonceTemplates(
    cl::ZeroOrMore, cl::location(global.params.linkonceTemplates),
    cl::values(
        clEnumValN(LinkonceTemplates::yes, "linkonce-templates",
                   "Use discardable linkonce_odr linkage for template symbols "
                   "and lazily & recursively define all referenced "
                   "instantiated symbols in each object file"),
        clEnumValN(LinkonceTemplates::aggressive,
                   "linkonce-templates-aggressive",
                   "Experimental, more aggressive variant")));

cl::opt<bool> disableLinkerStripDead(
    "disable-linker-strip-dead", cl::ZeroOrMore,
    cl::desc("Do not try to remove unused symbols during linking"),
    cl::cat(linkingCategory));

cl::opt<bool> noPLT(
    "fno-plt", cl::ZeroOrMore,
    cl::desc("Do not use the PLT to make function calls"));

static cl::opt<signed char> passmanager("passmanager",
    cl::desc("Setting the passmanager (new,legacy):"), cl::ZeroOrMore,
    #if LDC_LLVM_VER < 1500
      cl::init(0),
    #else
      cl::init(1),
    #endif
    cl::values(
        clEnumValN(0, "legacy", "Use the legacy passmanager (available for LLVM14 and below) "),
        clEnumValN(1, "new", "Use the new passmanager (available for LLVM14 and above)")));
bool isUsingLegacyPassManager() { return passmanager == 0; }

// Math options
bool fFastMath; // Storage for the dynamically created ffast-math option.
llvm::FastMathFlags defaultFMF;
void setDefaultMathOptions(llvm::TargetOptions &targetOptions) {
  if (fFastMath) {
    defaultFMF.setFast();
    targetOptions.UnsafeFPMath = true;
  }
}

cl::opt<bool>
    fNoDiscardValueNames("fno-discard-value-names", cl::ZeroOrMore,
                         cl::desc("Do not discard value names in LLVM IR"));

cl::opt<bool> fNullPointerIsValid(
    "fno-delete-null-pointer-checks", cl::ZeroOrMore,
    cl::desc(
        "Treat null pointer dereference as defined behavior when optimizing "
        "(instead of _un_defined behavior). This prevents the optimizer from "
        "assuming that any dereferenced pointer must not have been null and "
        "optimize away the branches accordingly."));

cl::opt<bool>
    fSplitStack("fsplit-stack", cl::ZeroOrMore,
                cl::desc("Use segmented stack (see Clang documentation)"));

cl::opt<bool, true>
    allinst("allinst", cl::ZeroOrMore, cl::location(global.params.allInst),
            cl::desc("Generate code for all template instantiations"));

cl::opt<unsigned, true> nestedTemplateDepth(
    "template-depth", cl::ZeroOrMore, cl::location(global.recursionLimit),
    cl::init(500),
    cl::desc("Set maximum number of nested template instantiations"));

// legacy options superseded by `-preview=dip<N>`
cl::opt<bool> useDIP25("dip25", cl::ZeroOrMore, cl::ReallyHidden,
                       cl::desc("Implement DIP25 (sealed references)"));
cl::opt<bool> useDIP1000("dip1000", cl::ZeroOrMore, cl::ReallyHidden,
                         cl::desc("Implement DIP1000 (scoped pointers)"));
static cl::opt<bool, true>
    useDIP1008("dip1008", cl::ZeroOrMore, cl::location(global.params.ehnogc),
               cl::desc("Implement DIP1008 (@nogc Throwable)"),
               cl::ReallyHidden);

cl::opt<bool, true> betterC(
    "betterC", cl::ZeroOrMore, cl::location(global.params.betterC),
    cl::desc("Omit generating some runtime information and helper functions"));

// `-cov[=<n>|ctfe]` parser.
struct CoverageParser : public cl::parser<DummyDataType> {
  explicit CoverageParser(cl::Option &O) : cl::parser<DummyDataType>(O) {}

  bool parse(cl::Option &O, llvm::StringRef /*ArgName*/, llvm::StringRef Arg,
             DummyDataType & /*Val*/) {
    global.params.cov = true;

    if (Arg.empty()) {
      return false;
    }

    if (Arg == "ctfe") {
      global.params.ctfe_cov = true;
      return false;
    }

    unsigned char percent = 0;
    if (Arg.getAsInteger(0, percent)) {
      return O.error("'" + Arg +
                     "' value invalid for required coverage percentage");
    }

    if (percent > 100) {
      return O.error("required coverage percentage must be <= 100");
    }

    global.params.covPercent = percent;
    return false;
  }
};

static cl::opt<DummyDataType, false, CoverageParser> coverageAnalysis(
    "cov", cl::ZeroOrMore, cl::ValueOptional,
    cl::desc("Compile-in code coverage analysis and .lst file generation\n"
             "Use -cov=<n> for n% minimum required coverage\n"
             "Use -cov=ctfe to include code executed during CTFE"));

cl::opt<CoverageIncrement> coverageIncrement(
    "cov-increment", cl::ZeroOrMore,
    cl::desc("Set the type of coverage line count increment instruction"),
    cl::init(CoverageIncrement::_default),
    cl::values(clEnumValN(CoverageIncrement::_default, "default",
                          "Use the default (atomic)"),
               clEnumValN(CoverageIncrement::atomic, "atomic", "Atomic increment"),
               clEnumValN(CoverageIncrement::nonatomic, "non-atomic",
                          "Non-atomic increment (not thread safe)"),
               clEnumValN(CoverageIncrement::boolean, "boolean",
                          "Don't read, just set counter to 1")));

// Compilation time tracing options
cl::opt<bool> fTimeTrace(
    "ftime-trace", cl::ZeroOrMore,
    cl::desc("Turn on time profiler. Generates JSON file "
             "based on the output filename (also see --ftime-trace-file)."));
cl::opt<unsigned> fTimeTraceGranularity(
    "ftime-trace-granularity", cl::ZeroOrMore, cl::init(500),
    cl::desc(
        "Minimum time granularity (in microseconds) traced by time profiler"));
cl::opt<std::string>
fTimeTraceFile("ftime-trace-file",
               cl::desc("Specify time trace file destination"),
               cl::value_desc("filename"));

cl::opt<LTOKind> ltoMode(
    "flto", cl::ZeroOrMore, cl::desc("Set LTO mode, requires linker support"),
    cl::init(LTO_None),
    cl::values(
        clEnumValN(LTO_Full, "full", "Merges all input into a single module"),
        clEnumValN(LTO_Thin, "thin",
                   "Parallel importing and codegen (faster than 'full')")));

cl::opt<std::string>
    saveOptimizationRecord("fsave-optimization-record",
                           cl::value_desc("filename"),
                           cl::desc("Generate a YAML optimization record file "
                                    "of optimizations performed by LLVM"),
                           cl::ValueOptional);

#if LDC_LLVM_VER >= 1300
// LLVM < 13 has "--warn-stack-size", but let's not do the effort of forwarding
// the string to that option, and instead let the user do it himself.
cl::opt<unsigned>
    fWarnStackSize("fwarn-stack-size", cl::ZeroOrMore, cl::init(UINT_MAX),
                   cl::desc("Warn for stack size bigger than the given number"),
                   cl::value_desc("threshold"));
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

#if LDC_LLVM_VER >= 1400
bool enableOpaqueIRPointers = false;
#endif

static cl::extrahelp
    footer("\n"
           "-d-debug can also be specified without options, in which case it "
           "enables all debug checks (i.e. asserts, boundschecks, contracts "
           "and invariants) as well as acting as -d-debug=1.\n\n"
           "Boolean options can take an optional value, e.g., "
           "-link-defaultlib-shared=<true,false>.\n"
           "Boolean options marked with (*) also have a -disable-FOO variant "
           "with inverted meaning.\n");

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
      if (to) {
        opt->setArgStr(to);
        opt->setHiddenFlag(cl::Hidden);
        map[to] = opt;
      }
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
      cl::values(
          clEnumValN(FloatABI::Default, "default",
                     "Target default floating-point ABI"),
          clEnumValN(FloatABI::Soft, "soft",
                     "Software floating-point ABI and operations"),
          clEnumValN(
              FloatABI::SoftFP, "softfp",
              "Soft-float ABI, but hardware floating-point instructions"),
          clEnumValN(FloatABI::Hard, "hard",
                     "Hardware floating-point ABI and instructions")));

#if LDC_LLVM_VER >= 1400
  renameAndHide("opaque-pointers", nullptr); // remove
  new cl::opt<bool, true>(
      "opaque-pointers", cl::ZeroOrMore, cl::location(enableOpaqueIRPointers),
      cl::desc("Use opaque IR pointers (experimental!)"), cl::Hidden);
#endif
}

/// Hides command line options exposed from within LLVM that are unlikely
/// to be useful for end users from the -help output.
void hideLLVMOptions() {
  static const char *const hiddenOptions[] = {
      "aarch64-neon-syntax", "aarch64-use-aa",
      "abort-on-max-devirt-iterations-reached",
      "addrsig", "align-loops", "allow-ginsert-as-artifact",
      "amdgpu-bypass-slow-div", "amdgpu-disable-loop-alignment",
      "amdgpu-disable-power-sched", "amdgpu-dpp-combine",
      "amdgpu-dump-hsa-metadata", "amdgpu-enable-flat-scratch",
      "amdgpu-enable-global-sgpr-addr", "amdgpu-enable-merge-m0",
      "amdgpu-enable-power-sched", "amdgpu-igrouplp",
      "amdgpu-promote-alloca-to-vector-limit",
      "amdgpu-reserve-vgpr-for-sgpr-spill", "amdgpu-sdwa-peephole",
      "amdgpu-use-aa-in-codegen", "amdgpu-verify-hsa-metadata",
      "amdgpu-vgpr-index-mode", "arm-add-build-attributes",
      "arm-implicit-it", "asm-instrumentation", "asm-show-inst",
      "atomic-counter-update-promoted", "atomic-first-counter",
      "basic-block-sections",
      "basicblock-sections", "bounds-checking-single-trap",
      "cfg-hide-cold-paths",
      "cfg-hide-deoptimize-paths", "cfg-hide-unreachable-paths",
      "code-model", "cost-kind", "cppfname", "cppfor", "cppgen",
      "cvp-dont-add-nowrap-flags",
      "cvp-dont-process-adds", "debug-counter", "debug-entry-values",
      "debugger-tune", "debugify-func-limit", "debugify-level",
      "debugify-quiet", "debug-info-correlate",
      "denormal-fp-math", "denormal-fp-math-f32", "disable-debug-info-verifier",
      "disable-i2p-p2i-opt",
      "disable-objc-arc-checkforcfghazards", "disable-promote-alloca-to-lds",
      "disable-promote-alloca-to-vector", "disable-slp-vectorization",
      "disable-spill-fusing",
      "do-counter-promotion", "dot-cfg-mssa", "dwarf64", "emit-call-site-info",
      "emit-dwarf-unwind",
      "emscripten-cxx-exceptions-allowed",
      "emscripten-cxx-exceptions-whitelist",
      "emulated-tls", "enable-approx-func-fp-math", "enable-correct-eh-support",
      "enable-cse-in-irtranslator", "enable-cse-in-legalizer",
      "enable-emscripten-cxx-exceptions", "enable-emscripten-sjlj",
      "enable-fp-mad", "enable-gvn-hoist", "enable-gvn-memdep",
      "enable-gvn-sink", "enable-implicit-null-checks", "enable-jmc-instrument",
      "enable-load-in-loop-pre",
      "enable-load-pre", "enable-loop-simplifycfg-term-folding",
      "enable-misched", "enable-name-compression", "enable-no-infs-fp-math",
      "enable-no-nans-fp-math", "enable-no-signed-zeros-fp-math",
      "enable-no-trapping-fp-math", "enable-objc-arc-annotations",
      "enable-objc-arc-opts", "enable-pie", "enable-scoped-noalias",
      "enable-split-backedge-in-load-pre",
      "enable-tbaa", "enable-unsafe-fp-math", "exception-model",
      "exhaustive-register-search", "expensive-combines",
      "experimental-debug-variable-locations",
      "fatal-assembler-warnings", "filter-print-funcs",
      "force-dwarf-frame-section", "force-opaque-pointers",
      "fs-profile-debug-bw-threshold", "fs-profile-debug-prob-diff-threshold",
      "generate-merged-base-profiles",
      "gpsize", "hash-based-counter-split",
      "hot-cold-split", "ignore-xcoff-visibility",
      "imp-null-check-page-size", "imp-null-max-insts-to-consider",
      "import-all-index", "incremental-linker-compatible",
      "instcombine-code-sinking", "instcombine-guard-widening-window",
      "instcombine-max-iterations", "instcombine-max-num-phis",
      "instcombine-max-sink-users",
      "instcombine-maxarray-size", "instcombine-negator-enabled",
      "instcombine-negator-max-depth", "instcombine-unsafe-select-transform",
      "instrprof-atomic-counter-update-all", "internalize-public-api-file",
      "internalize-public-api-list", "iterative-counter-promotion",
      "join-liveintervals", "jump-table-type", "limit-float-precision",
      "lower-global-dtors-via-cxa-atexit",
      "lto-embed-bitcode", "matrix-default-layout",
      "matrix-print-after-transpose-opt", "matrix-propagate-shape",
      "max-counter-promotions", "max-counter-promotions-per-loop",
      "mc-relax-all", "mc-x86-disable-arith-relaxation", "mcabac", "meabi",
      "memop-size-large", "memop-size-range", "merror-missing-parenthesis",
      "merror-noncontigious-register", "mfuture-regs", "mhvx",
      "mips-compact-branches", "mips16-constant-islands", "mips16-hard-float",
      "mir-strip-debugify-only", "misexpect-tolerance", "mlsm", "mno-compound",
      "mno-fixup", "mno-ldc1-sdc1", "mno-pairing", "mwarn-missing-parenthesis",
      "mwarn-noncontigious-register", "mwarn-sign-mismatch",
      "no-discriminators", "no-type-check", "no-xray-index",
      "nozero-initialized-in-bss", "nvptx-sched4reg",
      "objc-arc-annotation-target-identifier", "opaque-pointers",
      "pie-copy-relocations", "poison-checking-function-local",
      "polly-dump-after", "polly-dump-after-file", "polly-dump-before",
      "polly-dump-before-file", "pre-RA-sched", "print-after-all",
      "print-before-all", "print-machineinstrs", "print-module-scope",
      "print-pipeline-passes",
      "profile-estimator-loop-weight", "profile-estimator-loop-weight",
      "profile-file", "profile-info-file", "profile-verifier-noassert",
      "pseudo-probe-for-profiling",
      "r600-ir-structurize", "rdf-dump", "rdf-limit", "recip", "regalloc",
      "relax-elf-relocations", "remarks-section", "rewrite-map-file", "rng-seed",
      "runtime-counter-relocation", "safepoint-ir-verifier-print-only",
      "sample-profile-check-record-coverage",
      "sample-profile-check-sample-coverage",
      "sample-profile-inline-hot-threshold",
      "sample-profile-max-propagate-iterations", "shrink-wrap", "simplify-mir",
      "skip-ret-exit-block",
      "speculative-counter-promotion-max-exiting",
      "speculative-counter-promotion-to-loop", "spiller", "spirv-debug",
      "spirv-erase-cl-md", "spirv-lower-const-expr", "spirv-mem2reg",
      "spirv-no-deref-attr", "spirv-text", "spirv-verify-regularize-passes",
      "split-machine-functions", "spv-dump-deps",
      "spv-lower-saddwithoverflow-validate", "spvbool-validate",
      "spvmemmove-validate", "stack-alignment", "stack-protector-guard",
      "stack-protector-guard-offset", "stack-protector-guard-reg",
      "stack-size-section", "stack-symbol-ordering",
      "stackmap-version", "static-func-full-module-prefix",
      "static-func-strip-dirname-prefix", "stats", "stats-json", "strict-dwarf",
      "strip-debug", "struct-path-tbaa", "summary-file", "sve-tail-folding",
      "swift-async-fp",
      "tail-predication", "tailcallopt", "thinlto-assume-merged",
      "thread-model", "time-passes", "time-trace-granularity", "tls-size",
      "type-based-intrinsic-cost", "unfold-element-atomic-memcpy-max-elements",
      "unique-basic-block-section-names", "unique-bb-section-names",
      "unique-section-names", "unit-at-a-time", "use-ctors",
      "vec-extabi", "verify-debug-info", "verify-dom-info",
      "verify-legalizer-debug-locs", "verify-loop-info",
      "verify-loop-lcssa", "verify-machine-dom-info", "verify-regalloc",
      "verify-region-info", "verify-scev", "verify-scev-maps",
      "vp-counters-per-site", "vp-static-alloc",
      "wasm-enable-eh", "wasm-enable-sjlj",
      "x86-align-branch", "x86-align-branch-boundary",
      "x86-branches-within-32B-boundaries", "x86-early-ifcvt",
      "x86-pad-max-prefix-size",
      "x86-recip-refinement-steps", "x86-use-vzeroupper",
      "xcoff-traceback-table",

      // We enable -fdata-sections/-ffunction-sections by default where it makes
      // sense for reducing code size, so hide them to avoid confusion.
      //
      // We need our own switch as these two are defined by LLVM and linked to
      // static TargetMachine members, but the default we want to use depends
      // on the target triple (and thus we do not know it until after the
      // command line has been parsed).
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
