//===-- gen/cl_helpers.h - Command line processing helpers ------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Helpers to augment the LLVM command line parsing library with some extra
// functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_CL_HELPERS_H
#define LDC_GEN_CL_HELPERS_H

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

#if LDC_LLVM_VER < 306
#define LLVM_END_WITH_NULL END_WITH_NULL
#endif

template <typename TYPE> struct Array;
typedef Array<const char *> Strings;

namespace opts {
    namespace cl = llvm::cl;

    /// Helper class for options that set multiple flags
    class MultiSetter {
        std::vector<bool*> locations;
        bool invert;
        MultiSetter(bool); //not implemented, disable auto-conversion
    public:
        MultiSetter(bool invert, bool* p, ...) LLVM_END_WITH_NULL;

        void operator=(bool val);
        operator bool() const { return *locations[0]; }
    };

    /// Helper class to fill Strings with char* when given strings
    /// (Errors on empty strings)
    class StringsAdapter {
        const char* name;
        Strings** arrp;
    public:
        StringsAdapter(const char* name_, Strings*& arr) {
            name = name_;
            arrp = &arr;
            assert(name);
            assert(arrp);
        }

        void push_back(const char* cstr);

        void push_back(const std::string& str) {
            push_back(str.c_str());
        }
    };

    /// Helper class to determine values
    template<class DT>
    struct FlagParserDataType {};

    template<>
    struct FlagParserDataType<bool>{
        static const bool true_val = true;
        static const bool false_val = false;
    };

    template<>
    struct FlagParserDataType<cl::boolOrDefault> {
        static const cl::boolOrDefault true_val = cl::BOU_TRUE;
        static const cl::boolOrDefault false_val = cl::BOU_FALSE;
    };

    /// Helper class for fancier options
    template<class DT>
    class FlagParser : public cl::basic_parser_impl {
#if LDC_LLVM_VER >= 307
        cl::Option &Opt;
#endif
        llvm::SmallVector<std::pair<std::string, DT>, 2> switches;
    public:
        typedef DT parser_data_type;
        typedef cl::OptionValue<DT> OptVal;

#if LDC_LLVM_VER >= 307
        FlagParser(cl::Option &O) : basic_parser_impl(O), Opt(O) { }

        void initialize() {
            std::string Name(Opt.ArgStr);
            switches.push_back(make_pair("enable-" + Name, FlagParserDataType<DT>::true_val));
            switches.push_back(make_pair("disable-" + Name, FlagParserDataType<DT>::false_val));
            // Replace <foo> with -enable-<foo> and register -disable-<foo>
            // A literal option can only registered if the argstr is empty -
            // just do this first.
            Opt.setArgStr("");
            AddLiteralOption(Opt, strdup(switches[1].first.data()));
            Opt.setArgStr(switches[0].first.data());
        }
#else
        template <class Opt>
        void initialize(Opt &O) {
            std::string Name = O.ArgStr;
            switches.push_back(make_pair("enable-" + Name, FlagParserDataType<DT>::true_val));
            switches.push_back(make_pair("disable-" + Name, FlagParserDataType<DT>::false_val));
            // Replace <foo> with -enable-<foo>
            O.ArgStr = switches[0].first.data();
        }
#endif

        enum cl::ValueExpected getValueExpectedFlagDefault() const {
            return cl::ValueOptional;
        }

        bool parse(cl::Option &O, llvm::StringRef ArgName, llvm::StringRef ArgValue, DT &Val) {
            // Make a std::string out of it to make comparisons easier
            // (and avoid repeated conversion)
            llvm::StringRef argname = ArgName;

            typedef typename llvm::SmallVector<std::pair<std::string, DT>, 2>::iterator It;
            for (It I = switches.begin(), E = switches.end(); I != E; ++I) {
                llvm::StringRef name = I->first;
                if (name == argname
                    || (name.size() < argname.size()
                    && argname.substr(0, name.size()) == name
                    && argname[name.size()] == '=')) {
                    if (!parse(O, ArgValue, Val))
                    {
                        Val = (Val == I->second) ? FlagParserDataType<DT>::true_val : FlagParserDataType<DT>::false_val;
                        return false;
                    }
                    // Invalid option value
                    break;
                }
            }
            return true;
        }

        void getExtraOptionNames(llvm::SmallVectorImpl<const char*> &Names) {
            typedef typename llvm::SmallVector<std::pair<std::string, DT>, 2>::iterator It;
            for (It I = switches.begin() + 1, E = switches.end(); I != E; ++I) {
                Names.push_back(I->first.data());
            }
        }

        // getValueName - Do not print =<value> at all.
        const char *getValueName() const override { return nullptr; }

        void printOptionDiff(const cl::Option &O, bool V, OptVal Default, size_t GlobalWidth) const {
            printOptionName(O, GlobalWidth);
            std::string Str;
            {
                llvm::raw_string_ostream SS(Str);
                SS << V;
            }
            llvm::outs() << "= " << Str;
            static const size_t MaxOptWidth = 8; // arbitrary spacing for printOptionDiff
            size_t NumSpaces = \
                MaxOptWidth > Str.size() ? MaxOptWidth - Str.size() : 0;
            llvm::outs().indent(NumSpaces) << " (default: ";
            if (Default.hasValue())
                llvm::outs() << Default.getValue();
            else
                llvm::outs() << "*no default*";
            llvm::outs() << ")\n";
        }

        // An out-of-line virtual method to provide a 'home' for this class.
        void anchor() override;

    private:
        static bool parse(cl::Option &O, llvm::StringRef Arg, DT &Val) {
            if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" ||
                Arg == "1") {
                Val = FlagParserDataType<DT>::true_val;
                return false;
            }

            if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
                Val = FlagParserDataType<DT>::false_val;
                return false;
            }
            return O.error("'" + Arg +
                "' is invalid value for boolean argument! Try 0 or 1");
        }

    };

EXTERN_TEMPLATE_INSTANTIATION(class FlagParser<bool>);
EXTERN_TEMPLATE_INSTANTIATION(class FlagParser<cl::boolOrDefault>);
}

namespace llvm {
    namespace cl {
        // This template is selected in 2 cases:
        // - DT and ParserClass::parser_data_type are equal
        // - DT can be assigned to ParserClass::parser_data_type
        template <class ParserClass, class DT>
        void printOptionDiff(const Option &O, const opts::FlagParser<typename ParserClass::parser_data_type> &P, const DT &V,
            const OptionValue<DT> &Default, size_t GlobalWidth) {
            typename ParserClass::parser_data_type Val = V;
            const OptionValue<typename ParserClass::parser_data_type> DefaultVal = Default.getValue();
            P.printOptionDiff(O, Val, DefaultVal, GlobalWidth);
        }
    }
}

#endif
