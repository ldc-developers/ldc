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

#if LDC_LLVM_VER < 306
#define LLVM_END_WITH_NULL END_WITH_NULL
#endif

template <typename TYPE> struct Array;
typedef Array<const char *> Strings;

namespace opts {
    namespace cl = llvm::cl;

    /// Helper class for fancier options
    class FlagParser : public cl::parser<bool> {
        std::vector<std::pair<std::string, bool> > switches;
    public:
        template <class Opt>
        void initialize(Opt &O) {
            std::string Name = O.ArgStr;
            switches.push_back(make_pair("enable-" + Name, true));
            switches.push_back(make_pair("disable-" + Name, false));
            // Replace <foo> with -enable-<foo>
            O.ArgStr = switches[0].first.data();
        }

        bool parse(cl::Option &O, llvm::StringRef ArgName, llvm::StringRef ArgValue, bool &Val);

        void getExtraOptionNames(llvm::SmallVectorImpl<const char*> &Names);
    };

    /// Helper class for options that set multiple flags
    class MultiSetter {
        std::vector<bool*> locations;
        bool invert;
        MultiSetter(bool); //not implemented, disable auto-conversion
    public:
        MultiSetter(bool invert, bool* p, ...) LLVM_END_WITH_NULL;

        void operator=(bool val);
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

    /// Helper class to allow use of a parser<bool> with BoolOrDefault
    class BoolOrDefaultAdapter {
        cl::boolOrDefault value;
    public:
        operator cl::boolOrDefault() {
            return value;
        }

        void operator=(cl::boolOrDefault val) {
            value = val;
        }

        void operator=(bool val) {
            *this = (val ? cl::BOU_TRUE : cl::BOU_FALSE);
        }
    };
}

#endif
