#ifndef LDC_CL_HELPERS_H
#define LDC_CL_HELPERS_H

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"

struct Array;

namespace opts {
    namespace cl = llvm::cl;
    
    /// Helper class for fancier options
    class FlagParser : public cl::parser<bool> {
        std::vector<std::pair<std::string, bool> > switches;
    public:
        template <class Opt>
        void initialize(Opt &O) {
            assert(!(O.getMiscFlags() & cl::AllowInverse)
                && "FlagParser doesn't support redundant AllowInverse flag");
            
            std::string Name = O.ArgStr;
            switches.push_back(make_pair("enable-" + Name, true));
            switches.push_back(make_pair("disable-" + Name, false));
            // Replace <foo> with -enable-<foo>
            O.ArgStr = switches[0].first.c_str();
        }
        
        bool parse(cl::Option &O, const char *ArgName, const std::string &ArgValue, bool &Val);
        
        void getExtraOptionNames(std::vector<const char*> &Names);
    };
    
    /// Helper class for options that set multiple flags
    class MultiSetter {
        std::vector<bool*> locations;
        bool invert;
        MultiSetter(bool); //not implemented, disable auto-conversion
    public:
        MultiSetter(bool invert, bool* p, ...) END_WITH_NULL;
        
        void operator=(bool val);
    };
    
    /// Helper class to fill Array with char* when given strings
    /// (Errors on empty strings)
    class ArrayAdapter {
        const char* name;
        Array** arrp;
    public:
        ArrayAdapter(const char* name_, Array*& arr) {
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
