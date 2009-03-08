#include "gen/cl_helpers.h"

#include "root.h"
#include "mem.h"

#include <cctype>       // isupper, tolower
#include <algorithm>
#include <utility>
#include <stdarg.h>

namespace opts {

// Helper function
static char toLower(char c) {
    if (isupper(c))
        return tolower(c);
    return c;
}

bool FlagParser::parse(cl::Option &O, const char *ArgName, const std::string &Arg, bool &Val) {
    // Make a std::string out of it to make comparisons easier
    // (and avoid repeated conversion)
    std::string argname = ArgName;
    
    typedef std::vector<std::pair<std::string, bool> >::iterator It;
    for (It I = switches.begin(), E = switches.end(); I != E; ++I) {
        std::string name = I->first;
        if (name == argname
                || (name.length() < argname.length()
                    && argname.substr(0, name.length()) == name
                    && argname[name.length()] == '=')) {
            
            if (!cl::parser<bool>::parse(O, ArgName, Arg, Val)) {
                Val = (Val == I->second);
                return false;
            }
            // Invalid option value
            break;
        }
    }
    return true;
}

void FlagParser::getExtraOptionNames(std::vector<const char*> &Names) {
    typedef std::vector<std::pair<std::string, bool> >::iterator It;
    for (It I = switches.begin() + 1, E = switches.end(); I != E; ++I) {
        Names.push_back(I->first.c_str());
    }
}


MultiSetter::MultiSetter(bool invert, bool* p, ...) {
    this->invert = invert;
    if (p) {
        locations.push_back(p);
        va_list va;
        va_start(va, p);
        while (p = va_arg(va, bool*)) {
            locations.push_back(p);
        }
    }
}
        
void MultiSetter::operator=(bool val) {
    typedef std::vector<bool*>::iterator It;
    for (It I = locations.begin(), E = locations.end(); I != E; ++I) {
        **I = (val != invert);
    }
}


void ArrayAdapter::push_back(const char* cstr) {
    if (!cstr || !*cstr)
        error("Expected argument to '-%s'", name);
    
    if (!*arrp)
        *arrp = new Array;
    (*arrp)->push(mem.strdup(cstr));
}

} // namespace opts
