#ifndef LDC_CONF_CONFIGFILE_H
#define LDC_CONF_CONFIGFILE_H

#include <vector>

namespace libconfig
{
    class Config;
}

class ConfigFile
{
public:
    typedef std::vector<const char*>    s_vector;
    typedef s_vector::iterator          s_iterator;

public:
    ConfigFile();
    ~ConfigFile();

    bool read(const char* argv0, void* mainAddr, const char* filename);

    s_iterator switches_begin()   { return switches.begin(); }
    s_iterator switches_end()     { return switches.end(); }

private:
    libconfig::Config* cfg;

    s_vector switches;
};

#endif // LDC_CONF_CONFIGFILE_H
