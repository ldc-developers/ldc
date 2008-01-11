/*******************************************************************************

*******************************************************************************/

import tango.io.Console;

import tango.net.InternetAddress;

import tango.net.cluster.tina.CmdParser,
       tango.net.cluster.tina.CacheServer;

/*******************************************************************************

*******************************************************************************/

void main (char[][] args)
{
        auto arg = new CmdParser ("cache.server");

        // default number of cache entries
        arg.size = 8192;

        if (args.length > 1)
            arg.parse (args[1..$]);
                        
        if (arg.help)
            Cout ("usage: cacheserver -port=number -size=cachesize -log[=trace, info, warn, error, fatal, none]").newline;
        else
           (new CacheServer(new InternetAddress(arg.port), arg.log, arg.size)).start;
}
