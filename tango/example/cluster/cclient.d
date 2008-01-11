/*******************************************************************************


*******************************************************************************/

import tango.io.Stdout;

import tango.time.StopWatch;

import tango.util.log.Configurator;

import tango.net.cluster.NetworkCache;

import tango.net.cluster.tina.Cluster;

/*******************************************************************************


*******************************************************************************/

void main (char[][] args)
{
        StopWatch w;

        if (args.length > 1)
           {
           auto cluster = (new Cluster).join (args[1..$]);
           auto cache   = new NetworkCache (cluster, "my.cache.channel");

           while (true)
                 {
                 w.start;
                 for (int i=10000; i--;)
                      cache.put ("key", cache.EmptyMessage);

                 Stdout.formatln ("{} put/s", 10000/w.stop);

                 w.start;
                 for (int i=10000; i--;)
                      cache.get ("key");
        
                 Stdout.formatln ("{} get/s", 10000/w.stop);
                 }
           }
        else
           Stdout.formatln ("usage: cache cachehost:port ...");
}

