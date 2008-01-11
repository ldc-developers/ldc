/*******************************************************************************


*******************************************************************************/

import tango.io.Stdout;

import tango.time.StopWatch;

import tango.util.log.Configurator;

import tango.net.cluster.NetworkQueue;

import tango.net.cluster.tina.Cluster;

/*******************************************************************************


*******************************************************************************/

void main (char[][] args)
{
        StopWatch w;

        auto cluster = (new Cluster).join;
        auto queue   = new NetworkQueue (cluster, "my.queue.channel");

        while (true)
              {
              w.start;
              for (int i=10000; i--;)
                   queue.put (queue.EmptyMessage);

              Stdout.formatln ("{} put/s", 10000/w.stop);

              uint count;
              w.start;
              while (queue.get !is null)
                     ++count;
        
              Stdout.formatln ("{} get/s", count/w.stop);
              }
}

