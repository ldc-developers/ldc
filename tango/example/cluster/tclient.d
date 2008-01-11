/*******************************************************************************


*******************************************************************************/

import Add;

import tango.io.Stdout;

import tango.time.StopWatch;

import tango.util.log.Configurator;

import tango.net.cluster.tina.ClusterTask;

/*******************************************************************************


*******************************************************************************/

void main (char[][] args)
{
        // an implicit task instance
        auto add = new NetCall!(add);

        // an explicit task instance
        auto sub = new Subtract;

        StopWatch w;
        while (true)
              {
              w.start;
              for (int i=10000; i--;)
                  {
                  // both task types are used in the same manner
                  add (1, 2);
                  sub (3, 4);
                  }
              Stdout.formatln ("{} calls/s", 20000/w.stop);
              }
}

