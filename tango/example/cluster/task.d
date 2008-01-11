/*******************************************************************************

        Illustrates usage of cluster tasks

*******************************************************************************/

import Add, tango.io.Stdout, tango.net.cluster.tina.ClusterTask;

void main (char[][] args)
{
        scope add = new NetCall!(add);

        Stdout.formatln ("cluster expression of 3.0 + 4.0 = {}", add(3, 4));
}

