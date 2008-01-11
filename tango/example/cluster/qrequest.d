private import  tango.util.log.Configurator;

private import  tango.net.cluster.NetworkQueue;

private import  tango.net.cluster.tina.Cluster;

/*******************************************************************************

        Illustrates how to setup and use a Queue in synchronous mode

*******************************************************************************/

void main ()
{
        // join the cluster 
        auto cluster = (new Cluster).join;

        // access a queue of the specified name
        auto queue = new NetworkQueue (cluster, "my.queue.channel");

        // stuff something into the queue
        queue.log.info ("sending three messages to the queue");
        queue.put (queue.EmptyMessage);
        queue.put (queue.EmptyMessage);
        queue.put (queue.EmptyMessage);

        // retreive synchronously
        while (queue.get)
               queue.log.info ("retrieved msg");
}
