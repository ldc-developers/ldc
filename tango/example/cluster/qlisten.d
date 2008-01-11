private import  tango.core.Thread;

private import  tango.util.log.Configurator;

private import  tango.net.cluster.NetworkQueue;

private import  tango.net.cluster.tina.Cluster;

/*******************************************************************************

        Illustrates how to setup and use a Queue in asynchronous mode

*******************************************************************************/

void main ()
{
        void listen (IEvent event)
        {
                while (event.get)
                       event.log.info ("received asynch msg on channel " ~ event.channel.name);
        }
                

        // join the cluster 
        auto cluster = (new Cluster).join;

        // access a queue of the specified name
        auto queue = new NetworkQueue (cluster, "my.queue.channel");

        // listen for messages placed in my queue, via a delegate
        queue.createConsumer (&listen);

        // stuff something into the queue
        queue.log.info ("sending three messages to the queue");
        queue.put (queue.EmptyMessage);
        queue.put (queue.EmptyMessage);
        queue.put (queue.EmptyMessage);

        // wait for asynchronous msgs to arrive ...
        Thread.sleep (1);
}
