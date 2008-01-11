private import  tango.core.Thread;

private import  tango.util.log.Configurator;

private import  tango.net.cluster.tina.Cluster;

private import  tango.net.cluster.NetworkQueue;

/*******************************************************************************

*******************************************************************************/

void main()
{
        // open the cluster and a queue channel. Note that the queue has
        // been configured with a reply listener ...
        auto cluster = (new Cluster).join;
        auto queue = new NetworkQueue (cluster, "message.channel", 
                                      (IEvent event){event.log.info ("Received reply");}
                                      );

        void recipient (IEvent event)
        {
                auto msg = event.get;
        
                event.log.info ("Replying to message on channel "~msg.reply);
                event.reply (event.replyChannel(msg), queue.EmptyMessage);
        }

        // setup a listener to recieve and reply
        queue.createConsumer (&recipient);

        // toss a message out to the cluster
        queue.put (queue.EmptyMessage);

        // wait for completion ...
        Thread.sleep (1);
}
