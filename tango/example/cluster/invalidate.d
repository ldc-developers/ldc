private import  tango.core.Thread;

private import  tango.util.log.Configurator;

private import  tango.net.cluster.tina.Cluster;

private import  tango.net.cluster.QueuedCache,
                tango.net.cluster.CacheInvalidatee,
                tango.net.cluster.CacheInvalidator;

/*******************************************************************************

        Demonstrates how to invalidate cache entries across a cluster
        via a channel

*******************************************************************************/

void main()
{
        // access the cluster
        auto cluster = (new Cluster).join;

        // wrap a cache instance with a network listener
        auto dst = new CacheInvalidatee (cluster, "my.cache.channel", new QueuedCache!(char[], IMessage)(101));

        // connect an invalidator to that cache channel
        auto src = new CacheInvalidator (cluster, "my.cache.channel");

        // stuff something in the local cache
        dst.cache.put ("key", dst.EmptyMessage);

        // get it removed via a network broadcast
        src.log.info ("invalidating 'key' across the cluster");
        src.invalidate ("key");

        // wait for it to arrive ...
        Thread.sleep (1);
}
