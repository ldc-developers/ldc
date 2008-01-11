
private import  tango.io.FileConduit;

private import  tango.io.protocol.Reader,
                tango.io.protocol.Writer;

/*******************************************************************************

       Create a file for random access. Write some stuff to it, rewind to
       file start and read back.

*******************************************************************************/

void main()
{
        // open a file for reading
        auto fc = new FileConduit ("random.bin", FileConduit.ReadWriteCreate);

        // construct (binary) reader & writer upon this conduit
        auto read  = new Reader (fc);
        auto write = new Writer (fc);

        int x=10, y=20;

        // write some data and flush output since IO is buffered
        write (x) (y) ();

        // rewind to file start
        fc.seek (0);

        // read data back again, but swap destinations
        read (y) (x);

        assert (y is 10);
        assert (x is 20);

        fc.close();
}
