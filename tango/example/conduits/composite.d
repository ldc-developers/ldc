
private import  tango.io.protocol.Reader,
                tango.io.protocol.Writer,
                tango.io.FileConduit;

/*******************************************************************************

        Use cascading reads & writes to handle a composite class. There is
        just one primary call for output, and just one for input, but the
        classes propogate the request as appropriate. 

        Note that the class instances don't know how their content will be
        represented; that is dictated by the caller (via the reader/writer
        implementation).

        Note also that this only serializes the content. To serialize the
        classes too, take a look at the Pickle.d example.

*******************************************************************************/

void main()
{
        // define a serializable class (via interfaces)
        class Wumpus : IReadable, IWritable
        {
                private int     a = 11,
                                b = 112,
                                c = 1024;

                void read (IReader input)
                {
                        input (a) (b) (c);
                }

                void write (IWriter output)
                {
                        output (a) (b) (c);
                }
        }


        // define a serializable class (via interfaces)
        class Wombat : IReadable, IWritable
        {
                private Wumpus  wumpus;
                private char[]  x = "xyz";
                private bool    y = true;
                private float   z = 3.14159;

                this (Wumpus wumpus)
                {
                        this.wumpus = wumpus;
                }

                void read (IReader input)
                {
                        input (x) (y) (z) (wumpus);
                }

                void write (IWriter output)
                {
                        output (x) (y) (z) (wumpus);
                }
        }

        // construct a Wombat
        auto wombat = new Wombat (new Wumpus);

        // open a file for IO
        auto file = new FileConduit ("random.bin", FileConduit.ReadWriteCreate);

        // construct reader & writer upon the file, with binary IO
        auto output = new Writer (file);
        auto input = new Reader (file);

        // write both Wombat & Wumpus (and flush them)
        output (wombat) ();

        // rewind to file start
        file.seek (0);

        // read both back again
        input (wombat);
}

