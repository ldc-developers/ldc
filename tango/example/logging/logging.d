/*******************************************************************************

        Shows how the basic functionality of Logger operates.

*******************************************************************************/

private import tango.util.log.Log,
               tango.util.log.Configurator;

/*******************************************************************************

        Search for a set of prime numbers

*******************************************************************************/

void compute (Logger log, uint max)
        {
                byte*   feld;
                int     teste=1,
                        mom,
                        hits=0,
                        s=0,
                        e = 1;
                int     count;
                char    tmp[128] = void;

                void set (byte* f, uint x)
                {
                        *(f+(x)/16) |= 1 << (((x)%16)/2);
                }

                byte test (byte* f, uint x)
                {
                        return cast(byte) (*(f+(x)/16) & (1 << (((x)%16)/2)));
                }

                // information level
                log.info (log.format (tmp, "Searching prime numbers up to {}", max));

                feld = (new byte[max / 16 + 1]).ptr;

                // get milliseconds since application began
                auto begin = log.runtime;

                while ((teste += 2) < max)
                        if (! test (feld, teste)) 
                           {
                           if  ((++hits & 0x0f) == 0) 
                                // more information level
                                log.info (log.format (tmp, "found {}", hits)); 

                           for (mom=3*teste; mom < max; mom += teste<<1) 
                                set (feld, mom);
                           }

                // get number of milliseconds we took to compute
                auto period = log.runtime - begin;

                if (hits)
                    // more information
                    log.info (log.format (tmp, "{} prime numbers found in {} millsecs", hits, period));
                else
                   // a warning level
                   log.warn ("no prime numbers found");
        
                // check to see if we're enabled for 
                // tracing before we expend a lot of effort
                if (log.isEnabled (log.Level.Trace))
                   {        
                   e = max;
                   count = 0 - 2; 
                   if (s % 2 is 0) 
                       count++;
           
                   while ((count+=2) < e) 
                           // log trace information
                           if (! test (feld, count)) 
                                 log.trace (log.format (tmp, "prime found: {}", count));
                   }
}


/*******************************************************************************

        Compute a bunch of prime numbers

*******************************************************************************/

void main()
{
        // get a logger to represent this module. We could just as
        // easily share a name with some other module(s)
        auto log = Log.getLogger ("example.logging");
        try {
            compute (log, 1000);

            } catch (Exception x)
                    {
                    // log the exception as an error
                    log.error ("Exception: " ~ x.toString);
                    }
}
