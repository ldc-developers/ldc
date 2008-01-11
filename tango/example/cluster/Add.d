/*******************************************************************************

*******************************************************************************/

public import tango.net.cluster.NetworkCall;


/*******************************************************************************
        
        a Task function

*******************************************************************************/

real add (real x, real y)
{
        return x + y;
}


/*******************************************************************************

        a Task function

*******************************************************************************/

int divide (int x, int y)
{
        return x / y;
}


/*******************************************************************************

        a verbose Task message

*******************************************************************************/

class Subtract : NetworkCall
{
        double  a,
                b,
                result;

        double opCall (double a, double b, IChannel channel = null)
        {
                this.a = a;
                this.b = b;
                send (channel);
                return result;
        }

        override void execute ()
        {
                result = a - b;
        }

        override void read  (IReader input)  {input (a)(b)(result);}

        override void write (IWriter output) {output (a)(b)(result);}
}
