import  tango.core.Thread;

import  tango.util.log.Log,
        tango.util.log.Log4Layout,
        tango.util.log.SocketAppender;

import  tango.net.InternetAddress;


/*******************************************************************************

        Hooks up to Chainsaw for remote log capture. Chainsaw should be 
        configured to listen with an XMLSocketReciever

*******************************************************************************/

void main()
{
        // get a logger to represent this module
        auto logger = Log.getLogger ("example.chainsaw");

        // hook up an appender for XML output
        logger.addAppender (new SocketAppender (new InternetAddress("127.0.0.1", 4448), new Log4Layout));

        while (true)
              {
              logger.info ("Hello Chainsaw!");      
              Thread.sleep (1.0);
              }
}
