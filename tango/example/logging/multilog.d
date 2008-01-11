import tango.util.log.Log;
import tango.util.log.Log4Layout;
import tango.util.log.FileAppender;
import tango.util.log.ConsoleAppender;
import tango.util.log.RollingFileAppender;

/*******************************************************************************

        Shows how to setup multiple appenders on logging tree

*******************************************************************************/

void main ()
{
        // set default logging level at the root
        auto log = Log.getRootLogger;
        log.setLevel (log.Level.Trace);

        // 10 logs, all with 10 mbs each
        log.addAppender (new RollingFileAppender("rolling.log", 9, 1024*1024*10));

        // a single file appender, with an XML layout
        log.addAppender (new FileAppender ("single.log", new Log4Layout));

        // console appender
        log.addAppender (new ConsoleAppender);

        // log to all
        log.trace ("three-way logging");
}

