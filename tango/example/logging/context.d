/*******************************************************************************

        copyright:      Copyright (c) 2007 Stonecobra. All rights reserved

        license:        BSD style: $(LICENSE)
      
        version:        Initial release: December 2007
        
        author:         stonecobra

*******************************************************************************/

module context;

import tango.core.Thread,
       tango.util.log.Log,
       tango.util.log.Event,
       tango.util.log.EventLayout,
       tango.util.log.ConsoleAppender;

import tango.util.log.model.IHierarchy;

/*******************************************************************************

        Allows the dynamic setting of log levels on a per-thread basis.  
        Imagine that a user request comes into your production threaded 
        server.  You can't afford to turn logging up to trace for the sake 
        of debugging this one users problem, but you also can't afford to 
        find the problem and fix it.  So now you just set the override log 
        level to TRACE for the thread the user is on, and you get full trace 
        output for only that user.

*******************************************************************************/

class ThreadLocalDiagnosticContext : IHierarchy.Context
{
        private ThreadLocal!(DCData) dcData;
        private char[128] tmp;
		
    	/***********************************************************************
    
    	***********************************************************************/

        public this() 
        {
                dcData = new ThreadLocal!(DCData);
        }
		
    	/***********************************************************************
    
    	        set the 'diagnostic' Level for logging.  This overrides 
                the Level in the current Logger.  The default level starts 
                at NONE, so as not to modify the behavior of existing clients 
                of tango.util.log

    	***********************************************************************/

        void setLevel (ILevel.Level level) 
        {
                auto data = dcData.val;
                data.level = level;
                dcData.val = data;
        }
		
        /***********************************************************************
                
                All log appends will be checked against this to see if a 
                log level needs to be temporarily adjusted.

        ***********************************************************************/

        bool isEnabled (ILevel.Level setting, ILevel.Level level = ILevel.Level.Trace) 
        {
        	return level >= setting || level >= dcData.val.level;
        }

        /***********************************************************************
        
                Return the label to use for the current log message.  Usually 
                called by the Layout. This implementation returns "{}".

        ***********************************************************************/

        char[] label () 
        {
        	return dcData.val.getLabel;
        }
        
        /***********************************************************************
        
                Push another string into the 'stack'.  This strings will be 
                appened together when getLabel is called.

        ***********************************************************************/

        void push (char[] label) 
        {
                auto data = dcData.val;
                data.push(label);
        	dcData.val = data;
        }
        
        /***********************************************************************
        
                pop the current label off the stack.

        ***********************************************************************/

        void pop ()
        {
                auto data = dcData.val;
                data.pop;
        	dcData.val = data;
        }
        
        /***********************************************************************
        
                Clear the label stack.

        ***********************************************************************/

        void clear()
        {
                auto data = dcData.val;
                data.clear;
        	dcData.val = data;
        }
}


/*******************************************************************************
        
        The thread locally stored struct to hold the logging level and 
        the label stack.

*******************************************************************************/

private struct DCData {
	
        ILevel.Level    level = ILevel.Level.None;
        char[][8] 	stack;
        bool 		shouldUpdate = true;
        int         	stackIndex = 0;
        uint   	        labelLength;
        char[256]       labelContent;
	
	
	char[] getLabel() {
		if (shouldUpdate) {
			labelLength = 0;
			append(" {");
			for (int i = 0; i < stackIndex; i++) {
				append(stack[i]);
				if (i < stackIndex - 1) {
					append(" ");
				}
			}
			append("}");
			shouldUpdate = false;
		}
		return labelContent[0..labelLength];
	}
	
	void append(char[] x) {
        uint addition = x.length;
        uint newLength = labelLength + x.length;

        if (newLength < labelContent.length)
           {
           labelContent [labelLength..newLength] = x[0..addition];
           labelLength = newLength;
           }
	}
	
	void push(char[] label) {
		shouldUpdate = true;
		stack[stackIndex] = label.dup;
		stackIndex++;
	}
	
	void pop() {
		shouldUpdate = true;
		if (stackIndex > 0) {
			stack[stackIndex] = null;
			stackIndex--;
		}
	}
		
	void clear() {
		shouldUpdate = true;
		for (int i = 0; i < stack.length; i++) {
			stack[i] = null;
		}
	}
}


/*******************************************************************************

        Simple console appender that counts the number of log lines it 
        has written.

*******************************************************************************/

class TestingConsoleAppender : ConsoleAppender {

    int events = 0;
	
    this (EventLayout layout = null)
    {
    	super(layout);
    }

    override void append (Event event)
    {
    	events++;
    	super.append(event);
    }
}


/*******************************************************************************

        Testing harness for the DiagnosticContext functionality.

*******************************************************************************/

void main(char[][] args) 
{	
    //set up our appender that counts the log output.  This is the configuration 
    //equivalent of importing tango.util.log.Configurator.
    auto appender = new TestingConsoleAppender(new SimpleTimerLayout);
    Log.getRootLogger.addAppender(appender);

    char[128] tmp = 0;
    auto log = Log.getLogger("context");    
    log.setLevel(log.Level.Info);
    
    //first test, use all defaults, validating it is working.  None of the trace()
    //calls should count in the test.
    for (int i=0;i < 10; i++) {
    	log.info(log.format(tmp, "test1 {}", i));
    	log.trace(log.format(tmp, "test1 {}", i));
    }
    if (appender.events !is 10) {
    	log.error(log.format(tmp, "events:{}", appender.events));
    	throw new Exception("Incorrect Number of events in normal mode");	
    }
    
    appender.events = 0;
    
    //test the thread local implementation without any threads, as a baseline.
    //should be same result as test1
    auto context = new ThreadLocalDiagnosticContext;
    Log.getHierarchy.context(context);
    for (int i=0;i < 10; i++) {
    	log.info(log.format(tmp, "test2 {}", i));
    	log.trace(log.format(tmp, "test2 {}", i));
    }
    if (appender.events !is 10) {
    	log.error(log.format(tmp, "events:{}", appender.events));
    	throw new Exception("Incorrect Number of events in TLS single thread mode");	
    }
    
    appender.events = 0;
    
    //test the thread local implementation without any threads, as a baseline.
    //This should count all logging requests, because the DiagnosticContext has
    //'overridden' the logging level on ALL loggers up to TRACE.
    context.setLevel(log.Level.Trace);
    for (int i=0;i < 10; i++) {
    	log.info(log.format(tmp, "test3 {}", i));
    	log.trace(log.format(tmp, "test3 {}", i));
    }
    if (appender.events !is 20) {
    	log.error(log.format(tmp, "events:{}", appender.events));
    	throw new Exception("Incorrect Number of events in TLS single thread mode with level set");	
    }
    
    appender.events = 0;
    
    //test the thread local implementation without any threads, as a baseline.
    context.setLevel(log.Level.None);
    for (int i=0;i < 10; i++) {
    	log.info(log.format(tmp, "test4 {}", i));
    	log.trace(log.format(tmp, "test4 {}", i));
    }
    if (appender.events !is 10) {
    	log.error(log.format(tmp, "events:{}", appender.events));
    	throw new Exception("Incorrect Number of events in TLS single thread mode after level reset");	
    }
    
    //Now test threading.  set up a trace context in one thread, with a label, while
    //keeping the second thread at the normal configuration.
    appender.events = 0;
    ThreadGroup tg = new ThreadGroup();
    tg.create({
        char[128] tmp = 0;
        context.setLevel(log.Level.Trace);        
        context.push("specialthread");
        context.push("2ndlevel");
        for (int i=0;i < 10; i++) {
             log.info(log.format(tmp, "test5 {}", i));
             log.trace(log.format(tmp, "test5 {}", i));
        }
    });
    tg.create({
        char[128] tmp = 0;      
        context.setLevel(log.Level.None);
        for (int i=0;i < 10; i++) {
             log.info(log.format(tmp, "test6 {}", i));
             log.trace(log.format(tmp, "test6 {}", i));
        }
    });
    tg.joinAll();
    
    if (appender.events !is 30) {
    	log.error(log.format(tmp, "events:{}", appender.events));
    	throw new Exception("Incorrect Number of events in TLS multi thread mode");	
    }   
}

