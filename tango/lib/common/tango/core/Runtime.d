/**
 * The runtime module exposes information specific to the D runtime code.
 *
 * Copyright: Copyright (C) 2005-2006 Sean Kelly.  All rights reserved.
 * License:   BSD style: $(LICENSE)
 * Authors:   Sean Kelly
 */
module tango.core.Runtime;


private
{
    extern (C) bool rt_isHalting();

    alias bool function() moduleUnitTesterType;
}


////////////////////////////////////////////////////////////////////////////////
// Runtime
////////////////////////////////////////////////////////////////////////////////


/**
 * This struct encapsulates all functionality related to the underlying runtime
 * module for the calling context.
 */
struct Runtime
{
    /**
     * Returns true if the runtime is halting.  Under normal circumstances,
     * this will be set between the time that normal application code has
     * exited and before module dtors are called.
     *
     * Returns:
     *  true if the runtime is halting.
     */
    static bool isHalting()
    {
        return rt_isHalting();
    }


    /**
     * Overrides the default module unit tester with a user-supplied version.
     *
     * Params:
     *  h = The new unit tester.  Set to null to use the default unit tester.
     */
    static void moduleUnitTester( moduleUnitTesterType h )
    {
        sm_moduleUnitTester = h;
    }


private:
    static moduleUnitTesterType sm_moduleUnitTester = null;
}


////////////////////////////////////////////////////////////////////////////////
// Overridable Callbacks
////////////////////////////////////////////////////////////////////////////////


/**
 * This routine is called by the runtime to run module unit tests on startup.
 * The user-supplied unit tester will be called if one has been supplied,
 * otherwise all unit tests will be run in sequence.
 *
 * Returns:
 *  true if execution should continue after testing is complete and false if
 *  not.  Default behavior is to return true.
 */
extern (C) bool runModuleUnitTests()
{
    if( Runtime.sm_moduleUnitTester is null )
    {
        foreach( m; ModuleInfo )
        {
            if( m.unitTest )
                m.unitTest();
        }
        return true;
    }
    return Runtime.sm_moduleUnitTester();
}
