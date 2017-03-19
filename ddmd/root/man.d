/**
 * Compiler implementation of the D programming language
 * http://dlang.org
 *
 * Copyright: Copyright (c) 1999-2016 by Digital Mars, All Rights Reserved
 * Authors:   Walter Bright, http://www.digitalmars.com
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:    $(DMDSRC root/_man.d)
 */

module ddmd.root.man;

import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.string;
import core.sys.posix.unistd;
import core.sys.windows.windows;

version (Windows)
{
    extern (C++) void browse(const(char)* url)
    in
    {
        assert(strncmp(url, "http://", 7) == 0 || strncmp(url, "https://", 8) == 0);
    }
    body
    {
        ShellExecuteA(null, "open", url, null, null, SW_SHOWNORMAL);
    }
}
else version (OSX)
{
    extern (C++) void browse(const(char)* url)
    in
    {
        assert(strncmp(url, "http://", 7) == 0 || strncmp(url, "https://", 8) == 0);
    }
    body
    {
        pid_t childpid;
        const(char)*[5] args;
        char* browser = getenv("BROWSER");
        if (browser)
        {
            browser = strdup(browser);
            args[0] = browser;
            args[1] = url;
            args[2] = null;
        }
        else
        {
            args[0] = "open";
            args[1] = url;
            args[2] = null;
        }
        childpid = fork();
        if (childpid == 0)
        {
            execvp(args[0], cast(char**)args);
            perror(args[0]); // failed to execute
            return;
        }
    }
}
else version (Posix)
{
    extern (C++) void browse(const(char)* url)
    in
    {
        assert(strncmp(url, "http://", 7) == 0 || strncmp(url, "https://", 8) == 0);
    }
    body
    {
        pid_t childpid;
        const(char)*[3] args;
        const(char)* browser = getenv("BROWSER");
        if (browser)
            browser = strdup(browser);
        else
            browser = "x-www-browser";
        args[0] = browser;
        args[1] = url;
        args[2] = null;
        childpid = fork();
        if (childpid == 0)
        {
            execvp(args[0], cast(char**)args);
            perror(args[0]); // failed to execute
            return;
        }
    }
}
