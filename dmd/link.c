

// Copyright (c) 1999-2007 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.


#include	<stdio.h>
#include	<ctype.h>
#include	<assert.h>
#include	<stdarg.h>
#include	<string.h>
#include	<stdlib.h>

#if _WIN32
#include	<process.h>
#endif

#if linux
#include	<sys/types.h>
#include	<sys/wait.h>
#include	<unistd.h>
#endif

#include	"root.h"

#include	"mars.h"

#include	"mem.h"

#include "gen/logger.h"

int executecmd(char *cmd, char *args, int useenv);
int executearg0(char *cmd, char *args);

/**********************************
 * Delete generated EXE file.
 */

void deleteExeFile()
{
    if (global.params.exefile)
    {
	//printf("deleteExeFile() %s\n", global.params.exefile);
	remove(global.params.exefile);
    }
}

/******************************
 * Execute a rule.  Return the status.
 *	cmd	program to run
 *	args	arguments to cmd, as a string
 *	useenv	if cmd knows about _CMDLINE environment variable
 */

#if _WIN32
int executecmd(char *cmd, char *args, int useenv)
{
    int status;
    char *buff;
    size_t len;

    if (!global.params.quiet || global.params.verbose)
    {
	printf("%s %s\n", cmd, args);
	fflush(stdout);
    }

    if ((len = strlen(args)) > 255)
    {   char *q;
	static char envname[] = "@_CMDLINE";

	envname[0] = '@';
	switch (useenv)
	{   case 0:	goto L1;
	    case 2: envname[0] = '%';	break;
	}
	q = (char *) alloca(sizeof(envname) + len + 1);
	sprintf(q,"%s=%s", envname + 1, args);
	status = putenv(q);
	if (status == 0)
	    args = envname;
	else
	{
	L1:
	    error("command line length of %d is too long",len);
	}
    }

    status = executearg0(cmd,args);
#if _WIN32
    if (status == -1)
	status = spawnlp(0,cmd,cmd,args,NULL);
#endif
//    if (global.params.verbose)
//	printf("\n");
    if (status)
    {
	if (status == -1)
	    printf("Can't run '%s', check PATH\n", cmd);
	else
	    printf("--- errorlevel %d\n", status);
    }
    return status;
}
#endif

/**************************************
 * Attempt to find command to execute by first looking in the directory
 * where DMD was run from.
 * Returns:
 *	-1	did not find command there
 *	!=-1	exit status from command
 */

#if _WIN32
int executearg0(char *cmd, char *args)
{
    char *file;
    char *argv0 = global.params.argv0;

    //printf("argv0='%s', cmd='%s', args='%s'\n",argv0,cmd,args);

    // If cmd is fully qualified, we don't do this
    if (FileName::absolute(cmd))
	return -1;

    file = FileName::replaceName(argv0, cmd);

    //printf("spawning '%s'\n",file);
#if _WIN32
    return spawnl(0,file,file,args,NULL);
#elif linux
    char *full;
    int cmdl = strlen(cmd);

    full = (char*) mem.malloc(cmdl + strlen(args) + 2);
    if (full == NULL)
	return 1;
    strcpy(full, cmd);
    full [cmdl] = ' ';
    strcpy(full + cmdl + 1, args);

    int result = system(full);

    mem.free(full);
    return result;
#else
    assert(0);
#endif
}
#endif

/***************************************
 * Run the compiled program.
 * Return exit status.
 */

int runProgram()
{
    //printf("runProgram()\n");
    if (global.params.verbose)
    {
	printf("%s", global.params.exefile);
	for (size_t i = 0; i < global.params.runargs_length; i++)
	    printf(" %s", (char *)global.params.runargs[i]);
	printf("\n");
    }

    // Build argv[]
    Array argv;

    argv.push((void *)global.params.exefile);
    for (size_t i = 0; i < global.params.runargs_length; i++)
    {	char *a = global.params.runargs[i];

#if _WIN32
	// BUG: what about " appearing in the string?
	if (strchr(a, ' '))
	{   char *b = (char *)mem.malloc(3 + strlen(a));
	    sprintf(b, "\"%s\"", a);
	    a = b;
	}
#endif
	argv.push((void *)a);
    }
    argv.push(NULL);

#if _WIN32
    char *ex = FileName::name(global.params.exefile);
    if (ex == global.params.exefile)
	ex = FileName::combine(".", ex);
    else
	ex = global.params.exefile;
    return spawnv(0,ex,(char **)argv.data);
#elif linux
    pid_t childpid;
    int status;

    childpid = fork();
    if (childpid == 0)
    {
	char *fn = (char *)argv.data[0];
	if (!FileName::absolute(fn))
	{   // Make it "./fn"
	    fn = FileName::combine(".", fn);
	}
	execv(fn, (char **)argv.data);
	perror(fn);		// failed to execute
	return -1;
    }

    waitpid(childpid, &status, 0);

    status = WEXITSTATUS(status);
    return status;
#else
    assert(0);
#endif
}
