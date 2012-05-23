
// Copyright (c) 1999-2011 by Digital Mars
// All Rights Reserved
// written by Walter Bright
// http://www.digitalmars.com
// License for redistribution is by either the Artistic License
// in artistic.txt, or the GNU General Public License in gnu.txt.
// See the included readme.txt for details.

#ifndef POSIX
#define POSIX (linux || __APPLE__ || __FreeBSD__ || __OpenBSD__ || __sun&&__SVR4)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#if (defined (__SVR4) && defined (__sun))
#include <alloca.h>
#endif

#if _MSC_VER ||__MINGW32__
#include <malloc.h>
#include <string>
#endif

#if _WIN32
#include <windows.h>
#include <direct.h>
#include <errno.h>
#endif

#ifdef __HAIKU__
#include <iostream>
#endif

#if POSIX
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <utime.h>
#endif

#include "port.h"
#include "root.h"
#include "dchar.h"
#include "rmem.h"
#include "mars.h"

#if 0 //__SC__ //def DEBUG
extern "C" void __cdecl _assert(void *e, void *f, unsigned line)
{
    printf("Assert('%s','%s',%d)\n",e,f,line);
    fflush(stdout);
    *(char *)0 = 0;
}
#endif


/*************************************
 * Convert wchar string to ascii string.
 */

char *wchar2ascii(wchar_t *us)
{
    return wchar2ascii(us, wcslen(us));
}

char *wchar2ascii(wchar_t *us, unsigned len)
{
    unsigned i;
    char *p;

    p = (char *)mem.malloc(len + 1);
    for (i = 0; i <= len; i++)
        p[i] = (char) us[i];
    return p;
}

int wcharIsAscii(wchar_t *us)
{
    return wcharIsAscii(us, wcslen(us));
}

int wcharIsAscii(wchar_t *us, unsigned len)
{
    unsigned i;

    for (i = 0; i <= len; i++)
    {
        if (us[i] & ~0xFF)      // if high bits set
            return 0;           // it's not ascii
    }
    return 1;
}


/***********************************
 * Compare length-prefixed strings (bstr).
 */

int bstrcmp(unsigned char *b1, unsigned char *b2)
{
    return (*b1 == *b2 && memcmp(b1 + 1, b2 + 1, *b2) == 0) ? 0 : 1;
}

/***************************************
 * Convert bstr into a malloc'd string.
 */

char *bstr2str(unsigned char *b)
{
    char *s;
    unsigned len;

    len = *b;
    s = (char *) mem.malloc(len + 1);
    s[len] = 0;
    return (char *)memcpy(s,b + 1,len);
}

/**************************************
 * Print error message and exit.
 */

void error(const char *format, ...)
{
    va_list ap;

    va_start(ap, format);
    printf("Error: ");
    vprintf(format, ap);
    va_end( ap );
    printf("\n");
    fflush(stdout);

    exit(EXIT_FAILURE);
}

#if M_UNICODE
void error(const dchar *format, ...)
{
    va_list ap;

    va_start(ap, format);
    printf("Error: ");
    vwprintf(format, ap);
    va_end( ap );
    printf("\n");
    fflush(stdout);

    exit(EXIT_FAILURE);
}
#endif

void error_mem()
{
    error("out of memory");
}

/**************************************
 * Print warning message.
 */

void warning(const char *format, ...)
{
    va_list ap;

    va_start(ap, format);
    printf("Warning: ");
    vprintf(format, ap);
    va_end( ap );
    printf("\n");
    fflush(stdout);
}

/****************************** Object ********************************/

int Object::equals(Object *o)
{
    return o == this;
}

hash_t Object::hashCode()
{
    return (hash_t) this;
}

int Object::compare(Object *obj)
{
    return this - obj;
}

void Object::print()
{
    printf("%s %p\n", toChars(), this);
}

char *Object::toChars()
{
    return (char *)"Object";
}

dchar *Object::toDchars()
{
#if M_UNICODE
    return L"Object";
#else
    return toChars();
#endif
}

int Object::dyncast()
{
    return 0;
}

void Object::toBuffer(OutBuffer *b)
{
    b->writestring("Object");
}

void Object::mark()
{
}

/****************************** String ********************************/

String::String(char *str, int ref)
{
    this->str = ref ? str : mem.strdup(str);
    this->ref = ref;
}

String::~String()
{
    mem.free(str);
}

void String::mark()
{
    mem.mark(str);
}

hash_t String::calcHash(const char *str, size_t len)
{
    hash_t hash = 0;

    for (;;)
    {
        switch (len)
        {
            case 0:
                return hash;

            case 1:
                hash *= 37;
                hash += *(uint8_t *)str;
                return hash;

            case 2:
                hash *= 37;
                hash += *(uint16_t *)str;
                return hash;

            case 3:
                hash *= 37;
                hash += (*(uint16_t *)str << 8) +
                        ((uint8_t *)str)[2];
                return hash;

            default:
                hash *= 37;
                hash += *(uint32_t *)str;
                str += 4;
                len -= 4;
                break;
        }
    }
}

hash_t String::calcHash(const char *str)
{
    return calcHash(str, strlen(str));
}

hash_t String::hashCode()
{
    return calcHash(str, strlen(str));
}

unsigned String::len()
{
    return strlen(str);
}

int String::equals(Object *obj)
{
    return strcmp(str,((String *)obj)->str) == 0;
}

int String::compare(Object *obj)
{
    return strcmp(str,((String *)obj)->str);
}

char *String::toChars()
{
    return str;
}

void String::print()
{
    printf("String '%s'\n",str);
}


/****************************** FileName ********************************/

FileName::FileName(char *str, int ref)
    : String(str,ref)
{
}

char *FileName::combine(const char *path, const char *name)
{   char *f;
    size_t pathlen;
    size_t namelen;

    if (!path || !*path)
        return (char *)name;
    pathlen = strlen(path);
    namelen = strlen(name);
    f = (char *)mem.malloc(pathlen + 1 + namelen + 1);
    memcpy(f, path, pathlen);
#if POSIX
    if (path[pathlen - 1] != '/')
    {   f[pathlen] = '/';
        pathlen++;
    }
#elif _WIN32
    if (path[pathlen - 1] != '\\' &&
        path[pathlen - 1] != '/'  &&
        path[pathlen - 1] != ':')
    {   f[pathlen] = '\\';
        pathlen++;
    }
#else
    assert(0);
#endif
    memcpy(f + pathlen, name, namelen + 1);
    return f;
}

FileName::FileName(char *path, char *name)
    : String(combine(path,name),1)
{
}

// Split a path into an Array of paths
Strings *FileName::splitPath(const char *path)
{
    char c = 0;                         // unnecessary initializer is for VC /W4
    const char *p;
    OutBuffer buf;
    Strings *array;

    array = new Strings();
    if (path)
    {
        p = path;
        do
        {   char instring = 0;

            while (isspace((unsigned char)*p))         // skip leading whitespace
                p++;
            buf.reserve(strlen(p) + 1); // guess size of path
            // LDC remember first character
            const char* start = p;
            for (; ; p++)
            {
                c = *p;
                switch (c)
                {
                    case '"':
                        instring ^= 1;  // toggle inside/outside of string
                        continue;

#if MACINTOSH
                    case ',':
#endif
#if _WIN32
                    case ';':
#endif
#if POSIX
                    case ':':
#endif
                        p++;
                        break;          // note that ; cannot appear as part
                                        // of a path, quotes won't protect it

                    case 0x1A:          // ^Z means end of file
                    case 0:
                        break;

                    case '\r':
                        continue;       // ignore carriage returns

#if POSIX
                    case '~':
                        // LDC don't expand unless first character of path
                        if (p != start)
                            goto Ldefault;
                        buf.writestring(getenv("HOME"));
                        continue;
#endif

#if 0
                    case ' ':
                    case '\t':          // tabs in filenames?
                        if (!instring)  // if not in string
                            break;      // treat as end of path
#endif
                    default:
                    Ldefault:
                        buf.writeByte(c);
                        continue;
                }
                break;
            }
            if (buf.offset)             // if path is not empty
            {
                buf.writeByte(0);       // to asciiz
                array->push(buf.extractData());
            }
        } while (c);
    }
    return array;
}

hash_t FileName::hashCode()
{
#if _WIN32
    // We need a different hashCode because it must be case-insensitive
    size_t len = strlen(str);
    hash_t hash = 0;
    unsigned char *s = (unsigned char *)str;

    for (;;)
    {
        switch (len)
        {
            case 0:
                return hash;

            case 1:
                hash *= 37;
                hash += *(uint8_t *)s | 0x20;
                return hash;

            case 2:
                hash *= 37;
                hash += *(uint16_t *)s | 0x2020;
                return hash;

            case 3:
                hash *= 37;
                hash += ((*(uint16_t *)s << 8) +
                         ((uint8_t *)s)[2]) | 0x202020;
                break;

            default:
                hash *= 37;
                hash += *(uint32_t *)s | 0x20202020;
                s += 4;
                len -= 4;
                break;
        }
    }
#else
    // darwin HFS is case insensitive, though...
    return String::hashCode();
#endif
}

int FileName::compare(Object *obj)
{
    return compare(str, ((FileName *)obj)->str);
}

int FileName::compare(const char *name1, const char *name2)
{
#if _WIN32
    return stricmp(name1, name2);
#else
    return strcmp(name1, name2);
#endif
}

int FileName::equals(Object *obj)
{
    return compare(obj) == 0;
}

int FileName::equals(const char *name1, const char *name2)
{
    return compare(name1, name2) == 0;
}

/************************************
 * Return !=0 if absolute path name.
 */

int FileName::absolute(const char *name)
{
#if _WIN32
    return (*name == '\\') ||
           (*name == '/')  ||
           (*name && name[1] == ':');
#elif POSIX
    return (*name == '/');
#else
    assert(0);
#endif
}

/********************************
 * Return filename extension (read-only).
 * Points past '.' of extension.
 * If there isn't one, return NULL.
 */

char *FileName::ext(const char *str)
{
    char *e;
    size_t len = strlen(str);

    e = (char *)str + len;
    for (;;)
    {
        switch (*e)
        {   case '.':
                return e + 1;
#if POSIX
            case '/':
                break;
#endif
#if _WIN32
            case '\\':
            case ':':
            case '/':
                break;
#endif
            default:
                if (e == str)
                    break;
                e--;
                continue;
        }
        return NULL;
    }
}

char *FileName::ext()
{
    return ext(str);
}

/********************************
 * Return mem.malloc'd filename with extension removed.
 */

char *FileName::removeExt(const char *str)
{
    const char *e = ext(str);
    if (e)
    {   size_t len = (e - str) - 1;
        char *n = (char *)mem.malloc(len + 1);
        memcpy(n, str, len);
        n[len] = 0;
        return n;
    }
    return mem.strdup(str);
}

/********************************
 * Return filename name excluding path (read-only).
 */

char *FileName::name(const char *str)
{
    char *e;
    size_t len = strlen(str);

    e = (char *)str + len;
    for (;;)
    {
        switch (*e)
        {
#if POSIX
            case '/':
               return e + 1;
#endif
#if _WIN32
            case '/':
            case '\\':
                return e + 1;
            case ':':
                /* The ':' is a drive letter only if it is the second
                 * character or the last character,
                 * otherwise it is an ADS (Alternate Data Stream) separator.
                 * Consider ADS separators as part of the file name.
                 */
                if (e == str + 1 || e == str + len - 1)
                    return e + 1;
#endif
            default:
                if (e == str)
                    break;
                e--;
                continue;
        }
        return e;
    }
}

char *FileName::name()
{
    return name(str);
}

/**************************************
 * Return path portion of str.
 * Path will does not include trailing path separator.
 */

char *FileName::path(const char *str)
{
    char *n = name(str);
    char *path;
    size_t pathlen;

    if (n > str)
    {
#if POSIX
        if (n[-1] == '/')
            n--;
#elif _WIN32
        if (n[-1] == '\\' || n[-1] == '/')
            n--;
#else
        assert(0);
#endif
    }
    pathlen = n - str;
    path = (char *)mem.malloc(pathlen + 1);
    memcpy(path, str, pathlen);
    path[pathlen] = 0;
    return path;
}

/**************************************
 * Replace filename portion of path.
 */

const char *FileName::replaceName(const char *path, const char *name)
{   char *f;
    char *n;
    size_t pathlen;
    size_t namelen;

    if (absolute(name))
        return name;

    n = FileName::name(path);
    if (n == path)
        return name;
    pathlen = n - path;
    namelen = strlen(name);
    f = (char *)mem.malloc(pathlen + 1 + namelen + 1);
    memcpy(f, path, pathlen);
#if POSIX
    if (path[pathlen - 1] != '/')
    {   f[pathlen] = '/';
        pathlen++;
    }
#elif _WIN32
    if (path[pathlen - 1] != '\\' &&
        path[pathlen - 1] != '/' &&
        path[pathlen - 1] != ':')
    {   f[pathlen] = '\\';
        pathlen++;
    }
#else
    assert(0);
#endif
    memcpy(f + pathlen, name, namelen + 1);
    return f;
}

/***************************
 */

FileName *FileName::defaultExt(const char *name, const char *ext)
{
    char *e;
    char *s;
    size_t len;
    size_t extlen;

    e = FileName::ext(name);
    if (e)                              // if already has an extension
        return new FileName((char *)name, 0);

    len = strlen(name);
    extlen = strlen(ext);
    s = (char *)alloca(len + 1 + extlen + 1);
    memcpy(s,name,len);
    s[len] = '.';
    memcpy(s + len + 1, ext, extlen + 1);
    return new FileName(s, 0);
}

/***************************
 */

FileName *FileName::forceExt(const char *name, const char *ext)
{
    char *e;
    char *s;
    size_t len;
    size_t extlen;

    e = FileName::ext(name);
    if (e)                              // if already has an extension
    {
        len = e - name;
        extlen = strlen(ext);

        s = (char *)alloca(len + extlen + 1);
        memcpy(s,name,len);
        memcpy(s + len, ext, extlen + 1);
        return new FileName(s, 0);
    }
    else
        return defaultExt(name, ext);   // doesn't have one
}

/******************************
 * Return !=0 if extensions match.
 */

int FileName::equalsExt(const char *ext)
{   const char *e;

    e = FileName::ext();
    if (!e && !ext)
        return 1;
    if (!e || !ext)
        return 0;
#if POSIX
    return strcmp(e,ext) == 0;
#elif _WIN32
    return stricmp(e,ext) == 0;
#else
    assert(0);
#endif
}

/*************************************
 * Copy file from this to to.
 */

void FileName::CopyTo(FileName *to)
{
    File file(this);

#if _WIN32
    file.touchtime = mem.malloc(sizeof(WIN32_FIND_DATAA));      // keep same file time
#elif POSIX
    file.touchtime = mem.malloc(sizeof(struct stat)); // keep same file time
#else
    assert(0);
#endif
    file.readv();
    file.name = to;
    file.writev();
}

/*************************************
 * Search Path for file.
 * Input:
 *      cwd     if !=0, search current directory before searching path
 */

char *FileName::searchPath(Strings *path, const char *name, int cwd)
{
    if (absolute(name))
    {
        return exists(name) ? (char *)name : NULL;
    }
    if (cwd)
    {
        if (exists(name))
            return (char *)name;
    }
    if (path)
    {   unsigned i;

        for (i = 0; i < path->dim; i++)
        {
            char *p = path->tdata()[i];
            char *n = combine(p, name);

            if (exists(n))
                return n;
        }
    }
    return NULL;
}


/*************************************
 * Search Path for file in a safe manner.
 *
 * Be wary of CWE-22: Improper Limitation of a Pathname to a Restricted Directory
 * ('Path Traversal') attacks.
 *      http://cwe.mitre.org/data/definitions/22.html
 * More info:
 *      https://www.securecoding.cert.org/confluence/display/seccode/FIO02-C.+Canonicalize+path+names+originating+from+untrusted+sources
 * Returns:
 *      NULL    file not found
 *      !=NULL  mem.malloc'd file name
 */

char *FileName::safeSearchPath(Strings *path, const char *name)
{
#if _WIN32
    /* Disallow % / \ : and .. in name characters
     */
    for (const char *p = name; *p; p++)
    {
        char c = *p;
        if (c == '\\' || c == '/' || c == ':' || c == '%' ||
            (c == '.' && p[1] == '.'))
        {
            return NULL;
        }
    }

    return FileName::searchPath(path, name, 0);
#elif POSIX
    /* Even with realpath(), we must check for // and disallow it
     */
    for (const char *p = name; *p; p++)
    {
        char c = *p;
        if (c == '/' && p[1] == '/')
        {
            return NULL;
        }
    }

    if (path)
    {   unsigned i;

        /* Each path is converted to a cannonical name and then a check is done to see
         * that the searched name is really a child one of the the paths searched.
         */
        for (i = 0; i < path->dim; i++)
        {
            char *cname = NULL;
            char *cpath = canonicalName(path->tdata()[i]);
            //printf("FileName::safeSearchPath(): name=%s; path=%s; cpath=%s\n",
            //      name, (char *)path->data[i], cpath);
            if (cpath == NULL)
                goto cont;
            cname = canonicalName(combine(cpath, name));
            //printf("FileName::safeSearchPath(): cname=%s\n", cname);
            if (cname == NULL)
                goto cont;
            //printf("FileName::safeSearchPath(): exists=%i "
            //      "strncmp(cpath, cname, %i)=%i\n", exists(cname),
            //      strlen(cpath), strncmp(cpath, cname, strlen(cpath)));
            // exists and name is *really* a "child" of path
            if (exists(cname) && strncmp(cpath, cname, strlen(cpath)) == 0)
            {
                free(cpath);
                char *p = mem.strdup(cname);
                free(cname);
                return p;
            }
cont:
            if (cpath)
                free(cpath);
            if (cname)
                free(cname);
        }
    }
    return NULL;
#else
    assert(0);
#endif
}


int FileName::exists(const char *name)
{
#if POSIX
    struct stat st;

    if (stat(name, &st) < 0)
        return 0;
    if (S_ISDIR(st.st_mode))
        return 2;
    return 1;
#elif _WIN32
    DWORD dw;
    int result;

    dw = GetFileAttributesA(name);
    if (dw == -1L)
        result = 0;
    else if (dw & FILE_ATTRIBUTE_DIRECTORY)
        result = 2;
    else
        result = 1;
    return result;
#else
    assert(0);
#endif
}

void FileName::ensurePathExists(const char *path)
{
    //printf("FileName::ensurePathExists(%s)\n", path ? path : "");
    if (path && *path)
    {
        if (!exists(path))
        {
            char *p = FileName::path(path);
            if (*p)
            {
#if _WIN32
                size_t len = strlen(path);
                if (len > 2 && p[-1] == ':' && path + 2 == p)
                {   mem.free(p);
                    return;
                }
#endif
                ensurePathExists(p);
                mem.free(p);
            }
#if _WIN32
            if (path[strlen(path) - 1] != '\\')
#endif
#if POSIX
            if (path[strlen(path) - 1] != '\\')
#endif
            {
                //printf("mkdir(%s)\n", path);
#if _WIN32
                if (_mkdir(path))
#endif
#if POSIX
                if (mkdir(path, 0777))
#endif
                {
                    /* Don't error out if another instance of dmd just created
                     * this directory
                     */
                    if (errno != EEXIST)
                        error("cannot create directory %s", path);
                }
            }
        }
    }
}


/******************************************
 * Return canonical version of name in a malloc'd buffer.
 * This code is high risk.
 */
char *FileName::canonicalName(const char *name)
{
#if linux
    // Lovely glibc extension to do it for us
    return canonicalize_file_name(name);
#elif POSIX
  #if _POSIX_VERSION >= 200809L || defined (linux)
    // NULL destination buffer is allowed and preferred
    return realpath(name, NULL);
  #else
    char *cname = NULL;
    #if PATH_MAX
        /* PATH_MAX must be defined as a constant in <limits.h>,
         * otherwise using it is unsafe due to TOCTOU
         */
        size_t path_max = (size_t)PATH_MAX;
        if (path_max > 0)
        {
            /* Need to add one to PATH_MAX because of realpath() buffer overflow bug:
             * http://isec.pl/vulnerabilities/isec-0011-wu-ftpd.txt
             */
            cname = (char *)malloc(path_max + 1);
            if (cname == NULL)
                return NULL;
        }
    #endif
    return realpath(name, cname);
  #endif
#elif _WIN32
    /* Apparently, there is no good way to do this on Windows.
     * GetFullPathName isn't it.
     */
    assert(0);
    return NULL;
#else
    assert(0);
    return NULL;
#endif
}


/****************************** File ********************************/

File::File(FileName *n)
{
    ref = 0;
    buffer = NULL;
    len = 0;
    touchtime = NULL;
    name = n;
}

File::File(char *n)
{
    ref = 0;
    buffer = NULL;
    len = 0;
    touchtime = NULL;
    name = new FileName(n, 0);
}

File::~File()
{
    if (buffer)
    {
        if (ref == 0)
            mem.free(buffer);
#if _WIN32
        else if (ref == 2)
            UnmapViewOfFile(buffer);
#endif
    }
    if (touchtime)
        mem.free(touchtime);
}

void File::mark()
{
    mem.mark(buffer);
    mem.mark(touchtime);
    mem.mark(name);
}

/*************************************
 */

int File::read()
{
#if POSIX
    off_t size;
    ssize_t numread;
    int fd;
    struct stat buf;
    int result = 0;
    char *name;

    name = this->name->toChars();
    //printf("File::read('%s')\n",name);
    fd = open(name, O_RDONLY);
    if (fd == -1)
    {
        //printf("\topen error, errno = %d\n",errno);
        goto err1;
    }

    if (!ref)
        ::free(buffer);
    ref = 0;       // we own the buffer now

    //printf("\tfile opened\n");
    if (fstat(fd, &buf))
    {
        printf("\tfstat error, errno = %d\n",errno);
        goto err2;
    }
    size = buf.st_size;
    buffer = (unsigned char *) ::malloc(size + 2);
    if (!buffer)
    {
        printf("\tmalloc error, errno = %d\n",errno);
        goto err2;
    }

    numread = ::read(fd, buffer, size);
    if (numread != size)
    {
        printf("\tread error, errno = %d\n",errno);
        goto err2;
    }

    if (touchtime)
        memcpy(touchtime, &buf, sizeof(buf));

    if (close(fd) == -1)
    {
        printf("\tclose error, errno = %d\n",errno);
        goto err;
    }

    len = size;

    // Always store a wchar ^Z past end of buffer so scanner has a sentinel
    buffer[size] = 0;           // ^Z is obsolete, use 0
    buffer[size + 1] = 0;
    return 0;

err2:
    close(fd);
err:
    ::free(buffer);
    buffer = NULL;
    len = 0;

err1:
    result = 1;
    return result;
#elif _WIN32
    DWORD size;
    DWORD numread;
    HANDLE h;
    int result = 0;
    char *name;

    name = this->name->toChars();
    h = CreateFileA(name,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,0);
    if (h == INVALID_HANDLE_VALUE)
        goto err1;

    if (!ref)
        ::free(buffer);
    ref = 0;

    size = GetFileSize(h,NULL);
    buffer = (unsigned char *) ::malloc(size + 2);
    if (!buffer)
        goto err2;

    if (ReadFile(h,buffer,size,&numread,NULL) != TRUE)
        goto err2;

    if (numread != size)
        goto err2;

    if (touchtime)
    {
        if (!GetFileTime(h, NULL, NULL, &((WIN32_FIND_DATAA *)touchtime)->ftLastWriteTime))
            goto err2;
    }

    if (!CloseHandle(h))
        goto err;

    len = size;

    // Always store a wchar ^Z past end of buffer so scanner has a sentinel
    buffer[size] = 0;           // ^Z is obsolete, use 0
    buffer[size + 1] = 0;
    return 0;

err2:
    CloseHandle(h);
err:
    ::free(buffer);
    buffer = NULL;
    len = 0;

err1:
    result = 1;
    return result;
#else
    assert(0);
#endif
}

/*****************************
 * Read a file with memory mapped file I/O.
 */

int File::mmread()
{
#if POSIX
    return read();
#elif _WIN32
    HANDLE hFile;
    HANDLE hFileMap;
    DWORD size;
    char *name;

    name = this->name->toChars();
    hFile = CreateFile(name, GENERIC_READ,
                        FILE_SHARE_READ, NULL,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        goto Lerr;
    size = GetFileSize(hFile, NULL);
    //printf(" file created, size %d\n", size);

    hFileMap = CreateFileMapping(hFile,NULL,PAGE_READONLY,0,size,NULL);
    if (CloseHandle(hFile) != TRUE)
        goto Lerr;

    if (hFileMap == NULL)
        goto Lerr;

    //printf(" mapping created\n");

    if (!ref)
        mem.free(buffer);
    ref = 2;
    buffer = (unsigned char *)MapViewOfFileEx(hFileMap, FILE_MAP_READ,0,0,size,NULL);
    if (CloseHandle(hFileMap) != TRUE)
        goto Lerr;
    if (buffer == NULL)                 // mapping view failed
        goto Lerr;

    len = size;
    //printf(" buffer = %p\n", buffer);

    return 0;

Lerr:
    return GetLastError();                      // failure
#else
    assert(0);
#endif
}

/*********************************************
 * Write a file.
 * Returns:
 *      0       success
 */

int File::write()
{
#if POSIX
    int fd;
    ssize_t numwritten;
    char *name;

    name = this->name->toChars();
    fd = open(name, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd == -1)
        goto err;

    numwritten = ::write(fd, buffer, len);
    if (len != numwritten)
        goto err2;

    if (close(fd) == -1)
        goto err;

    if (touchtime)
    {   struct utimbuf ubuf;

        ubuf.actime = ((struct stat *)touchtime)->st_atime;
        ubuf.modtime = ((struct stat *)touchtime)->st_mtime;
        if (utime(name, &ubuf))
            goto err;
    }
    return 0;

err2:
    close(fd);
    ::remove(name);
err:
    return 1;
#elif _WIN32
    HANDLE h;
    DWORD numwritten;
    char *name;

    name = this->name->toChars();
    h = CreateFileA(name,GENERIC_WRITE,0,NULL,CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,NULL);
    if (h == INVALID_HANDLE_VALUE)
        goto err;

    if (WriteFile(h,buffer,len,&numwritten,NULL) != TRUE)
        goto err2;

    if (len != numwritten)
        goto err2;

    if (touchtime) {
        SetFileTime(h, NULL, NULL, &((WIN32_FIND_DATAA *)touchtime)->ftLastWriteTime);
    }
    if (!CloseHandle(h))
        goto err;
    return 0;

err2:
    CloseHandle(h);
    DeleteFileA(name);
err:
    return 1;
#else
    assert(0);
#endif
}

/*********************************************
 * Append to a file.
 * Returns:
 *      0       success
 */

int File::append()
{
#if POSIX
    return 1;
#elif _WIN32
    HANDLE h;
    DWORD numwritten;
    char *name;

    name = this->name->toChars();
    h = CreateFileA(name,GENERIC_WRITE,0,NULL,OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,NULL);
    if (h == INVALID_HANDLE_VALUE)
        goto err;

#if 1
    SetFilePointer(h, 0, NULL, FILE_END);
#else // INVALID_SET_FILE_POINTER doesn't seem to have a definition
    if (SetFilePointer(h, 0, NULL, FILE_END) == INVALID_SET_FILE_POINTER)
        goto err;
#endif

    if (WriteFile(h,buffer,len,&numwritten,NULL) != TRUE)
        goto err2;

    if (len != numwritten)
        goto err2;

    if (touchtime) {
        SetFileTime(h, NULL, NULL, &((WIN32_FIND_DATAA *)touchtime)->ftLastWriteTime);
    }
    if (!CloseHandle(h))
        goto err;
    return 0;

err2:
    CloseHandle(h);
err:
    return 1;
#else
    assert(0);
#endif
}

/**************************************
 */

void File::readv()
{
    if (read())
        error("Error reading file '%s'\n",name->toChars());
}

/**************************************
 */

void File::mmreadv()
{
    if (mmread())
        readv();
}

void File::writev()
{
    if (write())
        error("Error writing file '%s'\n",name->toChars());
}

void File::appendv()
{
    if (write())
        error("Error appending to file '%s'\n",name->toChars());
}

/*******************************************
 * Return !=0 if file exists.
 *      0:      file doesn't exist
 *      1:      normal file
 *      2:      directory
 */

int File::exists()
{
#if POSIX
    return 0;
#elif _WIN32
    DWORD dw;
    int result;
    char *name;

    name = this->name->toChars();
    if (touchtime)
        dw = ((WIN32_FIND_DATAA *)touchtime)->dwFileAttributes;
    else
        dw = GetFileAttributesA(name);
    if (dw == -1L)
        result = 0;
    else if (dw & FILE_ATTRIBUTE_DIRECTORY)
        result = 2;
    else
        result = 1;
    return result;
#else
    assert(0);
#endif
}

void File::remove()
{
#if POSIX
    ::remove(this->name->toChars());
#elif _WIN32
    DeleteFileA(this->name->toChars());
#else
    assert(0);
#endif
}

Files *File::match(char *n)
{
    return match(new FileName(n, 0));
}

Files *File::match(FileName *n)
{
#if POSIX
    return NULL;
#elif _WIN32
    HANDLE h;
    WIN32_FIND_DATAA fileinfo;
    Files *a;
    char *c;
    char *name;

    a = new Files();
    c = n->toChars();
    name = n->name();
    h = FindFirstFileA(c,&fileinfo);
    if (h != INVALID_HANDLE_VALUE)
    {
        do
        {
            // Glue path together with name
            char *fn;
            File *f;

            fn = (char *)mem.malloc(name - c + strlen(fileinfo.cFileName) + 1);
            memcpy(fn, c, name - c);
            strcpy(fn + (name - c), fileinfo.cFileName);
            f = new File(fn);
            f->touchtime = mem.malloc(sizeof(WIN32_FIND_DATAA));
            memcpy(f->touchtime, &fileinfo, sizeof(fileinfo));
            a->push(f);
        } while (FindNextFileA(h,&fileinfo) != FALSE);
        FindClose(h);
    }
    return a;
#else
    assert(0);
#endif
}

int File::compareTime(File *f)
{
#if POSIX
    return 0;
#elif _WIN32
    if (!touchtime)
        stat();
    if (!f->touchtime)
        f->stat();
    return CompareFileTime(&((WIN32_FIND_DATAA *)touchtime)->ftLastWriteTime, &((WIN32_FIND_DATAA *)f->touchtime)->ftLastWriteTime);
#else
    assert(0);
#endif
}

void File::stat()
{
#if POSIX
    if (!touchtime)
    {
        touchtime = mem.calloc(1, sizeof(struct stat));
    }
#elif _WIN32
    HANDLE h;

    if (!touchtime)
    {
        touchtime = mem.calloc(1, sizeof(WIN32_FIND_DATAA));
    }
    h = FindFirstFileA(name->toChars(),(WIN32_FIND_DATAA *)touchtime);
    if (h != INVALID_HANDLE_VALUE)
    {
        FindClose(h);
    }
#else
    assert(0);
#endif
}

void File::checkoffset(size_t offset, size_t nbytes)
{
    if (offset > len || offset + nbytes > len)
        error("Corrupt file '%s': offset x%llx off end of file",toChars(),(ulonglong)offset);
}

char *File::toChars()
{
    return name->toChars();
}


/************************* OutBuffer *************************/

OutBuffer::OutBuffer()
{
    data = NULL;
    offset = 0;
    size = 0;
}

OutBuffer::~OutBuffer()
{
    mem.free(data);
}

char *OutBuffer::extractData()
{
    char *p;

    p = (char *)data;
    data = NULL;
    offset = 0;
    size = 0;
    return p;
}

void OutBuffer::mark()
{
    mem.mark(data);
}

void OutBuffer::reserve(unsigned nbytes)
{
    //printf("OutBuffer::reserve: size = %d, offset = %d, nbytes = %d\n", size, offset, nbytes);
    if (size - offset < nbytes)
    {
#if defined (__x86_64__)
        size = (offset + nbytes) * 2 + 2;
#else
        size = (offset + nbytes) * 2;
#endif
        data = (unsigned char *)mem.realloc(data, size);
    }
}

void OutBuffer::reset()
{
    offset = 0;
}

void OutBuffer::setsize(unsigned size)
{
    offset = size;
}

void OutBuffer::write(const void *data, unsigned nbytes)
{
    reserve(nbytes);
    memcpy(this->data + offset, data, nbytes);
    offset += nbytes;
}

void OutBuffer::writebstring(unsigned char *string)
{
    write(string,*string + 1);
}

void OutBuffer::writestring(const char *string)
{
    write(string,strlen(string));
}

void OutBuffer::writedstring(const char *string)
{
#if M_UNICODE
    for (; *string; string++)
    {
        writedchar(*string);
    }
#else
    write(string,strlen(string));
#endif
}

void OutBuffer::writedstring(const wchar_t *string)
{
#if M_UNICODE
    write(string,wcslen(string) * sizeof(wchar_t));
#else
    for (; *string; string++)
    {
        writedchar(*string);
    }
#endif
}

void OutBuffer::prependstring(const char *string)
{   unsigned len;

    len = strlen(string);
    reserve(len);
    memmove(data + len, data, offset);
    memcpy(data, string, len);
    offset += len;
}

void OutBuffer::writenl()
{
#if _WIN32
#if M_UNICODE
    write4(0x000A000D);         // newline is CR,LF on Microsoft OS's
#else
    writeword(0x0A0D);          // newline is CR,LF on Microsoft OS's
#endif
#else
#if M_UNICODE
    writeword('\n');
#else
    writeByte('\n');
#endif
#endif
}

void OutBuffer::writeByte(unsigned b)
{
    reserve(1);
    this->data[offset] = (unsigned char)b;
    offset++;
}

void OutBuffer::writeUTF8(unsigned b)
{
    reserve(6);
    if (b <= 0x7F)
    {
        this->data[offset] = (unsigned char)b;
        offset++;
    }
    else if (b <= 0x7FF)
    {
        this->data[offset + 0] = (unsigned char)((b >> 6) | 0xC0);
        this->data[offset + 1] = (unsigned char)((b & 0x3F) | 0x80);
        offset += 2;
    }
    else if (b <= 0xFFFF)
    {
        this->data[offset + 0] = (unsigned char)((b >> 12) | 0xE0);
        this->data[offset + 1] = (unsigned char)(((b >> 6) & 0x3F) | 0x80);
        this->data[offset + 2] = (unsigned char)((b & 0x3F) | 0x80);
        offset += 3;
    }
    else if (b <= 0x1FFFFF)
    {
        this->data[offset + 0] = (unsigned char)((b >> 18) | 0xF0);
        this->data[offset + 1] = (unsigned char)(((b >> 12) & 0x3F) | 0x80);
        this->data[offset + 2] = (unsigned char)(((b >> 6) & 0x3F) | 0x80);
        this->data[offset + 3] = (unsigned char)((b & 0x3F) | 0x80);
        offset += 4;
    }
    else if (b <= 0x3FFFFFF)
    {
        this->data[offset + 0] = (unsigned char)((b >> 24) | 0xF8);
        this->data[offset + 1] = (unsigned char)(((b >> 18) & 0x3F) | 0x80);
        this->data[offset + 2] = (unsigned char)(((b >> 12) & 0x3F) | 0x80);
        this->data[offset + 3] = (unsigned char)(((b >> 6) & 0x3F) | 0x80);
        this->data[offset + 4] = (unsigned char)((b & 0x3F) | 0x80);
        offset += 5;
    }
    else if (b <= 0x7FFFFFFF)
    {
        this->data[offset + 0] = (unsigned char)((b >> 30) | 0xFC);
        this->data[offset + 1] = (unsigned char)(((b >> 24) & 0x3F) | 0x80);
        this->data[offset + 2] = (unsigned char)(((b >> 18) & 0x3F) | 0x80);
        this->data[offset + 3] = (unsigned char)(((b >> 12) & 0x3F) | 0x80);
        this->data[offset + 4] = (unsigned char)(((b >> 6) & 0x3F) | 0x80);
        this->data[offset + 5] = (unsigned char)((b & 0x3F) | 0x80);
        offset += 6;
    }
    else
        assert(0);
}

void OutBuffer::writedchar(unsigned b)
{
    reserve(Dchar_mbmax * sizeof(dchar));
    offset = (unsigned char *)Dchar::put((dchar *)(this->data + offset), (dchar)b) -
                this->data;
}

void OutBuffer::prependbyte(unsigned b)
{
    reserve(1);
    memmove(data + 1, data, offset);
    data[0] = (unsigned char)b;
    offset++;
}

void OutBuffer::writeword(unsigned w)
{
    reserve(2);
    *(unsigned short *)(this->data + offset) = (unsigned short)w;
    offset += 2;
}

void OutBuffer::writeUTF16(unsigned w)
{
    reserve(4);
    if (w <= 0xFFFF)
    {
        *(unsigned short *)(this->data + offset) = (unsigned short)w;
        offset += 2;
    }
    else if (w <= 0x10FFFF)
    {
        *(unsigned short *)(this->data + offset) = (unsigned short)((w >> 10) + 0xD7C0);
        *(unsigned short *)(this->data + offset + 2) = (unsigned short)((w & 0x3FF) | 0xDC00);
        offset += 4;
    }
    else
        assert(0);
}

void OutBuffer::write4(unsigned w)
{
    reserve(4);
    *(unsigned *)(this->data + offset) = w;
    offset += 4;
}

void OutBuffer::write(OutBuffer *buf)
{
    if (buf)
    {   reserve(buf->offset);
        memcpy(data + offset, buf->data, buf->offset);
        offset += buf->offset;
    }
}

void OutBuffer::write(Object *obj)
{
    if (obj)
    {
        writestring(obj->toChars());
    }
}

void OutBuffer::fill0(unsigned nbytes)
{
    reserve(nbytes);
    memset(data + offset,0,nbytes);
    offset += nbytes;
}

void OutBuffer::align(unsigned size)
{   unsigned nbytes;

    nbytes = ((offset + size - 1) & ~(size - 1)) - offset;
    fill0(nbytes);
}


////////////////////////////////////////////////////////////////
// The compiler shipped with Visual Studio 2005 (and possible
// other versions) does not support C99 printf format specfiers
// such as %z and %j
#if 0 && _MSC_VER
using std::string;
using std::wstring;

template<typename S>
inline void
search_and_replace(S& str, const S& what, const S& replacement)
{
    assert(!what.empty());
    size_t pos = str.find(what);
    while (pos != S::npos)
    {
        str.replace(pos, what.size(), replacement);
        pos = str.find(what, pos + replacement.size());
    }
}
#define WORKAROUND_C99_SPECIFIERS_BUG(S,tmp,f) \
    S tmp = f;                                 \
    search_and_replace(fmt, S("%z"), S("%l")); \
    search_and_replace(fmt, S("%j"), S("%l")); \
    f = tmp.c_str();
#else
#define WORKAROUND_C99_SPECIFIERS_BUG(S,tmp,f)
#endif

void OutBuffer::vprintf(const char *format, va_list args)
{
    char buffer[128];
    char *p;
    unsigned psize;
    int count;

    WORKAROUND_C99_SPECIFIERS_BUG(string, fmt, format);

    p = buffer;
    psize = sizeof(buffer);
    for (;;)
    {
#if _WIN32
        count = _vsnprintf(p,psize,format,args);
        if (count != -1)
            break;
        psize *= 2;
#elif POSIX
        va_list va;
        va_copy(va, args);
/*
  The functions vprintf(), vfprintf(), vsprintf(), vsnprintf()
  are equivalent to the functions printf(), fprintf(), sprintf(),
  snprintf(), respectively, except that they are called with a
  va_list instead of a variable number of arguments. These
  functions do not call the va_end macro. Consequently, the value
  of ap is undefined after the call. The application should call
  va_end(ap) itself afterwards.
 */
        count = vsnprintf(p,psize,format,va);
        va_end(va);
        if (count == -1)
            psize *= 2;
        else if (count >= psize)
            psize = count + 1;
        else
            break;
#else
    assert(0);
#endif
        p = (char *) alloca(psize);     // buffer too small, try again with larger size
    }
    write(p,count);
}

#if M_UNICODE
void OutBuffer::vprintf(const wchar_t *format, va_list args)
{
    dchar buffer[128];
    dchar *p;
    unsigned psize;
    int count;

    WORKAROUND_C99_SPECIFIERS_BUG(wstring, fmt, format);

    p = buffer;
    psize = sizeof(buffer) / sizeof(buffer[0]);
    for (;;)
    {
#if _WIN32
        count = _vsnwprintf(p,psize,format,args);
        if (count != -1)
            break;
        psize *= 2;
#elif POSIX
        va_list va;
        va_copy(va, args);
        count = vsnwprintf(p,psize,format,va);
        va_end(va);

        if (count == -1)
            psize *= 2;
        else if (count >= psize)
            psize = count + 1;
        else
            break;
#else
    assert(0);
#endif
        p = (dchar *) alloca(psize * 2);        // buffer too small, try again with larger size
    }
    write(p,count * 2);
}
#endif

void OutBuffer::printf(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vprintf(format,ap);
    va_end(ap);
}

#if M_UNICODE
void OutBuffer::printf(const wchar_t *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vprintf(format,ap);
    va_end(ap);
}
#endif

void OutBuffer::bracket(char left, char right)
{
    reserve(2);
    memmove(data + 1, data, offset);
    data[0] = left;
    data[offset + 1] = right;
    offset += 2;
}

/******************
 * Insert left at i, and right at j.
 * Return index just past right.
 */

unsigned OutBuffer::bracket(unsigned i, const char *left, unsigned j, const char *right)
{
    size_t leftlen = strlen(left);
    size_t rightlen = strlen(right);
    reserve(leftlen + rightlen);
    insert(i, left, leftlen);
    insert(j + leftlen, right, rightlen);
    return j + leftlen + rightlen;
}

void OutBuffer::spread(unsigned offset, unsigned nbytes)
{
    reserve(nbytes);
    memmove(data + offset + nbytes, data + offset,
        this->offset - offset);
    this->offset += nbytes;
}

/****************************************
 * Returns: offset + nbytes
 */

unsigned OutBuffer::insert(unsigned offset, const void *p, unsigned nbytes)
{
    spread(offset, nbytes);
    memmove(data + offset, p, nbytes);
    return offset + nbytes;
}

void OutBuffer::remove(unsigned offset, unsigned nbytes)
{
    memmove(data + offset, data + offset + nbytes, this->offset - (offset + nbytes));
    this->offset -= nbytes;
}

char *OutBuffer::toChars()
{
    writeByte(0);
    return (char *)data;
}

/********************************* Bits ****************************/

Bits::Bits()
{
    data = NULL;
    bitdim = 0;
    allocdim = 0;
}

Bits::~Bits()
{
    mem.free(data);
}

void Bits::mark()
{
    mem.mark(data);
}

void Bits::resize(unsigned bitdim)
{
    unsigned allocdim;
    unsigned mask;

    allocdim = (bitdim + 31) / 32;
    data = (unsigned *)mem.realloc(data, allocdim * sizeof(data[0]));
    if (this->allocdim < allocdim)
        memset(data + this->allocdim, 0, (allocdim - this->allocdim) * sizeof(data[0]));

    // Clear other bits in last word
    mask = (1 << (bitdim & 31)) - 1;
    if (mask)
        data[allocdim - 1] &= ~mask;

    this->bitdim = bitdim;
    this->allocdim = allocdim;
}

void Bits::set(unsigned bitnum)
{
    data[bitnum / 32] |= 1 << (bitnum & 31);
}

void Bits::clear(unsigned bitnum)
{
    data[bitnum / 32] &= ~(1 << (bitnum & 31));
}

int Bits::test(unsigned bitnum)
{
    return data[bitnum / 32] & (1 << (bitnum & 31));
}

void Bits::set()
{   unsigned mask;

    memset(data, ~0, allocdim * sizeof(data[0]));

    // Clear other bits in last word
    mask = (1 << (bitdim & 31)) - 1;
    if (mask)
        data[allocdim - 1] &= mask;
}

void Bits::clear()
{
    memset(data, 0, allocdim * sizeof(data[0]));
}

void Bits::copy(Bits *from)
{
    assert(bitdim == from->bitdim);
    memcpy(data, from->data, allocdim * sizeof(data[0]));
}

Bits *Bits::clone()
{
    Bits *b;

    b = new Bits();
    b->resize(bitdim);
    b->copy(this);
    return b;
}

void Bits::sub(Bits *b)
{
    unsigned u;

    for (u = 0; u < allocdim; u++)
        data[u] &= ~b->data[u];
}















