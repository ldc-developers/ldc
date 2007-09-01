
/**
 * C's &lt;string.h&gt;
 * Authors: Walter Bright, Digital Mars, www.digitalmars.com
 * License: Public Domain
 * Macros:
 *	WIKI=Phobos/StdCString
 */

module std.c.string;

extern (C):

void* memcpy(void* s1, void* s2, size_t n);	///
void* memmove(void* s1, void* s2, size_t n);	///
char* strcpy(char* s1, char* s2);		///
char* strncpy(char* s1, char* s2, size_t n);	///
char* strncat(char*  s1, char*  s2, size_t n);	///
int strcoll(char* s1, char* s2);		///
int strncmp(char* s1, char* s2, size_t n);	///
size_t strxfrm(char*  s1, char*  s2, size_t n);	///
void* memchr(void* s, int c, size_t n);		///
char* strchr(char* s, int c);			///
size_t strcspn(char* s1, char* s2);		///
char* strpbrk(char* s1, char* s2);		///
char* strrchr(char* s, int c);			///
size_t strspn(char* s1, char* s2);		///
char* strstr(char* s1, char* s2);		///
char* strtok(char*  s1, char*  s2);		///
void* memset(void* s, int c, size_t n);		///
char* strerror(int errnum);			///
size_t strlen(char* s);				///
int strcmp(char* s1, char* s2);			///
char* strcat(char* s1, char* s2);		///
int memcmp(void* s1, void* s2, size_t n);	///

version (Windows)
{
    int memicmp(char* s1, char* s2, size_t n);	///
}
