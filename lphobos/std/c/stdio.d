
/**
 * C's &lt;stdio.h&gt;
 * Authors: Walter Bright, Digital Mars, www.digitalmars.com
 * License: Public Domain
 * Macros:
 *	WIKI=Phobos/StdCStdio
 */



module std.c.stdio;

import std.c.stddef;
private import std.c.stdarg;

extern (C):

version (Win32)
{
    const int _NFILE = 60;	///
    const int BUFSIZ = 0x4000;	///
    const int EOF = -1;		///
    const int FOPEN_MAX = 20;	///
    const int FILENAME_MAX = 256;  /// 255 plus NULL
    const int TMP_MAX = 32767;	///
    const int _SYS_OPEN = 20;	///
    const int SYS_OPEN = _SYS_OPEN;	///
    const wchar WEOF = 0xFFFF;		///
}

version (linux)
{
    const int EOF = -1;
    const int FOPEN_MAX = 16;
    const int FILENAME_MAX = 4095;
    const int TMP_MAX = 238328;
    const int L_tmpnam = 20;
}

enum { SEEK_SET, SEEK_CUR, SEEK_END }

struct _iobuf
{
    align (1):
    version (Win32)
    {
	char	*_ptr;
	int	_cnt;
	char	*_base;
	int	_flag;
	int	_file;
	int	_charbuf;
	int	_bufsiz;
	int	__tmpnum;
    }
    version (linux)
    {
	char*	_read_ptr;
	char*	_read_end;
	char*	_read_base;
	char*	_write_base;
	char*	_write_ptr;
	char*	_write_end;
	char*	_buf_base;
	char*	_buf_end;
	char*	_save_base;
	char*	_backup_base;
	char*	_save_end;
	void*	_markers;
	_iobuf*	_chain;
	int	_fileno;
	int	_blksize;
	int	_old_offset;
	ushort	_cur_column;
	byte	_vtable_offset;
	char[1]	_shortbuf;
	void*	_lock;
    }
}

alias _iobuf FILE;	///

enum
{
    _F_RDWR = 0x0003,
    _F_READ = 0x0001,
    _F_WRIT = 0x0002,
    _F_BUF  = 0x0004,
    _F_LBUF = 0x0008,
    _F_ERR  = 0x0010,
    _F_EOF  = 0x0020,
    _F_BIN  = 0x0040,
    _F_IN   = 0x0080,
    _F_OUT  = 0x0100,
    _F_TERM = 0x0200,
}

version (Win32)
{
    extern FILE _iob[_NFILE];
    extern void function() _fcloseallp;
    extern ubyte __fhnd_info[_NFILE];

    enum
    {
	FHND_APPEND	= 0x04,
	FHND_DEVICE	= 0x08,
	FHND_TEXT	= 0x10,
	FHND_BYTE	= 0x20,
	FHND_WCHAR	= 0x40,
    }
}

version (Win32)
{
    enum
    {
	    _IOREAD	= 1,
	    _IOWRT	= 2,
	    _IONBF	= 4,
	    _IOMYBUF	= 8,
	    _IOEOF	= 0x10,
	    _IOERR	= 0x20,
	    _IOLBF	= 0x40,
	    _IOSTRG	= 0x40,
	    _IORW	= 0x80,
	    _IOFBF	= 0,
	    _IOAPP	= 0x200,
	    _IOTRAN	= 0x100,
    }
}

version (linux)
{
    enum
    {
	    _IOFBF = 0,
	    _IOLBF = 1,
	    _IONBF = 2,
    }
}

version (Win32)
{
    const FILE *stdin  = &_iob[0];	///
    const FILE *stdout = &_iob[1];	///
    const FILE *stderr = &_iob[2];	///
    const FILE *stdaux = &_iob[3];	///
    const FILE *stdprn = &_iob[4];	///
}

version (linux)
{
    extern FILE *stdin;
    extern FILE *stdout;
    extern FILE *stderr;
}

version (Win32)
{
    const char[] _P_tmpdir = "\\";
    const wchar[] _wP_tmpdir = "\\";
    const int L_tmpnam = _P_tmpdir.length + 12;
}

alias int fpos_t;	///

char *	 tmpnam(char *);	///
FILE *	 fopen(char *,char *);	///
FILE *	 _fsopen(char *,char *,int );	///
FILE *	 freopen(char *,char *,FILE *);	///
int	 fseek(FILE *,int,int);	///
int	 ftell(FILE *);	///
char *	 fgets(char *,int,FILE *);	///
int	 fgetc(FILE *);	///
int	 _fgetchar();	///
int	 fflush(FILE *);	///
int	 fclose(FILE *);	///
int	 fputs(char *,FILE *);	///
char *	 gets(char *);	///
int	 fputc(int,FILE *);	///
int	 _fputchar(int);	///
int	 puts(char *);	///
int	 ungetc(int,FILE *);	///
size_t	 fread(void *,size_t,size_t,FILE *);	///
size_t	 fwrite(void *,size_t,size_t,FILE *);	///
//int	 printf(char *,...);	///
int	 fprintf(FILE *,char *,...);	///
int	 vfprintf(FILE *,char *,va_list);	///
int	 vprintf(char *,va_list);	///
int	 sprintf(char *,char *,...);	///
int	 vsprintf(char *,char *,va_list);	///
int	 scanf(char *,...);	///
int	 fscanf(FILE *,char *,...);	///
int	 sscanf(char *,char *,...);	///
void	 setbuf(FILE *,char *);	///
int	 setvbuf(FILE *,char *,int,size_t);	///
int	 remove(char *);	///
int	 rename(char *,char *);	///
void	 perror(char *);	///
int	 fgetpos(FILE *,fpos_t *);	///
int	 fsetpos(FILE *,fpos_t *);	///
FILE *	 tmpfile();	///
int	 _rmtmp();
int      _fillbuf(FILE *);
int      _flushbu(int, FILE *);

int  getw(FILE *FHdl);	///
int  putw(int Word, FILE *FilePtr);	///

///
int  getchar()		{ return getc(stdin);		}
///
int  putchar(int c)	{ return putc(c,stdout);	}
///
int  getc(FILE *fp)	{ return fgetc(fp);		}
///
int  putc(int c,FILE *fp) { return fputc(c,fp);		}

version (Win32)
{
    ///
    int  ferror(FILE *fp)	{ return fp._flag&_IOERR;	}
    ///
    int  feof(FILE *fp)	{ return fp._flag&_IOEOF;	}
    ///
    void clearerr(FILE *fp)	{ fp._flag &= ~(_IOERR|_IOEOF); }
    ///
    void rewind(FILE *fp)	{ fseek(fp,0L,SEEK_SET); fp._flag&=~_IOERR; }
    int  _bufsize(FILE *fp)	{ return fp._bufsiz; }
    ///
    int  fileno(FILE *fp)	{ return fp._file; }
    int  _snprintf(char *,size_t,char *,...);
    int  _vsnprintf(char *,size_t,char *,va_list);
}

version (linux)
{
    int  ferror(FILE *fp);
    int  feof(FILE *fp);
    void clearerr(FILE *fp);
    void rewind(FILE *fp);
    int  _bufsize(FILE *fp);
    int  fileno(FILE *fp);
    int  snprintf(char *,size_t,char *,...);
    int  vsnprintf(char *,size_t,char *,va_list);
}

int      unlink(char *);	///
FILE *	 fdopen(int, char *);	///
int	 fgetchar();	///
int	 fputchar(int);	///
int	 fcloseall();	///
int	 filesize(char *);	///
int	 flushall();	///
int	 getch();	///
int	 getche();	///
int      kbhit();	///
char *   tempnam (char *dir, char *pfx);	///

wchar_t *  _wtmpnam(wchar_t *);	///
FILE *  _wfopen(wchar_t *, wchar_t *);
FILE *  _wfsopen(wchar_t *, wchar_t *, int);
FILE *  _wfreopen(wchar_t *, wchar_t *, FILE *);
wchar_t *  fgetws(wchar_t *, int, FILE *);	///
int  fputws(wchar_t *, FILE *);	///
wchar_t *  _getws(wchar_t *);
int  _putws(wchar_t *);
int  wprintf(wchar_t *, ...);	///
int  fwprintf(FILE *, wchar_t *, ...);	///
int  vwprintf(wchar_t *, va_list);	///
int  vfwprintf(FILE *, wchar_t *, va_list);	///
int  swprintf(wchar_t *, wchar_t *, ...);	///
int  vswprintf(wchar_t *, wchar_t *, va_list);	///
int  _snwprintf(wchar_t *, size_t, wchar_t *, ...);
int  _vsnwprintf(wchar_t *, size_t, wchar_t *, va_list);
int  wscanf(wchar_t *, ...);	///
int  fwscanf(FILE *, wchar_t *, ...);	///
int  swscanf(wchar_t *, wchar_t *, ...);	///
int  _wremove(wchar_t *);
void  _wperror(wchar_t *);
FILE *  _wfdopen(int, wchar_t *);
wchar_t *  _wtempnam(wchar_t *, wchar_t *);
wchar_t  fgetwc(FILE *);	///
wchar_t  _fgetwchar_t();
wchar_t  fputwc(wchar_t, FILE *);	///
wchar_t  _fputwchar_t(wchar_t);
wchar_t  ungetwc(wchar_t, FILE *);	///

///
wchar_t	 getwchar_t()		{ return fgetwc(stdin); }
///
wchar_t	 putwchar_t(wchar_t c)	{ return fputwc(c,stdout); }
///
wchar_t	 getwc(FILE *fp)	{ return fgetwc(fp); }
///
wchar_t	 putwc(wchar_t c, FILE *fp)	{ return fputwc(c, fp); }

int fwide(FILE* fp, int mode);	///
