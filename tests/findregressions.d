// Based on DSTRESS code by Thomas Kühne

module findregressions;

private import std.string;
private import std.conv;
private import std.stdio;
private import std.stream;
private import std.file;
private import std.c.stdlib;
private import std.date;


enum Result{
	UNTESTED	= 0,
	PASS		= 1 << 2,
	XFAIL		= 2 << 2,
	XPASS		= 3 << 2,
	FAIL		= 4 << 2,
	ERROR		= 5 << 2,
	BASE_MASK	= 7 << 2,

	EXT_MASK	= 3,
	BAD_MSG		= 1,
	BAD_GDB		= 2,
	
	MAX		= BAD_GDB + BASE_MASK
}

char[] toString(Result r){
	switch(r & Result.BASE_MASK){
		case Result.PASS: return "PASS";
		case Result.XPASS: return "XPASS";
		case Result.FAIL: return "FAIL";
		case Result.XFAIL: return "XFAIL";
		case Result.ERROR: return "ERROR";
		case Result.UNTESTED: return "UNTESTED";
		default:
			break;
	}
	throw new Exception(format("unhandled Result value %s", cast(int)r));
}

char[] dateString(){
	static char[] date;
	if(date is null){
		auto time = getUTCtime();
		auto year = YearFromTime(time);
		auto month = MonthFromTime(time);
		auto day = DateFromTime(time);
		date = format("%d-%02d-%02d", year, month+1, day); 
	}
	return date;
}

char[][] unique(char[][] a){
	char[][] b = a.sort;
	char[][] back;

	back ~= b[0];

	size_t ii=0;
	for(size_t i=0; i<b.length; i++){
		if(back[ii]!=b[i]){
			back~=b[i];
			ii++;
		}
	}

	return back;	
}

private{
	version(Windows){
		import std.c.windows.windows;
		extern(Windows) BOOL GetFileTime(HANDLE hFile, LPFILETIME lpCreationTime, LPFILETIME lpLastAccessTime, LPFILETIME lpLastWriteTime);
	}else version(linux){
		import std.c.linux.linux;
		version = Posix;
	}else version(Posix){
		import std.c.posix.posix;
	}else{
		static assert(0);
	}

	alias ulong FStime;

	FStime getFStime(char[] fileName){
		version(Windows){
			HANDLE h;
		
			if (useWfuncs){
				wchar* namez = std.utf.toUTF16z(fileName);
				h = CreateFileW(namez,GENERIC_WRITE,0,null,OPEN_ALWAYS,
					FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,cast(HANDLE)null);
			}else{
				char* namez = toMBSz(fileName);
				h = CreateFileA(namez,GENERIC_WRITE,0,null,OPEN_ALWAYS,
				FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,cast(HANDLE)null);
			}

			if (h == INVALID_HANDLE_VALUE)
				goto err;

			FILETIME creationTime;
			FILETIME accessTime;
			FILETIME writeTime;
		
			BOOL b = GetFileTime(h, &creationTime, &accessTime, &writeTime);
			if(b==1){
				long modA = writeTime.dwLowDateTime;
				long modB = writeTime.dwHighDateTime;
				return modA  | (modB << (writeTime.dwHighDateTime.sizeof*8));
			}

err:
			CloseHandle(h);
			throw new Exception("failed to query file modification : "~fileName);
		}else version(Posix){
			char* namez = toStringz(fileName);
			struct_stat statbuf;
		
			if(stat(namez, &statbuf)){
				throw new FileException(fileName, getErrno());
			}

			version(linux){
				return statbuf.st_mtime;
			}else version(OSX){
				return statbuf.st_mtimespec.tv_sec;
			}
		}else{
			static assert(0);
		}
	}
}

char[] cleanFileName(char[] file){
	char[] back;
	bool hadSep;

	foreach(char c; file){
		if(c == '/' || c == '\\'){
			if(!hadSep){
				back ~= '/';
				hadSep = true;
			}
		}else{
			back ~= c;
			hadSep = false;
		}
	}

	size_t start = 0;
	while(back[start] <= ' ' && start < back.length){
		start++;
	}

	size_t end = back.length-1;
	while(back[end] <= ' ' && end >= start){
		end--;
	}

	back = back[start .. end+1];

	return back;
}

class Test{
	char[] name;
	char[] file;
	Result r;

	this(char[] file){
		this.file = file;

		int start = rfind(file, "/");
		if(start<0){
			start = 0;
		}else{
			start += 1;
		}
		
		int end = rfind(file, ".");
		if(end < start){
			end = file.length;
		}

		name = file[start .. end];
	}
}


class Log{
	Test[char[]] tests;

	char[] id;

	this(char[] id){
		this.id = id;
	}


	void dropBogusResults(FStime recordTime, char[] testRoot){
		uint totalCount = tests.length; 
		
		char[][] sourcesTests = tests.keys;
		foreach(char[] source; sourcesTests){
			if(find(source, "complex/") < 0){
				try{
					FStime caseTime = getFStime(testRoot~std.path.sep~source);
					if(caseTime > recordTime){
						debug(drop) fwritefln(stderr, "dropped: %s", source);
						tests.remove(source);
					}
				}catch(Exception e){
					debug(drop) fwritefln(stderr, "dropped: %s", source);
					tests.remove(source);
				}
			}
			// asm-filter
			int i = find(source, "asm_p");
			if(i >= 0){
				tests.remove(source);
			}
		}
		tests.rehash;
		
		writefln("dropped %s outdated tests (%s remaining)", totalCount - tests.length, tests.length); 
	}

	
	bool add(char[] line){
		const char[] SUB = "Torture-Sub-";
		const char[] TORTURE = "Torture:";

		line = strip(line);
		int id = -1;
		Result r = Result.UNTESTED;

		if(line.length > SUB.length && line[0 .. SUB.length] == SUB){
			line = line[SUB.length .. $];
			id = 0;
			while(line[id] >= '0' && line[id] <= '9'){
				id++;
			}
			int start = id;
			id = std.conv.toUint(line[0 .. id]);

			while(line[start] != '-'){
				start++;
			}
			line = line[start+1 .. $];
		}

		char[][] token = split(line);
		if(token.length < 2){
			return false;
		}
		char[] file = strip(token[1]);

		switch(token[0]){
			case "PASS:":
				r = Result.PASS; break;
			case "FAIL:":
				r = Result.FAIL; break;
			case "XPASS:":
				r = Result.XPASS; break;
			case "XFAIL:":
				r = Result.XFAIL; break;
			case "ERROR:":
				r = Result.ERROR; break;
			default:{
				if(token[0] == TORTURE){
					throw new Exception("not yet handled: "~line);
				}else if(id > -1){
					throw new Exception(format("bug in SUB line: (%s) %s", id, line));
				}
			}
		}

		if(r != Result.UNTESTED){
			if(std.string.find(line, "bad error message") > -1){
				r |= Result.BAD_MSG;	
			}
			if(std.string.find(line, "bad debugger message") > -1){
				r |= Result.BAD_MSG;	
			}
			
			file = cleanFileName(file);
			
			if(id >= 0){
				// update sub
				id--;
		
				Test* test = file in tests;

				if(test is null){
					Test t = new Test(file);
					tests[file] = t;
					t.r = r;
				}else{
					if(test.r != Result.UNTESTED){
						test.r = Result.UNTESTED;
					}
					test.r = r;
				}
			}
			return true;
		}
		return false;
	}
}


int main(char[][] args){

	if(args.length < 2){
		fwritefln(stderr, "%s <old log> <new log>", args[0]);
		return 1;
	}
	
	
	Log[] logs;

	foreach(size_t id, char[] file; args[1 .. $]){
		writefln("parsing: %s", file);
		FStime logTime  = getFStime(file);
		debug fwritefln(stderr, "sourceTime: %s", logTime);

		Log l= new Log(file);
		Stream source = new BufferedFile(file, FileMode.In);
		while(!source.eof()){
			l.add(source.readLine());
		}
		
		l.dropBogusResults(logTime, "dstress");

		logs ~= l;
	}

	Log oldLog = logs[0];
	Log newLog = logs[1];

	foreach(Test t; newLog.tests.values){
		Test* oldT = t.file in oldLog.tests;

		if(oldT !is null){
			if(oldT.r == t.r)
				continue;
			else if(t.r >= Result.XPASS && oldT.r && oldT.r <= Result.XFAIL){
				writef("Regression   ");
			}
			else if(t.r && t.r <= Result.XFAIL && oldT.r >= Result.XPASS){
				writef("Improvement  ");
			}
			else {
				writef("Change       ");
			}
			writefln(toString(oldT.r), " -> ", toString(t.r), " : ", t.name, " in ", t.file);
		}
	}
	
	
	return 0;
}

