// RUN: %ldc -c %s

private import core.sys.windows.windows;
private import core.sys.windows.windef;
private import core.sys.windows.shellapi;
private import core.sys.windows.winuser;
private import core.sys.windows.uuid;
private import core.sys.windows.unknwn;
private import core.sys.windows.objbase;
private import core.sys.windows.objbase : CoInitialize, CoUninitialize;
private import core.sys.windows.objidl : IPersistFile;
private import core.sys.windows.shlobj;
private import core.sys.windows.wtypes : CLSCTX;

private import core.sys.windows.psapi;
private import core.sys.windows.winnt;
private import core.sys.windows.winbase;
private import core.sys.windows.winver;

void createShortcut(string exe_file_path, string dest_shortcut_file, string _description) {

	import std.utf : toUTF16z;
	LPCWSTR pathToObj = exe_file_path.toUTF16z;
	LPCWSTR pathToLink = dest_shortcut_file.toUTF16z;
	LPCWSTR description = _description.toUTF16z;
	HRESULT hRes;
	IShellLink psl;
	CoInitialize(NULL);
	hRes = CoCreateInstance(cast(GUID*)&CLSID_ShellLink, cast(IUnknown)NULL, CLSCTX.CLSCTX_INPROC_SERVER, cast(GUID*)&IID_IShellLinkW, cast(LPVOID*)&psl);
	if (SUCCEEDED(hRes))
	{
		IPersistFile ppf;
		psl.SetPath(pathToObj);
		psl.SetDescription(description);
		version(LDC){} else hRes = psl.QueryInterface(cast(IID*)&IID_IPersistFile, cast(LPVOID*) &ppf);
		if (SUCCEEDED(hRes))
		{
			ppf.Save(pathToLink, true);
			ppf.Release();
		}
		else {
			throw new Exception("Error creating shortcut " ~ hRes.to!string);
		}
		psl.Release();
	}
	else {
		throw new Exception("Error creating shortcut " ~ hRes.to!string);
	}
	CoUninitialize();
}
