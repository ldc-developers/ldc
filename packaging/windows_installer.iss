; Build it with a cmdline like this:
; "C:\Program Files (x86)\Inno Setup 6\iscc" "/OC:\output" /DLDCVersion=1.23.4 "/DLDCDir=C:\LDC\ldc2-1.23.4-windows-multilib" windows_installer.iss

;#define LDCVersion "1.24.0"
;#define LDCDir "C:\LDC\ldc2-1.24.0-windows-multilib"

; Strip revision from LDCVersion and use as app ID.
; => LDC 1.24.1 will upgrade LDC 1.24.0, but LDC 1.25.0-beta1 is a separate 1.25 family
#define LDCAppId RemoveFileExt(LDCVersion)

[Setup]
AppId=LDC_developers_LDC_{#LDCAppId}
AppName=LDC
AppVersion={#LDCVersion}
AppVerName=LDC {#LDCVersion}
ArchitecturesAllowed=x64
; Enable /CURRENTUSER cmdline option to install for current user only, requiring no admin privileges.
; This affects the default install dir (override with /DIR="x:\dirname") and the registry root key (HKCU, not HKLM).
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
DisableProgramGroupPage=yes
DisableReadyPage=yes
DefaultDirName={autopf64}\LDC {#LDCAppId}
OutputBaseFilename=ldc2-{#LDCVersion}-windows-multilib
Compression=lzma2/ultra64
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#LDCDir}\*"; Excludes: "\lib32,\lib64"; DestDir: "{app}"; Components: core; Flags: ignoreversion recursesubdirs
Source: "{#LDCDir}\lib64\*"; DestDir: "{app}\lib64"; Components: lib64; Flags: ignoreversion recursesubdirs
Source: "{#LDCDir}\lib32\*"; DestDir: "{app}\lib32"; Components: lib32; Flags: ignoreversion recursesubdirs

[Components]
Name: core; Description: "Executables, config file and imports"; Types: full compact custom; Flags: fixed
Name: lib64; Description: "x64 libraries"; Types: full compact
Name: lib32; Description: "x86 libraries"; Types: full

[Run]
; note: not added to PATH for silent installs with /SILENT or /VERYSILENT
Filename: "{cmd}"; Parameters: "/c echo hello"; Check: not IsInUserEnvPath; BeforeInstall: AddToUserEnvPath; Description: "Add to PATH environment variable for current user"; Flags: postinstall skipifsilent runhidden nowait
Filename: "{cmd}"; Parameters: "/c echo hello"; Check: IsAdminInstallMode and not IsInSystemEnvPath; BeforeInstall: AddToSystemEnvPath; Description: "Add to system PATH environment variable"; Flags: postinstall skipifsilent runhidden nowait unchecked
Filename: "{app}\README.txt"; Description: "View the README file"; Flags: postinstall shellexec skipifdoesntexist skipifsilent unchecked

[Registry]
; note: 32-bit registry view of HKLM\SOFTWARE (default admin install) or HKCU\SOFTWARE (/CURRENTUSER)
Root: HKA; Subkey: "SOFTWARE\LDC"; Flags: uninsdeletekeyifempty
Root: HKA; Subkey: "SOFTWARE\LDC\{#LDCAppId}"; Flags: uninsdeletekey
Root: HKA; Subkey: "SOFTWARE\LDC\{#LDCAppId}"; ValueType: string; ValueName: "InstallationFolder"; ValueData: "{app}"
Root: HKA; Subkey: "SOFTWARE\LDC\{#LDCAppId}"; ValueType: string; ValueName: "Version"; ValueData: "{#LDCVersion}"

[Code]
function GetTargetBinDir(): string;
begin
    result := ExpandConstant('{app}') + '\bin';
end;

const WM_SETTINGCHANGE = 26;

{ make the shell etc. reload environment variables }
procedure RefreshEnvironment();
var
    Dummy: string;
begin
    Dummy := 'Environment';
    SendBroadcastNotifyMessage(WM_SETTINGCHANGE, 0, CastStringToInteger(Dummy));
end;


{ add the target bin dir to PATH if not already present }
function AddToEnvPath(const RootKey: Integer; const SubKeyName: string): Boolean;
var
    Path: string;
    Dir: string;
begin
    result := False;

    if not RegQueryStringValue(RootKey, SubKeyName, 'Path', Path) then Path := '';

    Dir := GetTargetBinDir();

    { skip if already present }
    if Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Path) + ';') > 0 then exit;

    { prepend `<Dir>;` }
    Path := Dir + ';' + Path;

    result := RegWriteStringValue(RootKey, SubKeyName, 'Path', Path);
end;

{ add the target bin dir to user PATH if not already present }
procedure AddToUserEnvPath();
begin
    if AddToEnvPath(HKCU, 'Environment') then
    begin
        Log(Format('Added to user PATH: %s', [GetTargetBinDir()]));
        RefreshEnvironment();
    end;
end;

{ add the target bin dir to system PATH if not already present }
procedure AddToSystemEnvPath();
begin
    if AddToEnvPath(HKLM, 'System\CurrentControlSet\Control\Session Manager\Environment') then
    begin
        Log(Format('Added to system PATH: %s', [GetTargetBinDir()]));
        RefreshEnvironment();
    end;
end;


{ remove the target bin dir from PATH if present }
function RemoveFromEnvPath(const RootKey: Integer; const SubKeyName: string): Boolean;
var
    Path: string;
    Dir: string;
    P: Integer;
begin
    result := False;

    if not RegQueryStringValue(RootKey, SubKeyName, 'Path', Path) then exit;

    Dir := GetTargetBinDir();

    P := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Path));
    if P = 0 then exit;

    { remove `<Dir>;` from Path }
    Delete(Path, P, Length(Dir) + 1);

    result := RegWriteStringValue(RootKey, SubKeyName, 'Path', Path);
end;

{ remove the target bin dir from user PATH if present }
procedure RemoveFromUserEnvPath();
begin
    if RemoveFromEnvPath(HKCU, 'Environment') then
    begin
        Log(Format('Removed from user PATH: %s', [GetTargetBinDir()]));
        RefreshEnvironment();
    end;
end;

{ remove the target bin dir from system PATH if present }
procedure RemoveFromSystemEnvPath();
begin
    if RemoveFromEnvPath(HKLM, 'System\CurrentControlSet\Control\Session Manager\Environment') then
    begin
        Log(Format('Removed from system PATH: %s', [GetTargetBinDir()]));
        RefreshEnvironment();
    end;
end;


{ check if the target bin dir is already in PATH }
function IsInEnvPath(const RootKey: Integer; const SubKeyName: string): Boolean;
var
    Path: string;
begin
    result := False;
    if RegQueryStringValue(RootKey, SubKeyName, 'Path', Path) then
        result := Pos(';' + Uppercase(GetTargetBinDir()) + ';', ';' + Uppercase(Path) + ';') > 0;
end;

{ check if the target bin dir is already in user PATH }
function IsInUserEnvPath(): Boolean;
begin
    result := IsInEnvPath(HKCU, 'Environment');
end;

{ check if the target bin dir is already in system PATH }
function IsInSystemEnvPath(): Boolean;
begin
    result := IsInEnvPath(HKLM, 'System\CurrentControlSet\Control\Session Manager\Environment');
end;


{ adapt 'Next' button label because of hidden ready page }
procedure CurPageChanged(CurPageID: Integer);
begin
    if CurPageID = wpSelectComponents then
        WizardForm.NextButton.Caption := SetupMessage(msgButtonInstall);
end;

{ remove bin dir from user/system PATH at post-uninstall }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
    if (CurUninstallStep = usPostUninstall) then
    begin
        RemoveFromUserEnvPath();
        if IsAdminInstallMode then
            RemoveFromSystemEnvPath();
    end;
end;
