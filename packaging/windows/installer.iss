; MR MAKER for ui.clover - Windows 설치 프로그램 (Inno Setup 6)
; 빌드: iscc /DAppVersion=1.0.0 /DSourceDir=..\..\dist\MRMaker /DVariant=cpu /DOutputDir=..\..\dist\release installer.iss

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\MRMaker"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist\release"
#endif
#ifndef Variant
  #define Variant "cpu"
#endif

#define AppName "MR MAKER for ui.clover"
#define AppExe "MRMaker.exe"
#define KoreanIsl AddBackslash(CompilerPath) + "Languages\Korean.isl"

[Setup]
AppId={{6B0E4C1A-3B7D-4F8E-9C2A-7D51E0A9F3B4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=ui.clover
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; 관리자 권한 없이 사용자 계정에 설치 (회사 PC 에서도 설치 가능). 설치 시 "모든 사용자"도 선택 가능
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=MRMaker-{#AppVersion}-windows-x64-{#Variant}-setup
SetupIconFile=..\..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
#if Variant == "cuda"
; CUDA 빌드는 용량이 커서 설치 파일을 여러 조각(.bin)으로 나눔 — 모두 같은 폴더에 두고 실행
DiskSpanning=yes
DiskSliceSize=max
#endif

[Languages]
#if FileExists(KoreanIsl)
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
#endif
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; 이전 이름(Music Separator)으로 설치된 바로가기 정리 (같은 AppId 라 업그레이드로 설치됨)
Type: files; Name: "{group}\Music Separator.lnk"
Type: files; Name: "{autodesktop}\Music Separator.lnk"
Type: files; Name: "{app}\MusicSeparator.exe"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
; 프로그램 내 [업데이트] 로 조용히 설치한 경우(/relaunch=1) 설치 후 다시 실행
Filename: "{app}\{#AppExe}"; Flags: nowait runasoriginaluser; Check: ShouldRelaunch

[UninstallDelete]
; 임시 캐시는 삭제. 다운로드한 AI 모델(%LOCALAPPDATA%\MusicSeparator\models)은 재설치를 위해 유지
Type: filesandordirs; Name: "{localappdata}\MusicSeparator\cache"

[Code]
function ShouldRelaunch: Boolean;
begin
  Result := ExpandConstant('{param:relaunch|0}') = '1';
end;
