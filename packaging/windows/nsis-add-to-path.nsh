; SPDX-License-Identifier: Apache-2.0
;
; PATH handling for the NetVis Windows installer (#149).
;
; CPack drops this file into the main install Section through
; CPACK_NSIS_EXTRA_INSTALL_COMMANDS (see the NSIS block in CMakeLists.txt). It is
; !include'd rather than inlined as a CMake string on purpose: CPack round-trips
; those strings through a generated CPackConfig.cmake, so every backslash, double
; quote and ${...} in them has to survive two layers of CMake escaping. A real
; .nsh file has no such problem and can be read as the NSIS it is.
;
; Why NetVis does its own PATH edit instead of letting the CPack template do it:
; the template's AddToPath reads PATH into an NSIS string, which a standard
; makensis build caps at 1024 characters, and bails out with "Warning! PATH too
; long installer unable to modify PATH!" past that. netvis-path.ps1 goes at the
; registry directly and has no such limit.
;
; The template's AddToPath also has two checks that make it skip the edit
; without a word: `IfFileExists "<dir>\*.*"` (the directory must be visible to
; the installer process) and a `GetFullPathName /SHORT` dedupe (when
; GetShortPathName fails outright the template ends up searching PATH for just
; ";", which matches anything). Either can misfire when the install directory is
; on a subst or mapped drive -- a drive letter belongs to a logon session, and
; an elevated installer can run in a different one from the prompt that mapped
; the drive. That is a suspected cause of the PATH side of the "virtual drive"
; report in #149, not a confirmed one, and it says nothing about an install that
; fails outright on such a drive, which needs its own repro. This snippet does not
; detect or warn about such a drive: the entry it writes is only as durable as
; the drive mapping is.
;
; CPACK_NSIS_MODIFY_PATH stays ON so the options page and the three PATH radio
; buttons keep working; only the edit itself moves here.
;
; Requires, from the CPack template: $DO_NOT_ADD_TO_PATH and $ADD_TO_PATH_ALL_USERS
; (declared there, ASSIGNED here), the options-page INI that .onInit extracts into
; $PLUGINSDIR, and ConditionalAddToRegistry. See nsh-syntax-check.nsi.

  Push $R0
  Push $R1

  ; Read the user's choice from the options page directly rather than trusting
  ; the template's copy of it. The template assigns these variables, and writes
  ; DoNotAddToPath / AddToPathAllUsers to the uninstall key, only inside its
  ; MUI_STARTMENU_WRITE_BEGIN block -- which MUI skips when the user ticks "Do not
  ; create shortcuts". Without this, ticking that box would leave both variables
  ; empty, and this snippet would edit the current user's PATH even though the
  ; default "Do not add NetVis to the system PATH" was still selected. The INI is
  ; extracted in .onInit, so this also holds for a silent (/S) install, where the
  ; page never shows and the default ("do not add") applies.
  ReadINIStr $DO_NOT_ADD_TO_PATH "$PLUGINSDIR\NSIS.InstallOptions.ini" "Field 2" "State"
  ReadINIStr $ADD_TO_PATH_ALL_USERS "$PLUGINSDIR\NSIS.InstallOptions.ini" "Field 3" "State"

  ; Record the choice where the uninstaller reads it back, through the template's
  ; own helper so the uninstall key name is the template's business, not ours.
  ; This must happen before $DO_NOT_ADD_TO_PATH is overwritten at the bottom.
  ; (The helper writes nothing for an empty value.)
  Push "DoNotAddToPath"
  Push "$DO_NOT_ADD_TO_PATH"
  Call ConditionalAddToRegistry
  Push "AddToPathAllUsers"
  Push "$ADD_TO_PATH_ALL_USERS"
  Call ConditionalAddToRegistry

  ; Edit PATH only when the page explicitly says "0" for "Do not add". A missing
  ; or unreadable choice is treated as "do not add": touching the environment on
  ; a guess is worse than asking the user to do it.
  StrCmp $DO_NOT_ADD_TO_PATH "0" 0 netvis_path_suppress

  ; The page offers all-users (HKLM) or current-user (HKCU); mirror that choice.
  ; "Current user" is the account the installer runs as. The installer is elevated
  ; (RequestExecutionLevel admin), so when a standard user types an administrator's
  ; credentials at the UAC prompt, that is the administrator's HKCU, not the
  ; hive of the person who started the installer.
  StrCpy $R0 "user"
  StrCmp $ADD_TO_PATH_ALL_USERS "1" 0 +2
    StrCpy $R0 "machine"

  DetailPrint "NetVis: adding $INSTDIR\bin to the $R0 PATH"
  nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\share\netvis\netvis-path.ps1" -Action add -Directory "$INSTDIR\bin" -Scope $R0'
  Pop $R1
  StrCmp $R1 "0" netvis_path_ok 0

  ; Non-fatal by design: the install itself is complete and working. What may be
  ; missing is `netvis_mcp` on PATH, so say exactly that.
  ;
  ; $DO_NOT_ADD_TO_PATH is deliberately left alone on this branch, so the template's
  ; own "-Add to path" section still runs next and gets a turn. Its AddToPath
  ; works when PATH fits in an NSIS string, and gives up WITHOUT touching PATH when
  ; it does not, so it can only help. That matters where powershell.exe is blocked
  ; or refuses an unsigned script (an AllSigned policy set by Group Policy
  ; overrides -ExecutionPolicy Bypass; Constrained Language Mode blocks the
  ; registry calls): those machines kept a working PATH edit before this file
  ; existed, and should not lose it.
  DetailPrint "NetVis: netvis-path.ps1 did not update PATH (exit code: $R1); falling back to the installer's built-in PATH edit"
  MessageBox MB_OK|MB_ICONINFORMATION "NetVis is installed, but its PATH helper could not update PATH (exit code: $R1).$\n$\nThe installer will now try its built-in PATH edit, which cannot handle a PATH longer than 1024 characters. If netvis_mcp does not run by name afterwards, add this folder to your PATH by hand:$\n$\n$INSTDIR\bin" /SD IDOK
  Goto netvis_path_end

  netvis_path_ok:
  ; Tell running shells the environment changed, so a newly-opened terminal
  ; picks up the new PATH without a sign-out.
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  netvis_path_suppress:
  ; Either this snippet has handled PATH, or the user chose not to have it
  ; touched. Suppress the template's own "-Add to path" section, which runs
  ; immediately after this one: it would otherwise call AddToPath, which reads
  ; this process's stale environment and so would add a second, duplicate entry.
  ; The real choice was already written to the uninstall key above, so overwriting
  ; the variable here does not affect the uninstaller.
  StrCpy $DO_NOT_ADD_TO_PATH "1"

  netvis_path_end:
  Pop $R1
  Pop $R0
