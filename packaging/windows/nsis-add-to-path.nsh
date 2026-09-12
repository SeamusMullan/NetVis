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
; long installer unable to modify PATH!" past that. It also gates the edit on
; `IfFileExists "<dir>\*.*"` and dedupes with `GetFullPathName /SHORT`, neither of
; which is reliable on a virtual drive (subst, mounted VHD, mapped network drive)
; or on a volume with 8.3 name generation switched off -- there it skips the edit
; and says nothing at all. netvis-path.ps1 goes at the registry directly and has
; neither limit.
;
; CPACK_NSIS_MODIFY_PATH stays ON so the options page, the three PATH radio
; buttons and the uninstall registry entries all keep working; only the edit
; itself moves here.

  Push $R0
  Push $R1

  ; The user picked "Do not add NetVis to the system PATH" on the options page.
  StrCmp $DO_NOT_ADD_TO_PATH "1" netvis_path_done 0

  ; The page offers all-users (HKLM) or current-user (HKCU); mirror that choice.
  StrCpy $R0 "user"
  StrCmp $ADD_TO_PATH_ALL_USERS "1" 0 +2
    StrCpy $R0 "machine"

  DetailPrint "NetVis: adding $INSTDIR\bin to the $R0 PATH"
  nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\share\netvis\netvis-path.ps1" -Action add -Directory "$INSTDIR\bin" -Scope $R0'
  Pop $R1
  StrCmp $R1 "0" netvis_path_ok 0

  ; Non-fatal by design. A machine that refuses the registry write (locked-down
  ; policy, no elevation) still gets a complete, working NetVis; what it does not
  ; get is `netvis_mcp` on PATH, so say exactly that instead of failing the
  ; install or leaving the user to guess.
  DetailPrint "NetVis: PATH was NOT modified (netvis-path.ps1 exit code: $R1)"
  MessageBox MB_OK|MB_ICONINFORMATION "NetVis is installed, but PATH could not be updated automatically.$\n$\nTo run netvis_mcp by name, add this folder to your PATH:$\n$\n$INSTDIR\bin" /SD IDOK
  Goto netvis_path_done

  netvis_path_ok:
  ; Tell running shells the environment changed, so a newly-opened terminal
  ; picks up the new PATH without a sign-out.
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  netvis_path_done:
  ; Suppress the template's own "-Add to path" section, which runs immediately
  ; after this one and calls the broken AddToPath this file replaces. The user's
  ; real choice was already written to the uninstall registry key further up the
  ; section, so overwriting the variable here does not affect the uninstaller.
  StrCpy $DO_NOT_ADD_TO_PATH "1"

  Pop $R1
  Pop $R0
