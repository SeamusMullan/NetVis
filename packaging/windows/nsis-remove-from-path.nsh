; SPDX-License-Identifier: Apache-2.0
;
; The uninstall half of packaging/windows/nsis-add-to-path.nsh (#149). CPack drops
; this into the Uninstall Section via CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS, which
; the template places BEFORE the file-deletion commands -- so netvis-path.ps1 is
; still on disk when this runs.
;
; $DO_NOT_ADD_TO_PATH and $ADD_TO_PATH_ALL_USERS are read back from the uninstall
; registry key immediately above this point, so this sees the choice the user
; actually made at install time.
;
; The template's own un.RemoveFromPath still runs at the end of the section. That
; is harmless: by then the entry is gone, so its search finds nothing and it
; leaves PATH alone. (Its own long-PATH read returns an empty string rather than
; a truncated one, so it cannot damage a long PATH either.)

  Push $R0
  Push $R1

  ; Nothing was added, so there is nothing to take away.
  StrCmp $DO_NOT_ADD_TO_PATH "1" netvis_unpath_done 0

  ; Installs made before this mechanism existed have no script to call; their
  ; PATH entry is left to the template's un.RemoveFromPath as before.
  IfFileExists "$INSTDIR\share\netvis\netvis-path.ps1" 0 netvis_unpath_done

  StrCpy $R0 "user"
  StrCmp $ADD_TO_PATH_ALL_USERS "1" 0 +2
    StrCpy $R0 "machine"

  DetailPrint "NetVis: removing $INSTDIR\bin from the $R0 PATH"
  nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\share\netvis\netvis-path.ps1" -Action remove -Directory "$INSTDIR\bin" -Scope $R0'
  Pop $R1
  StrCmp $R1 "0" 0 netvis_unpath_done
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  netvis_unpath_done:
  Pop $R1
  Pop $R0
