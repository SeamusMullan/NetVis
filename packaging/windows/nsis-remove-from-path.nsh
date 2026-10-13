; SPDX-License-Identifier: Apache-2.0
;
; The uninstall half of packaging/windows/nsis-add-to-path.nsh (#149). CPack drops
; this into the Uninstall Section via CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS, which
; the template places BEFORE the file-deletion commands -- so netvis-path.ps1 is
; still on disk when this runs.
;
; $DO_NOT_ADD_TO_PATH and $ADD_TO_PATH_ALL_USERS are read back from the uninstall
; registry key immediately above this point, so this sees the choice the user
; actually made at install time (nsis-add-to-path.nsh writes that choice itself,
; so it is there even when the template's own write was skipped).
;
; The template's own un.RemoveFromPath still runs at the end of the section. That
; is harmless when this snippet succeeded: by then the entry is gone, so its search
; finds nothing and it leaves PATH alone. When this snippet fails it is the
; fallback -- but its long-PATH read returns an empty string rather than the
; value, so it finds nothing and leaves a PATH of 1024+ characters alone too. It
; cannot damage a long PATH either way.

  Push $R0
  Push $R1

  ; Nothing was added, so there is nothing to take away.
  StrCmp $DO_NOT_ADD_TO_PATH "1" netvis_unpath_done 0

  ; The script is installed with the uninstaller that calls it, so this only
  ; fails if someone has deleted the file from the install directory.
  IfFileExists "$INSTDIR\share\netvis\netvis-path.ps1" netvis_unpath_run 0
  DetailPrint "NetVis: $INSTDIR\share\netvis\netvis-path.ps1 is missing; PATH was NOT modified (remove $INSTDIR\bin from PATH by hand if it is listed)"
  Goto netvis_unpath_done

  netvis_unpath_run:
  StrCpy $R0 "user"
  StrCmp $ADD_TO_PATH_ALL_USERS "1" 0 +2
    StrCpy $R0 "machine"

  DetailPrint "NetVis: removing $INSTDIR\bin from the $R0 PATH"
  nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\share\netvis\netvis-path.ps1" -Action remove -Directory "$INSTDIR\bin" -Scope $R0'
  Pop $R1
  StrCmp $R1 "0" netvis_unpath_ok 0

  ; Say so. A stale entry pointing at a deleted folder is harmless to run but
  ; untidy, and without this line the log gives no hint it is still there when
  ; powershell.exe could not be launched at all (no output to capture).
  DetailPrint "NetVis: PATH was NOT modified (netvis-path.ps1 exit code: $R1); the installer's built-in removal will try next, but it cannot read a PATH over 1024 characters, so remove $INSTDIR\bin from PATH by hand if it is still listed"
  Goto netvis_unpath_done

  netvis_unpath_ok:
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  netvis_unpath_done:
  Pop $R1
  Pop $R0
