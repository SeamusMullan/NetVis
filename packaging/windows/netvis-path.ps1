# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
    Add or remove one directory from the Windows PATH environment variable.

.DESCRIPTION
    The NSIS installer calls this during install and uninstall to put
    <install dir>\bin on PATH, which is what lets an MCP client resolve
    `netvis_mcp` by bare name.

    CPack's stock NSIS template does this itself, in NSIS, and cannot (#149):

      * It reads the whole PATH into an NSIS string. A standard makensis build
        caps those at 1024 characters, so on any machine with a longer PATH the
        read returns nothing and the installer gives up with
        "Warning! PATH too long installer unable to modify PATH!".
      * It gates the edit on `IfFileExists "<dir>\*.*"` and dedupes against
        `GetFullPathName /SHORT`. Both are unreliable on virtual drives (subst,
        mounted VHD, mapped network drives) and on volumes with 8.3 name
        generation disabled, where the edit is skipped with no message at all.

    Neither limit exists here: the registry API returns the value whatever its
    length, and nothing about this script cares which kind of volume the
    directory lives on.

.PARAMETER Action
    'add' or 'remove'.

.PARAMETER Directory
    The directory to add to or remove from PATH.

.PARAMETER Scope
    'machine' (HKLM, all users) or 'user' (HKCU, current user only).

.PARAMETER SelfTest
    Run the string-manipulation unit tests and exit. Touches no registry key.
    CI runs this on every push; see .github/workflows/ci.yml.

.OUTPUTS
    Exit code 0 on success (including "nothing to do"), 1 on failure. The
    installer treats a failure as non-fatal and tells the user to edit PATH by
    hand, so a locked-down machine gets a working NetVis and an accurate
    message rather than a failed install.
#>
[CmdletBinding(DefaultParameterSetName = 'Apply')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Apply')]
    [ValidateSet('add', 'remove')]
    [string]$Action,

    [Parameter(Mandatory = $true, ParameterSetName = 'Apply')]
    [ValidateNotNullOrEmpty()]
    [string]$Directory,

    [Parameter(ParameterSetName = 'Apply')]
    [ValidateSet('machine', 'user')]
    [string]$Scope = 'machine',

    [Parameter(Mandatory = $true, ParameterSetName = 'SelfTest')]
    [switch]$SelfTest
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# --- Pure string helpers -----------------------------------------------------
# Split out from the registry access so they can be tested without a registry,
# an installer, or an administrator. -SelfTest below is the whole test suite.

# Compare two PATH entries the way Windows resolves them: case-insensitively,
# ignoring a trailing separator. "C:\NetVis\bin", "c:\netvis\bin\" and
# "C:\NetVis\bin " are one entry, not three.
function Test-SamePathEntry {
    param([string]$A, [string]$B)
    $na = $A.Trim().TrimEnd('\', '/')
    $nb = $B.Trim().TrimEnd('\', '/')
    return [string]::Equals($na, $nb, [StringComparison]::OrdinalIgnoreCase)
}

# Split a raw PATH value into entries, dropping the empty fields that a
# trailing or doubled ';' produces. An empty field in PATH means "the current
# directory" to some resolvers, so preserving them is a security misfeature,
# not fidelity.
function Split-PathValue {
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value)) { return @() }
    return @($Value -split ';' | Where-Object { $_.Trim().Length -gt 0 })
}

# Append $Directory unless an equivalent entry is already present. Appends
# rather than prepends: PATH order is the user's, and an installer that jumps
# the queue can shadow a tool the user deliberately put first.
function Add-EntryToPathValue {
    param([string]$Value, [string]$Directory)
    $entries = @(Split-PathValue -Value $Value)
    foreach ($e in $entries) {
        if (Test-SamePathEntry -A $e -B $Directory) { return $Value }
    }
    return (@($entries) + $Directory) -join ';'
}

# Drop every equivalent entry, not just the first: a PATH that accumulated the
# directory twice (an older installer, a hand edit) must come out clean.
function Remove-EntryFromPathValue {
    param([string]$Value, [string]$Directory)
    $entries = @(Split-PathValue -Value $Value |
        Where-Object { -not (Test-SamePathEntry -A $_ -B $Directory) })
    return $entries -join ';'
}

# --- Registry access ---------------------------------------------------------

function Get-EnvironmentKey {
    param([string]$Scope)
    if ($Scope -eq 'machine') {
        $key = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey(
            'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', $true)
    }
    else {
        $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    }
    if ($null -eq $key) {
        throw "cannot open the $Scope environment registry key for writing"
    }
    return $key
}

function Invoke-PathUpdate {
    param([string]$Action, [string]$Directory, [string]$Scope)

    $key = Get-EnvironmentKey -Scope $Scope
    try {
        # DoNotExpandEnvironmentNames is the point of using the registry API
        # directly. Get-ItemProperty and [Environment]::GetEnvironmentVariable
        # both EXPAND a REG_EXPAND_SZ PATH, so writing the result back would
        # bake %SystemRoot% into a literal and quietly break PATH for anyone
        # who later moves or re-images the volume.
        $raw = $key.GetValue(
            'Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $raw = [string]$raw

        # Preserve whatever kind the value already has. A PATH holding %VARS%
        # must stay REG_EXPAND_SZ; rewriting it as REG_SZ leaves the literal
        # text in place and every entry behind a variable stops resolving.
        $kind = [Microsoft.Win32.RegistryValueKind]::ExpandString
        try {
            $existing = $key.GetValueKind('Path')
            if ($existing -eq [Microsoft.Win32.RegistryValueKind]::String) {
                $kind = [Microsoft.Win32.RegistryValueKind]::String
            }
        }
        catch [System.IO.IOException] {
            # No Path value yet - create it as REG_EXPAND_SZ, per the default above.
        }

        if ($Action -eq 'add') {
            $updated = Add-EntryToPathValue -Value $raw -Directory $Directory
        }
        else {
            $updated = Remove-EntryFromPathValue -Value $raw -Directory $Directory
        }

        if ($updated -eq $raw) {
            Write-Output "PATH ($Scope) already correct; left unchanged."
            return
        }

        $key.SetValue('Path', $updated, $kind)
        Write-Output "PATH ($Scope) updated: $Action '$Directory'."
    }
    finally {
        $key.Close()
    }
}

# --- Self test ---------------------------------------------------------------

# Sets $script:SelfTestFailed rather than returning a code: this function
# writes its transcript to the pipeline, so a `return 1` would come back as the
# last element of an array of strings, not as an exit status.
$script:SelfTestFailed = $false

function Invoke-SelfTest {
    $failures = New-Object System.Collections.ArrayList

    function Assert-Equal {
        param([string]$Name, [string]$Expected, [string]$Actual)
        if ($Expected -ceq $Actual) {
            Write-Output "  ok   $Name"
        }
        else {
            Write-Output "  FAIL $Name"
            Write-Output "       expected: '$Expected'"
            Write-Output "       actual:   '$Actual'"
            [void]$failures.Add($Name)
        }
    }

    Write-Output 'netvis-path.ps1 self test'

    Assert-Equal 'add to empty PATH' `
        'C:\NetVis\bin' (Add-EntryToPathValue -Value '' -Directory 'C:\NetVis\bin')

    Assert-Equal 'add appends, does not prepend' `
        'C:\Windows;C:\NetVis\bin' `
        (Add-EntryToPathValue -Value 'C:\Windows' -Directory 'C:\NetVis\bin')

    Assert-Equal 'add is idempotent' `
        'C:\Windows;C:\NetVis\bin' `
        (Add-EntryToPathValue -Value 'C:\Windows;C:\NetVis\bin' -Directory 'C:\NetVis\bin')

    Assert-Equal 'add ignores case and a trailing slash' `
        'C:\Windows;c:\netvis\bin\' `
        (Add-EntryToPathValue -Value 'C:\Windows;c:\netvis\bin\' -Directory 'C:\NetVis\bin')

    Assert-Equal 'add drops empty fields' `
        'C:\Windows;C:\NetVis\bin' `
        (Add-EntryToPathValue -Value 'C:\Windows;;' -Directory 'C:\NetVis\bin')

    Assert-Equal 'add preserves unexpanded variables' `
        '%SystemRoot%\system32;C:\NetVis\bin' `
        (Add-EntryToPathValue -Value '%SystemRoot%\system32' -Directory 'C:\NetVis\bin')

    Assert-Equal 'add works on a virtual drive letter' `
        'C:\Windows;X:\NetVis\bin' `
        (Add-EntryToPathValue -Value 'C:\Windows' -Directory 'X:\NetVis\bin')

    Assert-Equal 'remove takes the entry out' `
        'C:\Windows;C:\Other' `
        (Remove-EntryFromPathValue -Value 'C:\Windows;C:\NetVis\bin;C:\Other' -Directory 'C:\NetVis\bin')

    Assert-Equal 'remove takes out every duplicate' `
        'C:\Windows' `
        (Remove-EntryFromPathValue -Value 'C:\NetVis\bin;C:\Windows;c:\netvis\bin\' -Directory 'C:\NetVis\bin')

    Assert-Equal 'remove of an absent entry is a no-op' `
        'C:\Windows' `
        (Remove-EntryFromPathValue -Value 'C:\Windows' -Directory 'C:\NetVis\bin')

    Assert-Equal 'remove leaves no trailing separator' `
        'C:\Windows' `
        (Remove-EntryFromPathValue -Value 'C:\Windows;C:\NetVis\bin' -Directory 'C:\NetVis\bin')

    Assert-Equal 'remove of the only entry yields empty' `
        '' (Remove-EntryFromPathValue -Value 'C:\NetVis\bin' -Directory 'C:\NetVis\bin')

    # The regression this whole file exists for: a PATH far longer than the
    # 1024-character NSIS string that made the stock installer give up.
    $long = (1..200 | ForEach-Object { "C:\some\padding\directory\number$_" }) -join ';'
    if ($long.Length -le 1024) { throw 'self test bug: the long PATH is not long' }
    $grown = Add-EntryToPathValue -Value $long -Directory 'C:\NetVis\bin'
    Assert-Equal 'add survives a PATH over 1024 characters' `
        "$long;C:\NetVis\bin" $grown
    Assert-Equal 'remove survives a PATH over 1024 characters' `
        $long (Remove-EntryFromPathValue -Value $grown -Directory 'C:\NetVis\bin')

    if ($failures.Count -gt 0) {
        Write-Output "FAILED: $($failures.Count) assertion(s): $($failures -join ', ')"
        $script:SelfTestFailed = $true
        return
    }
    Write-Output 'all assertions passed'
}

# --- Entry point -------------------------------------------------------------

try {
    if ($PSCmdlet.ParameterSetName -eq 'SelfTest') {
        Invoke-SelfTest
        if ($script:SelfTestFailed) { exit 1 }
        exit 0
    }
    Invoke-PathUpdate -Action $Action -Directory $Directory.Trim() -Scope $Scope
    exit 0
}
catch {
    Write-Output "netvis-path.ps1: $($_.Exception.Message)"
    exit 1
}
