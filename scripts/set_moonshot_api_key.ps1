$ErrorActionPreference = "Stop"

Write-Host "Kimi API key secure setup" -ForegroundColor Cyan
Write-Host "1. Revoke the key that was pasted into chat."
Write-Host "2. Create a new key in the Kimi console."
Write-Host "3. Paste the NEW key below. Input will stay hidden."
Write-Host ""

$secureKey = Read-Host "New MOONSHOT_API_KEY" -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    if ([string]::IsNullOrWhiteSpace($plainKey) -or -not $plainKey.StartsWith("sk-")) {
        throw "The value does not look like a Kimi API key. Nothing was saved."
    }
    [Environment]::SetEnvironmentVariable("MOONSHOT_API_KEY", $plainKey, "User")
    Write-Host ""
    Write-Host "Saved securely as a Windows user environment variable." -ForegroundColor Green
    Write-Host "The key was not written to this repository or printed on screen."
}
finally {
    if ($null -ne $pointer) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    $plainKey = $null
    $secureKey.Dispose()
}

Write-Host ""
Read-Host "Press Enter to close"
