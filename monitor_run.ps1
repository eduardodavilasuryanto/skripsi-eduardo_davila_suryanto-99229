<#
.SYNOPSIS
    Pemantau berkala satu run train_asr.py yang berjalan di luar sesi interaktif.

.DESCRIPTION
    Menempel pada proses train_asr.py yang sedang berjalan, dikenali dari
    command line yang memuat train_asr.py dan <RunName>.yaml, lalu setiap
    -IntervalMinutes menulis satu potret keadaan ke logs\<RunName>.monitor.log:
    proses hidup atau tidak, baris progres dan evaluasi terakhir dari
    logs\<RunName>.log, keadaan GPU dari nvidia-smi, sumber daya, dan skema
    daya. Keberadaan proses diperiksa setiap -PollSeconds; begitu proses keluar,
    potret akhir beserta ringkasan run_report.json ditulis dan skrip selesai.

    Setiap potret juga mencatat beban dan kinerja CPU, aktivitas
    disk, beban CPU proses teratas termasuk total seluruh proses python, status
    isi dan kuras baterai, serta peristiwa sambung-putus charger (Kernel-Power
    105) sejak potret sebelumnya, untuk menelusuri penyebab perlambatan
    pelatihan. Seluruhnya terbaca tanpa hak administrator.

    Skrip ini hanya membaca. Proses training dan berkasnya tidak disentuh; log
    training dibuka dengan FileShare.ReadWrite agar Tee-Object tetap dapat
    menulis.

.EXAMPLE
    pwsh -NoProfile -File monitor_run.ps1 -RunName xlsr_d1_trial
#>
param(
    [string]$RunName = 'xlsr_d1_trial',
    [int]$IntervalMinutes = 30,
    [int]$PollSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

$TrainLog = Join-Path $PSScriptRoot "logs\$RunName.log"
$MonitorLog = Join-Path $PSScriptRoot "logs\$RunName.monitor.log"
$RunReport = Join-Path $PSScriptRoot "outputs\runs\$RunName\run_report.json"
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$GpuFields = 'memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.max.sm,' +
    'clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,' +
    'clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown'
$OverlayNames = @{
    '961cc777-2547-4f9d-8174-7d86181b8a7a' = 'efisiensi daya terbaik'
    '00000000-0000-0000-0000-000000000000' = 'seimbang'
    'ded574b5-45a0-4f42-8737-46345c09c238' = 'kinerja terbaik'
}
$PowerSchemesKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes'
# Kinerja CPU dinyatakan relatif terhadap frekuensi dasar, sehingga dapat melampaui 100.
$CpuDiskCounters = [ordered]@{
    'utilitas CPU %' = '\Processor Information(_Total)\% Processor Utility'
    'kinerja CPU %'  = '\Processor Information(_Total)\% Processor Performance'
    'waktu CPU %'    = '\Processor(_Total)\% Processor Time'
    'disk sibuk %'   = '\PhysicalDisk(_Total)\% Disk Time'
    'disk MB/s'      = '\PhysicalDisk(_Total)\Disk Bytes/sec'
}
$CounterSamples = 5
$PowerEventLookbackMinutes = 30

# --------------------------------------------------------------------------
# Pembacaan
# --------------------------------------------------------------------------

function Write-Monitor([string[]]$Lines) {
    [System.IO.File]::AppendAllText($MonitorLog, (($Lines -join "`n") + "`n"), $Utf8NoBom)
}

function Get-TrainProcess {
    @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.CommandLine -and $_.CommandLine -like '*train_asr.py*' -and $_.CommandLine -like "*$RunName.yaml*"
    })
}

function Read-SharedLines([string]$Path) {
    if (-not (Test-Path $Path)) { return @() }
    $stream = [System.IO.FileStream]::new($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite)
    try {
        $reader = [System.IO.StreamReader]::new($stream, $Utf8NoBom)
        return @($reader.ReadToEnd() -split "`r?`n")
    } finally {
        $stream.Dispose()
    }
}

function Select-LastOrNone([object[]]$Items) {
    if ($Items.Count -eq 0) { return '(belum ada)' }
    return [string]$Items[-1]
}

function Get-GpuState {
    try {
        return ((& nvidia-smi "--query-gpu=$GpuFields" --format=csv,noheader 2>&1) | Out-String).Trim()
    } catch {
        return "nvidia-smi gagal: $($_.Exception.Message)"
    }
}

function Get-PowerState {
    $parts = @()
    try {
        $battery = @(Get-CimInstance Win32_Battery)
        if ($battery.Count -gt 0) {
            $source = switch ($battery[0].BatteryStatus) { 1 { 'baterai' } 2 { 'AC' } default { "status $_" } }
            $parts += "sumber $source, baterai $($battery[0].EstimatedChargeRemaining)%"
        }
    } catch {
        $parts += "baterai tidak terbaca"
    }
    try {
        $overlay = Get-ItemProperty $PowerSchemesKey -ErrorAction Stop
        foreach ($name in 'ActiveOverlayAcPowerScheme', 'ActiveOverlayDcPowerScheme') {
            $property = $overlay.PSObject.Properties[$name]
            if ($property) {
                $guid = [string]$property.Value
                $label = if ($OverlayNames.ContainsKey($guid)) { $OverlayNames[$guid] } else { 'tidak dikenal' }
                $parts += "$name $guid ($label)"
            }
        }
    } catch {
        $parts += "mode daya tidak terbaca"
    }
    $parts += ((& powercfg /getactivescheme 2>&1) | Out-String).Trim()
    return $parts -join '; '
}

function Get-CpuDiskState {
    try {
        $samples = @((Get-Counter -Counter @($CpuDiskCounters.Values) -SampleInterval 1 -MaxSamples $CounterSamples `
            -ErrorAction Stop).CounterSamples | Where-Object { $_.Status -eq 0 })
        $parts = foreach ($label in $CpuDiskCounters.Keys) {
            $suffix = $CpuDiskCounters[$label].ToLower()
            $values = @($samples | Where-Object { $_.Path.EndsWith($suffix) } | ForEach-Object { $_.CookedValue })
            if ($values.Count -eq 0) { "$label n/a"; continue }
            $average = ($values | Measure-Object -Average).Average
            if ($label -eq 'disk MB/s') { $average = $average / 1MB }
            '{0} {1:N1}' -f $label, $average
        }
        return "rerata $CounterSamples detik: " + ($parts -join ', ')
    } catch {
        return "tidak terbaca: $($_.Exception.Message)"
    }
}

function Get-TopCpuProcesses {
    # Persen terhadap seluruh prosesor logis; sampel proses yang keluar di tengah pengambilan disaring.
    try {
        $logical = [Environment]::ProcessorCount
        $samples = @((Get-Counter '\Process(*)\% Processor Time' -SampleInterval 2 -MaxSamples 1 `
            -ErrorAction SilentlyContinue).CounterSamples |
            Where-Object { $_.Status -eq 0 -and $_.InstanceName -notin '_total', 'idle' })
        $byName = @($samples | Group-Object InstanceName | ForEach-Object {
            [pscustomobject]@{
                Name  = $_.Name
                Pct   = ($_.Group | Measure-Object CookedValue -Sum).Sum / $logical
                Count = $_.Count
            }
        } | Sort-Object Pct -Descending)
        $python = @($byName | Where-Object { $_.Name -eq 'python' })
        $pythonText = if ($python.Count -gt 0) { '{0:N1}% ({1} proses)' -f $python[0].Pct, $python[0].Count } else { 'tidak ada' }
        $top = @($byName | Select-Object -First 5 | ForEach-Object { '{0} {1:N1}%' -f $_.Name, $_.Pct })
        return "python total $pythonText; teratas: $($top -join ', ')"
    } catch {
        return "tidak terbaca: $($_.Exception.Message)"
    }
}

function Get-BatteryFlow {
    try {
        $battery = @(Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction Stop)
        if ($battery.Count -eq 0) { return 'tidak ada baterai' }
        $b = $battery[0]
        return 'AC {0}, mengisi {1} {2} mW, menguras {3} {4} mW, kapasitas {5} mWh' -f `
            $b.PowerOnline, $b.Charging, $b.ChargeRate, $b.Discharging, $b.DischargeRate, $b.RemainingCapacity
    } catch {
        return "tidak terbaca: $($_.Exception.Message)"
    }
}

function Get-PowerEvents([datetime]$Since) {
    try {
        $events = @(Get-WinEvent -FilterHashtable @{
                LogName = 'System'; ProviderName = 'Microsoft-Windows-Kernel-Power'; Id = 105; StartTime = $Since
            } -ErrorAction SilentlyContinue | Sort-Object TimeCreated)
        if ($events.Count -eq 0) { return "tidak ada sejak $($Since.ToString('MM-dd HH:mm:ss'))" }
        $items = @($events | ForEach-Object {
            $ac = @(([xml]$_.ToXml()).Event.EventData.Data | Where-Object { $_.Name -eq 'AcOnline' } | ForEach-Object { $_.'#text' })
            '{0} AC={1}' -f $_.TimeCreated.ToString('MM-dd HH:mm:ss'), ($ac -join '')
        })
        return "$($items.Count) peristiwa: $($items -join '; ')"
    } catch {
        return "tidak terbaca: $($_.Exception.Message)"
    }
}

function Get-RunReportSummary {
    if (-not (Test-Path $RunReport)) { return 'run_report.json tidak ada' }
    $result = (Get-Content $RunReport -Raw -Encoding utf8 | ConvertFrom-Json).result
    return "global_step $($result.global_step), stopped_early $($result.stopped_early), " +
        "WER terbaik $($result.best_validation_wer) pada langkah $($result.best_step), " +
        "seconds_per_step $($result.seconds_per_step), puncak tercadang $($result.train_peak_reserved_mib) MiB, " +
        "teralokasi $($result.train_peak_allocated_mib) MiB"
}

# --------------------------------------------------------------------------
# Potret
# --------------------------------------------------------------------------

function Write-Snapshot([string]$Label, [switch]$Final) {
    # Fungsi PowerShell membuka larik kosong menjadi $null, dan $null.Count
    # melempar galat di bawah StrictMode; setiap pemanggilan karena itu dibungkus @().
    $since = $script:LastSnapshotAt
    $script:LastSnapshotAt = Get-Date
    $processes = @(Get-TrainProcess)
    $lines = Read-SharedLines $TrainLog
    $progress = @($lines | Where-Object { $_ -match '^\[progres\] langkah' })
    $evalWer = @($lines | Where-Object { $_ -match '^\[evaluasi\] langkah \d+: WER' })
    $evalCost = @($lines | Where-Object { $_ -match '^\[evaluasi\] langkah \d+: [\d.]+ detik' })
    $errors = @($lines | Where-Object { $_ -match '^Traceback|GAGAL|OutOfMemoryError|RuntimeError' })
    $rates = @($progress | Select-Object -Last 4 | ForEach-Object {
        if ($_ -match ': ([\d.]+) s/langkah') { $Matches[1] }
    })
    $processText = if ($processes.Count -gt 0) {
        'hidup, PID ' + (($processes | ForEach-Object { $_.ProcessId }) -join ', ')
    } else {
        'tidak ada'
    }

    $out = @(
        "==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Label ====",
        "Proses training    : $processText",
        "Progres terakhir   : $(Select-LastOrNone $progress)",
        "Laju 4 jendela     : $($rates -join ', ') s/langkah",
        "Evaluasi terakhir  : $(Select-LastOrNone $evalWer)",
        "Biaya evaluasi     : $(Select-LastOrNone $evalCost)",
        "Baris galat di log : $($errors.Count)",
        "GPU                : $(Get-GpuState)",
        "Daya               : $(Get-PowerState)",
        "CPU dan disk       : $(Get-CpuDiskState)",
        "Proses CPU         : $(Get-TopCpuProcesses)",
        "Baterai            : $(Get-BatteryFlow)",
        "Peristiwa charger  : $(Get-PowerEvents $since)"
    )
    if ($Final) {
        $out += "Laporan run        : $(Get-RunReportSummary)"
        $out += 'Baris akhir log    :'
        $out += @($lines | Where-Object { $_ -ne '' } | Select-Object -Last 8 | ForEach-Object { "    $_" })
    }
    Write-Monitor ($out + '')
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if (@(Get-TrainProcess).Count -eq 0) {
    Write-Monitor @("==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | GAGAL: proses train_asr.py untuk $RunName tidak ditemukan ====", '')
    exit 1
}

Write-Monitor @(
    ("# Pemantau $RunName dimulai $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), potret setiap $IntervalMinutes menit, " +
        "pemeriksaan proses setiap $PollSeconds detik."),
    "# Kolom GPU: $GpuFields",
    ("# CPU dan disk: rerata $CounterSamples sampel per detik. Proses CPU: persen terhadap seluruh prosesor logis. " +
        "Peristiwa charger: Kernel-Power 105 sejak potret sebelumnya; potret awal mundur $PowerEventLookbackMinutes menit."),
    ''
)
$script:LastSnapshotAt = (Get-Date).AddMinutes(-$PowerEventLookbackMinutes)
Write-Snapshot 'awal'
$nextSnapshot = (Get-Date).AddMinutes($IntervalMinutes)

while ($true) {
    Start-Sleep -Seconds $PollSeconds
    if (@(Get-TrainProcess).Count -eq 0) {
        Write-Snapshot 'akhir: proses training sudah keluar' -Final
        break
    }
    if ((Get-Date) -ge $nextSnapshot) {
        Write-Snapshot 'berkala'
        $nextSnapshot = $nextSnapshot.AddMinutes($IntervalMinutes)
    }
}
