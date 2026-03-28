param(
    [string]$ArchiveDir = "data/archive",
    [string]$OutputDir = "data/processed",
    [string]$PreferredRevision = "r1"
)

$ErrorActionPreference = "Stop"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

$labelMap = @{
    "0" = "entailment"
    "1" = "neutral"
    "2" = "contradiction"
}

function Normalize-Text {
    param([string]$Text)
    if ($null -eq $Text) {
        return ""
    }

    $normalized = $Text -replace "\s+", " "
    return $normalized.Trim()
}

function Write-Utf8NoBomJson {
    param(
        [string]$OutputPath,
        [object]$Payload
    )

    $parent = Split-Path -Parent $OutputPath
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    $json = $Payload | ConvertTo-Json -Depth 6
    [System.IO.File]::WriteAllText((Resolve-Path $parent | Join-Path -ChildPath (Split-Path $OutputPath -Leaf)), $json, $utf8NoBom)
}

function Convert-Split {
    param(
        [string]$InputPath,
        [string]$OutputPath
    )

    $rows = Import-Csv $InputPath
    $seen = @{}
    $cleaned = New-Object System.Collections.Generic.List[object]
    $stats = [ordered]@{
        input_rows = 0
        kept_rows = 0
        removed_missing = 0
        removed_unknown_label = 0
        removed_duplicates = 0
    }

    foreach ($row in $rows) {
        $stats.input_rows += 1

        $premise = Normalize-Text $row.premise
        $hypothesis = Normalize-Text $row.hypothesis
        $labelKey = Normalize-Text $row.label
        $reason = Normalize-Text $row.reason
        $uid = Normalize-Text $row.uid

        if ([string]::IsNullOrWhiteSpace($premise) -or [string]::IsNullOrWhiteSpace($hypothesis) -or [string]::IsNullOrWhiteSpace($labelKey)) {
            $stats.removed_missing += 1
            continue
        }

        if (-not $labelMap.ContainsKey($labelKey)) {
            $stats.removed_unknown_label += 1
            continue
        }

        $dedupeKey = "{0}`t{1}`t{2}" -f $premise, $hypothesis, $labelKey
        if ($seen.ContainsKey($dedupeKey)) {
            $stats.removed_duplicates += 1
            continue
        }

        $seen[$dedupeKey] = $true
        $cleaned.Add([ordered]@{
            uid = $uid
            premise = $premise
            hypothesis = $hypothesis
            label = $labelMap[$labelKey]
            reason = $reason
        })
    }

    $stats.kept_rows = $cleaned.Count
    Write-Utf8NoBomJson -OutputPath $OutputPath -Payload $cleaned
    return $stats
}

if (-not (Test-Path $ArchiveDir)) {
    throw "Archive directory not found: $ArchiveDir"
}

$revisions = @("r1", "r2", "r3")
$splitMap = [ordered]@{
    train = "train"
    dev = "validation"
    test = "test"
}

$summary = New-Object System.Collections.Generic.List[object]

foreach ($revision in $revisions) {
    foreach ($sourceSplit in $splitMap.Keys) {
        $targetSplit = $splitMap[$sourceSplit]
        $inputPath = Join-Path $ArchiveDir ("{0}_{1}.csv" -f $sourceSplit, $revision)
        if (-not (Test-Path $inputPath)) {
            continue
        }

        $outputPath = Join-Path $OutputDir (Join-Path $revision ("{0}.json" -f $targetSplit))
        $stats = Convert-Split -InputPath $inputPath -OutputPath $OutputPath
        $summary.Add([ordered]@{
            revision = $revision
            split = $targetSplit
            input_path = $inputPath
            output_path = $outputPath
            input_rows = $stats.input_rows
            kept_rows = $stats.kept_rows
            removed_missing = $stats.removed_missing
            removed_unknown_label = $stats.removed_unknown_label
            removed_duplicates = $stats.removed_duplicates
        })
    }
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
}

$summaryPath = Join-Path $OutputDir "conversion_summary.json"
Write-Utf8NoBomJson -OutputPath $summaryPath -Payload $summary

$preferredDir = Join-Path $OutputDir $PreferredRevision
foreach ($targetSplit in $splitMap.Values) {
    $sourcePath = Join-Path $preferredDir ("{0}.json" -f $targetSplit)
    if (Test-Path $sourcePath) {
        Copy-Item -Path $sourcePath -Destination (Join-Path "data" ("{0}.json" -f $targetSplit)) -Force
    }
}

Write-Output "Converted archive CSV files into cleaned JSON datasets."
Write-Output "Preferred revision copied to data/train.json, data/validation.json, and data/test.json."
Write-Output "Summary written to $summaryPath."
