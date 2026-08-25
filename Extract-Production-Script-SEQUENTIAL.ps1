<#
.SYNOPSIS
    PowerShell 5.1 Sequential SQLite Data Extractor - Guaranteed Working
    
.DESCRIPTION
    Simplified extraction script optimized for PowerShell 5.1:
    - Sequential processing (no runspace issues)
    - 100% PowerShell 5.1 compatible
    - ASCII-only characters
    - Compatible with ANN database structure
    - Memory management and optimization
    - Excel export with CSV fallback
    - Detailed error reporting
    
.PARAMETER InputFolder
    Directory containing SQLite database files
    
.PARAMETER OutputFolder  
    Directory where Excel files will be created
    
.PARAMETER NonInteractive
    Run without user prompts
    
.PARAMETER MaxObjects
    Maximum objects to process per oilfield (for testing)
#>

param(
    [string]$InputFolder = "",
    [string]$OutputFolder = "", 
    [switch]$NonInteractive,
    [string]$SelectField = "",
    [string]$SelectObject = "",
    [int]$MaxObjects = 0,
    [switch]$UseIncrementalMode,
    [int]$SkipLargeObjects = 0,
    [switch]$Verbose
)

# Configuration
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$Interactive = -not $NonInteractive

# UTF-8 encoding setup
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
    
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        try {
            chcp 65001 | Out-Null
        }
        catch {
            # Ignore chcp errors
        }
    }
}
catch {
    Write-Warning "Could not set UTF-8 encoding"
}

# PowerShell optimizations
if ($PSVersionTable.PSVersion.Major -ge 5) {
    $PSDefaultParameterValues = @{
        '*:Encoding' = 'UTF8'
        'Out-File:Encoding' = 'UTF8'
    }
}

Write-Host ""
Write-Host "SQL Extractor - Sequential Processing Version" -ForegroundColor Magenta
Write-Host "=============================================" -ForegroundColor Magenta
$psVersion = $PSVersionTable.PSVersion.ToString()
Write-Host "PowerShell Version: $psVersion" -ForegroundColor Cyan
Write-Host "Processing Mode: Sequential (no parallel issues)" -ForegroundColor Yellow

# SQLite detection
$SQLitePath = ".\sqlite3.exe"
if (-not (Test-Path $SQLitePath)) {
    $SQLitePath = "sqlite3.exe"
    if (-not (Get-Command $SQLitePath -ErrorAction SilentlyContinue)) {
        Write-Error "sqlite3.exe not found. Please ensure it's in PATH or in the script directory."
        exit 1
    }
}
Write-Host "Using SQLite: $SQLitePath" -ForegroundColor Green

# ImportExcel module
try {
    Import-Module ImportExcel -Force -ErrorAction SilentlyContinue
    $HasImportExcel = $true
    Write-Host "ImportExcel module loaded - Excel export enabled" -ForegroundColor Green
}
catch {
    Write-Host "Using CSV export - ImportExcel not available" -ForegroundColor Yellow
    $HasImportExcel = $false
}

# UTF-8 string handling
function ConvertTo-UTF8String {
    param([string]$InputString)
    
    if ([string]::IsNullOrEmpty($InputString)) {
        return $InputString
    }
    
    try {
        $utf8Bytes = [System.Text.Encoding]::UTF8.GetBytes($InputString)
        return [System.Text.Encoding]::UTF8.GetString($utf8Bytes)
    }
    catch {
        return $InputString
    }
}

# SQLite query function with better error handling
function Invoke-SQLiteQueryOptimized {
    param(
        [string]$DatabasePath,
        [string]$Query,
        [int]$TimeoutSeconds = 300
    )
    
    try {
        if ($Verbose) {
            Write-Host "    Executing query on: $DatabasePath" -ForegroundColor Gray
        }
        
        # Performance settings
        $pragmas = @(
            "PRAGMA synchronous = OFF",
            "PRAGMA journal_mode = MEMORY", 
            "PRAGMA temp_store = MEMORY",
            "PRAGMA cache_size = 100000"
        )
        
        # Apply pragmas one by one
        foreach ($pragma in $pragmas) {
            try {
                $pragmaArgs = @($DatabasePath, $pragma)
                & $SQLitePath $pragmaArgs | Out-Null
                if ($LASTEXITCODE -ne 0 -and $Verbose) {
                    Write-Warning "    PRAGMA failed: $pragma"
                }
            }
            catch {
                if ($Verbose) {
                    Write-Warning "    PRAGMA error: $pragma"
                }
            }
        }
        
        # Execute main query
        $originalEncoding = [Console]::OutputEncoding
        try {
            [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
            $queryArgs = @($DatabasePath, "-header", "-csv", $Query)
            $output = & $SQLitePath $queryArgs
        }
        finally {
            [Console]::OutputEncoding = $originalEncoding
        }
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "SQLite query failed for $DatabasePath with exit code $LASTEXITCODE"
            if ($Verbose) {
                Write-Host "    Failed query: $Query" -ForegroundColor Red
            }
            return $null
        }
        
        if (-not $output -or $output.Count -eq 0) {
            if ($Verbose) {
                Write-Host "    Query returned no results" -ForegroundColor Yellow
            }
            return $null
        }
        
        # Convert to objects
        try {
            $result = $output | ConvertFrom-Csv
            
            if (-not $result) {
                if ($Verbose) {
                    Write-Host "    ConvertFrom-Csv returned null" -ForegroundColor Yellow
                }
                return $null
            }
            
            # Handle UTF-8
            foreach ($row in $result) {
                if ($row) {
                    foreach ($property in $row.PSObject.Properties) {
                        if ($property.Value -is [string]) {
                            $property.Value = ConvertTo-UTF8String -InputString $property.Value
                        }
                    }
                }
            }
            
            if ($Verbose) {
                Write-Host "    Query returned " + $result.Count + " rows" -ForegroundColor Gray
            }
            
            return $result
        }
        catch {
            Write-Warning "Failed to convert SQLite output to objects: $($_.Exception.Message)"
            return $null
        }
    }
    catch {
        Write-Warning "Query execution failed: $($_.Exception.Message)"
        if ($Verbose) {
            Write-Host "    Query: $Query" -ForegroundColor Red
            Write-Host "    Database: $DatabasePath" -ForegroundColor Red
        }
        return $null
    }
}

# Date formatting
function Format-DateOptimized {
    param([string]$DateString)
    
    if ([string]::IsNullOrWhiteSpace($DateString)) {
        return ""
    }
    
    if ($DateString -match '^(\d{4})[/-](\d{1,2})[/-](\d{1,2})') {
        $year = $Matches[1]
        $month = [int]$Matches[2]
        $day = [int]$Matches[3]
        return "{0:D2}.{1:D2}.{2}" -f $day, $month, $year
    }
    
    return $DateString
}

# Excel export with better error handling
function Export-ExcelOptimized {
    param(
        [array]$Data,
        [string]$Path,
        [string]$WorksheetName,
        [int]$MaxRowsPerSheet = 500000
    )
    
    if (-not $Data -or $Data.Count -eq 0) {
        Write-Warning "No data to export for $WorksheetName"
        return 0
    }
    
    try {
        # Clean worksheet name - remove invalid chars
        $cleanName = $WorksheetName -replace '[<>:"/\\|?*\[\]]', '_'
        $cleanName = $cleanName.Substring(0, [Math]::Min($cleanName.Length, 31))
        
        if ($HasImportExcel -and $Data.Count -le $MaxRowsPerSheet) {
            try {
                $Data | Export-Excel -Path $Path -WorksheetName $cleanName -AutoSize -TableStyle Medium2 -FreezeTopRow -ClearSheet
                return 1
            }
            catch {
                Write-Warning "Excel export failed: $($_.Exception.Message)"
                # Fall through to CSV
            }
        }
        
        # CSV export (fallback or large dataset)
        $csvPath = $Path -replace '\.xlsx$', "_$cleanName.csv"
        try {
            $Data | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
            $csvFileName = Split-Path $csvPath -Leaf
            $csvMessage = "      Exported to CSV: " + $csvFileName
            Write-Host $csvMessage -ForegroundColor Yellow
            return 1
        }
        catch {
            Write-Error "CSV export failed: $($_.Exception.Message)"
            return 0
        }
    }
    catch {
        Write-Error "Export function failed: $($_.Exception.Message)"
        return 0
    }
}

# Main database processing function - SEQUENTIAL
function Process-DatabaseSequential {
    param(
        [string]$DatabasePath,
        [string]$OutputFolder,
        [string]$SelectedField = "",
        [string]$SelectedObject = ""
    )
    
    try {
        $dbFileName = [System.IO.Path]::GetFileNameWithoutExtension($DatabasePath)
        $processingMessage = "Processing: " + $dbFileName
        Write-Host $processingMessage -ForegroundColor Green
        
        # Incremental mode check
        if ($UseIncrementalMode) {
            $existingFiles = Get-ChildItem -Path $OutputFolder -Filter "*$dbFileName*.xlsx" -ErrorAction SilentlyContinue
            if ($existingFiles) {
                $dbLastWrite = (Get-Item $DatabasePath).LastWriteTime
                $newestOutput = ($existingFiles | Sort-Object LastWriteTime -Descending)[0].LastWriteTime
                
                if ($newestOutput -gt $dbLastWrite) {
                    Write-Host "  Skipping - output newer than database in incremental mode" -ForegroundColor Yellow
                    return
                }
            }
        }
        
        # Test database connectivity
        $testQuery = "SELECT COUNT(*) as TableCount FROM sqlite_master WHERE type='table'"
        $testResult = Invoke-SQLiteQueryOptimized -DatabasePath $DatabasePath -Query $testQuery
        if (-not $testResult) {
            Write-Error "Cannot connect to database: $DatabasePath"
            return
        }
        
        if ($Verbose) {
            Write-Host "  Database connectivity test passed" -ForegroundColor Gray
        }
        
        # Get oilfields
        Write-Host "  Getting oilfields..." -ForegroundColor Cyan
        $oilfieldQuery = "SELECT OILFIELD_ID, OILFIELD_NAME FROM DW_OILFIELD ORDER BY OILFIELD_NAME"
        $oilfields = Invoke-SQLiteQueryOptimized -DatabasePath $DatabasePath -Query $oilfieldQuery
        
        if (-not $oilfields -or $oilfields.Count -eq 0) {
            Write-Warning "  No oilfields found or database query failed"
            return
        }
        
        $oilfieldCount = $oilfields.Count
        $oilfieldFoundMessage = "  Found " + $oilfieldCount + " oilfields in database"
        Write-Host $oilfieldFoundMessage -ForegroundColor Cyan
        
        # Filter oilfields if specified
        if ($SelectedField -and $SelectedField.ToUpper() -ne "ALL") {
            $oilfields = $oilfields | Where-Object { 
                $_.OILFIELD_NAME -like "*$SelectedField*" -or $_.OILFIELD_ID -eq $SelectedField 
            }
            $filteredCount = $oilfields.Count
            $filterMessage = "  Filtered to " + $filteredCount + " oilfields matching '" + $SelectedField + "'"
            Write-Host $filterMessage -ForegroundColor Yellow
        }
        
        $processedFields = 0
        $totalSheets = 0
        
        foreach ($oilfield in $oilfields) {
            try {
                $processedFields++
                $fieldName = $oilfield.OILFIELD_NAME -replace '[<>:"/\\|?*]', '_'
                $outputFile = Join-Path $OutputFolder "production_${fieldName}_${dbFileName}.xlsx"
                
                $oilfieldName = $oilfield.OILFIELD_NAME
                $totalFieldCount = $oilfields.Count
                $fieldProcessMessage = "  [" + $processedFields + "/" + $totalFieldCount + "] Processing field: " + $oilfieldName
                Write-Host $fieldProcessMessage -ForegroundColor Cyan
                
                # Get objects filtered by oilfield ID
                $oilfieldId = $oilfield.OILFIELD_ID
                Write-Host "    Getting objects for oilfield ID: $oilfieldId..." -ForegroundColor Gray
                
                $objectQuery = "SELECT DISTINCT o.OBJECT_ID, o.OBJECT_NAME, (SELECT COUNT(*) FROM DW_MTH_OP_RAP r WHERE r.OBJECT_ID = o.OBJECT_ID) as RecordCount FROM DW_OBJECT o WHERE o.OILFIELD_ID = $oilfieldId AND EXISTS (SELECT 1 FROM DW_MTH_OP_RAP r WHERE r.OBJECT_ID = o.OBJECT_ID) ORDER BY o.OBJECT_NAME"
                
                $objects = Invoke-SQLiteQueryOptimized -DatabasePath $DatabasePath -Query $objectQuery
                
                if (-not $objects -or $objects.Count -eq 0) {
                    Write-Warning "    No objects with production data found for this oilfield"
                    continue
                }
                
                $objectCount = $objects.Count
                $objectFoundMessage = "    Found " + $objectCount + " objects with production data"
                Write-Host $objectFoundMessage -ForegroundColor Gray
                
                # Filter objects if specified
                if ($SelectedObject -and $SelectedObject.ToUpper() -ne "ALL") {
                    $objects = $objects | Where-Object { 
                        $_.OBJECT_NAME -like "*$SelectedObject*" -or $_.OBJECT_ID -eq $SelectedObject 
                    }
                    $filteredObjectCount = $objects.Count
                    $objectFilterMessage = "    Filtered to " + $filteredObjectCount + " objects matching '" + $SelectedObject + "'"
                    Write-Host $objectFilterMessage -ForegroundColor Gray
                }
                
                # Skip large objects if specified
                if ($SkipLargeObjects -gt 0) {
                    $originalCount = $objects.Count
                    $objects = $objects | Where-Object { [int]$_.RecordCount -le $SkipLargeObjects }
                    if ($objects.Count -lt $originalCount) {
                        $skippedCount = $originalCount - $objects.Count
                        $skipMessage = "    Skipped " + $skippedCount + " large objects (>" + $SkipLargeObjects + " records)"
                        Write-Host $skipMessage -ForegroundColor Yellow
                    }
                }
                
                # Limit objects for testing
                if ($MaxObjects -gt 0 -and $objects.Count -gt $MaxObjects) {
                    $objects = $objects | Select-Object -First $MaxObjects
                    $limitMessage = "    Limited to first " + $MaxObjects + " objects for testing"
                    Write-Host $limitMessage -ForegroundColor Yellow
                }
                
                if ($objects.Count -eq 0) {
                    Write-Warning "    No objects remaining after filtering"
                    continue
                }
                
                # Process objects SEQUENTIALLY
                $exportedSheets = 0
                $objectIndex = 0
                
                foreach ($object in $objects) {
                    try {
                        $objectIndex++
                        $objectId = $object.OBJECT_ID
                        $objectName = $object.OBJECT_NAME -replace '[<>:"/\\|?*\[\]]', '_'
                        $recordCount = [int]$object.RecordCount
                        
                        $objectProgressMessage = "      [" + $objectIndex + "/" + $objects.Count + "] Processing: " + $objectName + " [" + $recordCount + " records]"
                        Write-Host $objectProgressMessage -ForegroundColor Gray
                        
                        # Production query for ANN database
                        $productionQuery = "SELECT substr(r.DT, 1, 10) as Date, COALESCE(w.WELL_NAME, 'Well_' || r.WELL_ID) as Well, ROUND(COALESCE(r.OIL_T, 0.0), 3) as Oil_Produced_Tonnes, ROUND(COALESCE(r.OIL_M3, 0.0), 3) as Oil_Produced_M3, ROUND(COALESCE(r.WAT_LIQ_INJ_T, 0.0), 3) as Water_Produced_Tonnes, ROUND(COALESCE(r.WAT_LIQ_INJ_M3, 0.0), 3) as Water_Produced_M3, ROUND(COALESCE(r.GAS_NAT_M3, 0.0), 3) as Natural_Gas_M3, ROUND(COALESCE(r.GAS_ASC_INJ_M3, 0.0), 3) as Associated_Gas_M3, ROUND(COALESCE(r.GAS_COND_T, 0.0), 3) as Condensate_Tonnes, ROUND(COALESCE(r.GAS_COND_M3, 0.0), 3) as Condensate_M3, ROUND(COALESCE(r.WORKTIME / 24.0, 0.0), 2) as Working_Days FROM DW_MTH_OP_RAP r LEFT JOIN DW_WELLS w ON r.WELL_ID = w.WELL_ID WHERE r.OBJECT_ID = $objectId ORDER BY r.DT, w.WELL_NAME"
                        
                        # Execute query
                        $productionData = Invoke-SQLiteQueryOptimized -DatabasePath $DatabasePath -Query $productionQuery
                        
                        if ($productionData -and $productionData.Count -gt 0) {
                            # Format dates
                            foreach ($row in $productionData) {
                                if ($row.Date) {
                                    $row.Date = Format-DateOptimized -DateString $row.Date
                                }
                            }
                            
                            # Export to Excel/CSV
                            $sheets = Export-ExcelOptimized -Data $productionData -Path $outputFile -WorksheetName $objectName
                            $exportedSheets += $sheets
                            
                            if ($sheets -gt 0) {
                                $exportSuccessMessage = "        Exported successfully [" + $productionData.Count + " rows]"
                                Write-Host $exportSuccessMessage -ForegroundColor Green
                            }
                        }
                        else {
                            Write-Warning "        No production data found for this object"
                        }
                        
                        # Memory cleanup
                        if ($objectIndex % 5 -eq 0) {
                            [System.GC]::Collect()
                        }
                    }
                    catch {
                        $objectErrorMessage = "        Object processing failed: " + $_.Exception.Message
                        Write-Warning $objectErrorMessage
                    }
                }
                
                if ($exportedSheets -gt 0) {
                    $totalSheets += $exportedSheets
                    $outputFileName = Split-Path $outputFile -Leaf
                    $completedMessage = "    * Completed: " + $outputFileName + " [" + $exportedSheets + " sheets]"
                    Write-Host $completedMessage -ForegroundColor Green
                }
                else {
                    $noDataMessage = "    No data exported for " + $oilfieldName
                    Write-Warning $noDataMessage
                }
            }
            catch {
                $fieldErrorMessage = "  Field processing failed: " + $_.Exception.Message
                Write-Warning $fieldErrorMessage
            }
        }
        
        if ($totalSheets -gt 0) {
            $dbCompleteMessage = "  Database processing complete: " + $totalSheets + " total sheets exported"
            Write-Host $dbCompleteMessage -ForegroundColor Green
        }
        else {
            Write-Warning "  No data was exported from this database"
        }
    }
    catch {
        $dbErrorMessage = "Database processing failed: " + $_.Exception.Message
        Write-Error $dbErrorMessage
        if ($Verbose) {
            Write-Host "Stack trace: " + $_.ScriptStackTrace -ForegroundColor Red
        }
    }
}

# Main execution
try {
    # Directory setup
    if ([string]::IsNullOrWhiteSpace($InputFolder)) {
        if ($Interactive) {
            $InputFolder = Read-Host "Enter input folder containing SQLite databases"
        }
        else {
            $InputFolder = Get-Location
        }
    }
    
    if ([string]::IsNullOrWhiteSpace($OutputFolder)) {
        if ($Interactive) {
            $OutputFolder = Read-Host "Enter output folder for Excel files"
        }
        else {
            $OutputFolder = Join-Path $InputFolder "exported"
        }
    }
    
    # Create output folder
    if (-not (Test-Path $OutputFolder)) {
        New-Item -ItemType Directory -Path $OutputFolder -Force | Out-Null
        $dirMessage = "Created output directory: " + $OutputFolder
        Write-Host $dirMessage -ForegroundColor Green
    }
    
    # Find database files
    $dbFiles = Get-ChildItem -Path $InputFolder -Filter "*.sqldb" -Recurse
    
    if (-not $dbFiles) {
        $noDbMessage = "No .sqldb files found in " + $InputFolder
        Write-Error $noDbMessage
        exit 1
    }
    
    # Display summary
    Write-Host ""
    Write-Host "=== PROCESSING SUMMARY ===" -ForegroundColor Magenta
    $dbCount = $dbFiles.Count
    $dbFoundMessage = "Found databases: " + $dbCount
    Write-Host $dbFoundMessage -ForegroundColor Green
    $inputMessage = "Input folder: " + $InputFolder
    Write-Host $inputMessage -ForegroundColor Green
    $outputMessage = "Output folder: " + $OutputFolder
    Write-Host $outputMessage -ForegroundColor Green
    
    if ($SelectField) {
        $fieldMessage = "Selected field: " + $SelectField
        Write-Host $fieldMessage -ForegroundColor Green
    }
    else {
        Write-Host "Selected field: All fields" -ForegroundColor Green
    }
    
    if ($SelectObject) {
        $objectMessage = "Selected object: " + $SelectObject
        Write-Host $objectMessage -ForegroundColor Green
    }
    else {
        Write-Host "Selected object: All objects" -ForegroundColor Green
    }
    
    if ($MaxObjects -gt 0) {
        $maxMessage = "Max objects per field: " + $MaxObjects
        Write-Host $maxMessage -ForegroundColor Yellow
    }
    
    if ($UseIncrementalMode) {
        Write-Host "Incremental mode: Enabled" -ForegroundColor Green
    }
    else {
        Write-Host "Incremental mode: Disabled" -ForegroundColor Green
    }
    
    if ($SkipLargeObjects -gt 0) {
        $skipMessage = "Skip large objects: >" + $SkipLargeObjects + " records"
        Write-Host $skipMessage -ForegroundColor Yellow
    }
    
    if ($Verbose) {
        Write-Host "Verbose logging: Enabled" -ForegroundColor Green
    }
    
    Write-Host ""
    
    # Process databases
    $totalStartTime = Get-Date
    $totalFilesProcessed = 0
    
    for ($i = 0; $i -lt $dbFiles.Count; $i++) {
        $dbFile = $dbFiles[$i]
        $startTime = Get-Date
        
        $currentNum = $i + 1
        $totalNum = $dbFiles.Count
        $progressPrefix = "[" + $currentNum + "/" + $totalNum + "] "
        Write-Host $progressPrefix -NoNewline -ForegroundColor White
        
        # Get file size
        $fileSizeMB = [math]::Round($dbFile.Length / 1MB, 2)
        $sizeInfo = "[" + $fileSizeMB + " MB] "
        Write-Host $sizeInfo -NoNewline -ForegroundColor Gray
        
        Process-DatabaseSequential -DatabasePath $dbFile.FullName -OutputFolder $OutputFolder -SelectedField $SelectField -SelectedObject $SelectObject
        
        $elapsed = (Get-Date) - $startTime
        $elapsedMinutes = [math]::Round($elapsed.TotalMinutes, 2)
        if ($elapsed.TotalMinutes -gt 0) {
            $throughputMBMin = [math]::Round($fileSizeMB / $elapsed.TotalMinutes, 2)
            $timeMessage = "  Processing time: " + $elapsedMinutes + " minutes [" + $throughputMBMin + " MB/min]"
        }
        else {
            $timeMessage = "  Processing time: " + $elapsedMinutes + " minutes"
        }
        Write-Host $timeMessage -ForegroundColor Green
        
        $totalFilesProcessed++
        
        # Memory management
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
    }
    
    # Final summary
    $totalElapsed = (Get-Date) - $totalStartTime
    $outputFiles = Get-ChildItem -Path $OutputFolder -Filter "*.xlsx" -ErrorAction SilentlyContinue
    $outputCSVs = Get-ChildItem -Path $OutputFolder -Filter "*.csv" -ErrorAction SilentlyContinue
    
    $allFiles = @()
    if ($outputFiles) {
        $allFiles += $outputFiles
    }
    if ($outputCSVs) {
        $allFiles += $outputCSVs
    }
    
    $totalOutputSizeMB = 0
    if ($allFiles.Count -gt 0) {
        $totalOutputSizeMB = [math]::Round(($allFiles | Measure-Object Length -Sum).Sum / 1MB, 2)
    }
    
    Write-Host ""
    Write-Host "EXTRACTION COMPLETE!" -ForegroundColor Magenta
    Write-Host "=====================" -ForegroundColor Magenta
    Write-Host "PERFORMANCE METRICS:" -ForegroundColor Yellow
    $processedMessage = "  Databases processed: " + $totalFilesProcessed
    Write-Host $processedMessage -ForegroundColor Green
    
    $totalMinutes = [math]::Round($totalElapsed.TotalMinutes, 2)
    $totalTimeMessage = "  Total processing time: " + $totalMinutes + " minutes"
    Write-Host $totalTimeMessage -ForegroundColor Green
    
    $avgMinutes = 0
    if ($dbFiles.Count -gt 0) {
        $avgMinutes = [math]::Round($totalElapsed.TotalMinutes / $dbFiles.Count, 2)
    }
    $avgTimeMessage = "  Average per database: " + $avgMinutes + " minutes"
    Write-Host $avgTimeMessage -ForegroundColor Green
    
    if ($outputFiles) {
        $excelCount = $outputFiles.Count
        $excelMessage = "  Excel files created: " + $excelCount
        Write-Host $excelMessage -ForegroundColor Green
    }
    
    if ($outputCSVs) {
        $csvCount = $outputCSVs.Count
        $csvMessage = "  CSV files created: " + $csvCount
        Write-Host $csvMessage -ForegroundColor Green
    }
    
    $sizeMessage = "  Total output size: " + $totalOutputSizeMB + " MB"
    Write-Host $sizeMessage -ForegroundColor Green
    $savedMessage = "  Files saved to: " + $OutputFolder
    Write-Host $savedMessage -ForegroundColor Yellow
    Write-Host ""
    Write-Host "All processing completed successfully!" -ForegroundColor Cyan
    Write-Host "* Sequential processing (no runspace issues)" -ForegroundColor Cyan
    Write-Host "* UTF-8 encoding for Cyrillic text" -ForegroundColor Cyan
    Write-Host "* Robust error handling" -ForegroundColor Cyan
    Write-Host "* Detailed progress reporting" -ForegroundColor Cyan
}
catch {
    $errorMessage = $_.Exception.Message
    $finalErrorMessage = "Extraction failed: " + $errorMessage
    Write-Error $finalErrorMessage
    $errorDetails = "Error details: " + $errorMessage
    Write-Host $errorDetails -ForegroundColor Red
    if ($Verbose) {
        Write-Host "Stack trace: " + $_.ScriptStackTrace -ForegroundColor Red
    }
    exit 1
}
finally {
    # Final cleanup
    [System.GC]::Collect()
}