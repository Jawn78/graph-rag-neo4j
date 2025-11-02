Param(
    [string]$Neo4jHome = $env:NEO4J_HOME,
    [int]$WaitSeconds = 30
)

# Helper function for error handling
function Exit-WithError {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

function Wait-ForNeo4j {
    param(
        [string]$ServiceName,
        [int]$TimeoutSeconds = 30
    )
    
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $service = Get-Service $ServiceName -ErrorAction SilentlyContinue
        if ($service.Status -eq "Running") {
            # Try to connect to verify it's really ready
            try {
                $result = Invoke-WebRequest "http://localhost:7474" -UseBasicParsing
                if ($result.StatusCode -eq 200) {
                    Write-Host "Neo4j is ready!"
                    return $true
                }
            } catch {
                # Still starting up
            }
        }
        Write-Host "Waiting for Neo4j to start... ($([int]$timer.Elapsed.TotalSeconds)s)"
        Start-Sleep -Seconds 2
    }
    return $false
}

# Check for Neo4j installation
if ($Neo4jHome -and (Test-Path "$Neo4jHome\bin\neo4j.bat")) {
    Write-Host "Starting Neo4j from $Neo4jHome..."
    try {
        & "$Neo4jHome\bin\neo4j.bat" start
        if ($LASTEXITCODE -ne 0) {
            Exit-WithError "Failed to start Neo4j using neo4j.bat"
        }
        
        Write-Host "Checking Neo4j status..."
        & "$Neo4jHome\bin\neo4j.bat" status
        if ($LASTEXITCODE -ne 0) {
            Exit-WithError "Neo4j status check failed"
        }
    } catch {
        Exit-WithError "Error running Neo4j commands: $_"
    }
} else {
    Write-Host "Looking for Neo4j service..."
    $svc = Get-Service | Where-Object { $_.Name -match "neo4j" -or $_.DisplayName -match "Neo4j" } | Select-Object -First 1
    
    if ($svc) {
        Write-Host "Found Neo4j service: $($svc.Name)"
        if ($svc.Status -ne "Running") {
            try {
                Start-Service $svc.Name
                if (-not (Wait-ForNeo4j -ServiceName $svc.Name -TimeoutSeconds $WaitSeconds)) {
                    Exit-WithError "Neo4j did not start within $WaitSeconds seconds"
                }
            } catch {
                Exit-WithError "Failed to start Neo4j service: $_"
            }
        } else {
            Write-Host "Neo4j service is already running"
        }
        Get-Service $svc.Name
    } else {
        Write-Host @"
Neo4j not found. Either:
1. Set NEO4J_HOME environment variable to your Neo4j installation directory
2. Install via winget:
   winget install Neo4j.Neo4j-Community
"@ -ForegroundColor Yellow
        exit 1
    }
}
