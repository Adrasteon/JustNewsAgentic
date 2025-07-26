# PowerShell script to build and start agents sequentially

# List of agents
$agents = @(
    "mcp_bus",
    "chief_editor",
    "scout",
    "fact_checker",
    "analyst",
    "synthesizer",
    "critic"
)

# Function to build and start an agent
function BuildAndStart-Agent {
    param (
        [string]$AgentName
    )

    Write-Host "Building and starting agent: $AgentName"

    # Navigate to the agent's directory
    $agentPath = "agents/$AgentName"
    if (!(Test-Path $agentPath)) {
        Write-Host "Agent directory not found: $agentPath" -ForegroundColor Red
        return
    }

    Push-Location $agentPath

    try {
        # Build the Docker image
        docker build -t $AgentName .
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to build agent: $AgentName" -ForegroundColor Red
            Pop-Location
            return
        }

        # Start the Docker container
        docker run -d --name $AgentName $AgentName
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to start agent: $AgentName" -ForegroundColor Red
            Pop-Location
            return
        }

        Write-Host "Agent $AgentName started successfully" -ForegroundColor Green
    } catch {
        Write-Host "Error occurred while processing agent: $AgentName" -ForegroundColor Red
    } finally {
        Pop-Location
    }
}

# Start MCP Bus first
Write-Host "Starting MCP Bus..."
docker-compose up -d mcp_bus

# Iterate through agents and build/start them sequentially
foreach ($agent in $agents) {
    BuildAndStart-Agent -AgentName $agent
}

Write-Host "All agents started."
