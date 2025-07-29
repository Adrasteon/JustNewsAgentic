#!/bin/bash
# Start JustNews V4 Services as Background Daemons
# Focus on Scout Agent first

echo "🚀 Starting JustNews V4 Services as Background Daemons"
echo "======================================================"

# First, stop any existing services to prevent conflicts
echo "🛑 Stopping existing services..."

# Kill existing processes by port
echo "   Checking for processes on ports 8000, 8002, 8007..."
for port in 8000 8002 8007; do
    PID=$(lsof -t -i:$port 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo "   Killing process on port $port (PID: $PID)"
        kill -9 $PID 2>/dev/null || true
        sleep 1
    fi
done

# Kill any remaining python processes with our service names
echo "   Cleaning up any remaining service processes..."
pkill -f "uvicorn.*main:app.*--port.*800[027]" 2>/dev/null || true
pkill -f "mcp_bus" 2>/dev/null || true
pkill -f "scout.*agent" 2>/dev/null || true
pkill -f "memory.*agent" 2>/dev/null || true

# Wait for cleanup
sleep 2

echo "✅ Cleanup complete"
echo ""

# Activate the rapids environment for all services
source /home/adra/miniconda3/etc/profile.d/conda.sh
conda activate rapids-25.06

echo "✅ Environment: rapids-25.06 activated"

# Start MCP Bus (Communication Hub)
echo "📡 Starting MCP Bus..."
cd /home/adra/JustNewsAgentic/mcp_bus
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > mcp_bus.log 2>&1 &
MCP_BUS_PID=$!
echo "✅ MCP Bus started (PID: $MCP_BUS_PID) - Log: mcp_bus/mcp_bus.log"

# Wait for MCP Bus to start with timeout
echo "   Waiting for MCP Bus to respond..."
for i in {1..10}; do
    if curl -s http://localhost:8000/agents > /dev/null 2>&1; then
        echo "   ✅ MCP Bus responding after ${i} seconds"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "   ⚠️ MCP Bus not responding after 10 seconds, continuing anyway..."
    fi
    sleep 1
done

# Start Scout Agent (our main focus)
echo "🕵️ Starting Scout Agent..."
cd /home/adra/JustNewsAgentic/agents/scout
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8002 > scout_agent.log 2>&1 &
SCOUT_PID=$!
echo "✅ Scout Agent started (PID: $SCOUT_PID) - Log: agents/scout/scout_agent.log"

# Wait for Scout Agent to start with timeout
echo "   Waiting for Scout Agent to respond..."
for i in {1..15}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "   ✅ Scout Agent responding after ${i} seconds"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "   ⚠️ Scout Agent not responding after 15 seconds, continuing anyway..."
    fi
    sleep 1
done

# Start Memory Agent (Database Storage)
echo "💾 Starting Memory Agent..."
cd /home/adra/JustNewsAgentic/agents/memory
# Set MCP Bus URL for native deployment
export MCP_BUS_URL="http://localhost:8000"
# Set PostgreSQL connection for native deployment
export POSTGRES_HOST="localhost"
export POSTGRES_DB="justnews"
export POSTGRES_USER="adra"
export POSTGRES_PASSWORD="justnews123"
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8007 > memory_agent.log 2>&1 &
MEMORY_PID=$!
echo "✅ Memory Agent started (PID: $MEMORY_PID) - Log: agents/memory/memory_agent.log"

# Wait for Memory Agent to start with timeout
echo "   Waiting for Memory Agent to respond..."
for i in {1..10}; do
    if curl -s http://localhost:8007/health > /dev/null 2>&1; then
        echo "   ✅ Memory Agent responding after ${i} seconds"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "   ⚠️ Memory Agent not responding after 10 seconds, check logs"
    fi
    sleep 1
done

# Check status
echo ""
echo "🔍 Service Status Check:"
echo "======================================================"

if curl -s http://localhost:8000/agents > /dev/null; then
    echo "✅ MCP Bus: Running (http://localhost:8000)"
else
    echo "❌ MCP Bus: Not responding"
fi

if curl -s http://localhost:8002/health > /dev/null; then
    echo "✅ Scout Agent: Running (http://localhost:8002)"
else
    echo "❌ Scout Agent: Not responding"
fi

if curl -s http://localhost:8007/health > /dev/null; then
    echo "✅ Memory Agent: Running (http://localhost:8007)"
else
    echo "❌ Memory Agent: Not responding"
fi

echo ""
echo "📋 Process Information:"
echo "MCP Bus PID: $MCP_BUS_PID"
echo "Scout Agent PID: $SCOUT_PID"
echo "Memory Agent PID: $MEMORY_PID"
echo ""
echo "📁 Log Files:"
echo "  MCP Bus: /home/adra/JustNewsAgentic/mcp_bus/mcp_bus.log"
echo "  Scout Agent: /home/adra/JustNewsAgentic/agents/scout/scout_agent.log"
echo "  Memory Agent: /home/adra/JustNewsAgentic/agents/memory/memory_agent.log"
echo ""
echo "🎯 Next Steps:"
echo "  1. ✅ Scout Agent is working! Use: python test_proper_news_urls.py"
echo "  2. Test Scout → Memory storage pipeline"
echo "  3. Build complete news processing workflow"
