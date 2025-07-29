#!/bin/bash
# Stop All JustNews V4 Services

echo "🛑 Stopping JustNews V4 Services"
echo "================================="

# Kill processes by port
echo "Stopping services by port..."
for port in 8000 8002 8007; do
    PID=$(lsof -t -i:$port 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo "  Killing process on port $port (PID: $PID)"
        kill -15 $PID 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            echo "  Force killing PID $PID"
            kill -9 $PID 2>/dev/null || true
        fi
    else
        echo "  No process found on port $port"
    fi
done

# Kill any remaining service processes
echo "Cleaning up remaining processes..."
pkill -f "uvicorn.*main:app.*--port.*800[027]" 2>/dev/null || true
pkill -f "mcp_bus" 2>/dev/null || true
pkill -f "scout.*agent" 2>/dev/null || true
pkill -f "memory.*agent" 2>/dev/null || true

sleep 1

# Verify cleanup
echo ""
echo "🔍 Verification:"
for port in 8000 8002 8007; do
    PID=$(lsof -t -i:$port 2>/dev/null)
    if [ -z "$PID" ]; then
        echo "  ✅ Port $port is free"
    else
        echo "  ❌ Port $port still occupied by PID $PID"
    fi
done

echo ""
echo "✅ All JustNews services stopped"
