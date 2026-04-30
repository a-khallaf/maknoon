#!/bin/bash
set -e

# Mission: Zero-Trust Mesh (Phase 1)
# Verification of Identity-Based Tunneling

COMPOSE_FILE="deploy/docker/mission-mesh.yml"

echo "🏗️  Provisioning Zero-Trust Mesh..."
docker compose -f $COMPOSE_FILE up -d --build

# Wait for Gateway to generate ID and start listening
echo "📡 Waiting for Maknoon Gateway to broadcast Multiaddr..."
MAX_RETRIES=30
GATEWAY_ADDR=""
for i in $(seq 1 $MAX_RETRIES); do
    # Look for a non-loopback Multiaddr in the logs
    GATEWAY_ADDR=$(docker compose -f $COMPOSE_FILE logs maknoon-gateway | grep "/ip4/" | grep -v "127.0.0.1" | head -n 1 | awk '{print $NF}' | tr -d '\r')
    if [ ! -z "$GATEWAY_ADDR" ]; then
        break
    fi
    sleep 1
done

if [ -z "$GATEWAY_ADDR" ]; then
    echo "❌ FAILED: Could not retrieve Gateway Multiaddr."
    docker compose -f $COMPOSE_FILE logs maknoon-gateway
    exit 1
fi

echo "🆔 Gateway Multiaddr Found: $GATEWAY_ADDR"

# Start the Client Tunnel in the background
echo "🤝 Establishing Identity-Bound Tunnel from Client..."
# Get the container ID for maknoon-client
CLIENT_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q maknoon-client)

docker exec -d $CLIENT_CONTAINER sh -c \
    "maknoon keygen --no-password -o client-id && \
     maknoon tunnel start --p2p --p2p-addr $GATEWAY_ADDR --port 1080 --identity client-id > tunnel.log 2>&1"

# Wait for tunnel to be ready
echo "⏳ Waiting for SOCKS5 proxy to listen on port 1080..."
MAX_RETRIES=15
for i in $(seq 1 $MAX_RETRIES); do
    if docker exec $CLIENT_CONTAINER netstat -tln | grep ":1080 " > /dev/null; then
        echo "✅ Tunnel Ready."
        break
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        echo "❌ TIMEOUT: Tunnel failed to start."
        docker exec $CLIENT_CONTAINER cat tunnel.log
        docker compose -f $COMPOSE_FILE down
        exit 1
    fi
    sleep 1
done

# Test Connectivity
echo "🧪 Verifying end-to-end connectivity via SOCKS5..."
# We use 'curl' inside the client node to reach the provider through the proxy
TEST_RESULT=$(docker exec $CLIENT_CONTAINER \
    curl -v -s --socks5-hostname 127.0.0.1:1080 http://provider:80 2>&1)

if [[ $TEST_RESULT == *"Directory listing for /"* ]]; then
    echo "✅ SUCCESS: Zero-Trust Mesh verified! Traffic successfully routed through PQC Tunnel."
else
    echo "❌ FAILED: Unexpected response or connection failure."
    echo "Response: $TEST_RESULT"
    exit 1
fi

echo "🧹 Tearing down Mesh..."
docker compose -f $COMPOSE_FILE down
