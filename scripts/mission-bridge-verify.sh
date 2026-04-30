#!/bin/bash
set -e

# Mission 2: Multi-Network PQC Bridge
# Verification of cross-network tunneling via P2P.

source "$(dirname "$0")/common.sh"
COMPOSE_FILE="deploy/docker/mission-bridge.yml"

trap 'fail_trap "Multi-Network Bridge" "$COMPOSE_FILE"' EXIT

echo "🏗️  Provisioning Multi-Network Bridge Infrastructure..."
docker compose -f $COMPOSE_FILE up -d --build

# Wait for nodes to generate IDs
echo "⏳ Waiting for Gateway-B to initialize..."
sleep 5

# Step 1: Discover Gateway-B Multiaddr
echo "📡 Discovering Gateway-B Multiaddr on Public Mesh..."
GATEWAY_B_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q gateway-b)
NET_PUBLIC=$(docker network ls --filter name=public-mesh --format "{{.Name}}")
GATEWAY_B_IP_PUB=$(docker inspect -f "{{with index .NetworkSettings.Networks \"$NET_PUBLIC\"}}{{.IPAddress}}{{end}}" $GATEWAY_B_CONTAINER)

MAX_RETRIES=30
GATEWAY_B_ADDR=""
for i in $(seq 1 $MAX_RETRIES); do
    GATEWAY_B_ADDR=$(docker compose -f $COMPOSE_FILE logs gateway-b | grep "/ip4/$GATEWAY_B_IP_PUB" | grep "/tcp/" | head -n 1 | awk '{print $NF}' | tr -d '\r')
    if [ ! -z "$GATEWAY_B_ADDR" ]; then
        break
    fi
    sleep 1
done

if [ -z "$GATEWAY_B_ADDR" ]; then
    echo "❌ FAILED: Could not find Gateway-B Multiaddr."
    exit 1
fi
echo "🆔 Gateway-B Multiaddr: $GATEWAY_B_ADDR"

# Step 2: Start Gateway-A and connect to Gateway-B
echo "🤝 Gateway-A: Establishing Bridge to Gateway-B..."
GATEWAY_A_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q gateway-a)

# We bind to 0.0.0.0 so the client container can reach us
docker exec -d $GATEWAY_A_CONTAINER sh -c \
    "maknoon keygen --no-password -o gateway-a-id && \
     maknoon tunnel start --p2p --p2p-addr $GATEWAY_B_ADDR --port 1080 --bind 0.0.0.0 --identity gateway-a-id --trace > tunnel.log 2>&1"

# Wait for tunnel to be ready
wait_for_port "$GATEWAY_A_CONTAINER" 1080 || (docker exec $GATEWAY_A_CONTAINER cat tunnel.log && exit 1)

# Step 3: Client queries service via Gateway-A
echo "🧪 Verifying cross-network connectivity via Bridge..."
CLIENT_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q client)

# We use the internal IP of gateway-a on Net-A
NET_A=$(docker network ls --filter name=net-a --format "{{.Name}}")
GATEWAY_A_IP_A=$(docker inspect -f "{{with index .NetworkSettings.Networks \"$NET_A\"}}{{.IPAddress}}{{end}}" $GATEWAY_A_CONTAINER)

# Note: We use --socks5-hostname because 'service:80' only exists in Net-B
# Gateway-B will resolve it.
TEST_RESULT=$(docker exec $CLIENT_CONTAINER \
    curl -s --socks5-hostname $GATEWAY_A_IP_A:1080 http://service:80)

if [[ $TEST_RESULT == *"Hello from Isolated Net-B"* ]]; then
    echo "✅ SUCCESS: Multi-Network Bridge verified! Traffic routed between isolated networks."
else
    echo "❌ FAILED: Unexpected response or connection failure."
    echo "Response: $TEST_RESULT"
    exit 1
fi

echo "🧹 Tearing down Bridge..."
docker compose -f $COMPOSE_FILE down
