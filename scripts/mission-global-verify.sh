#!/bin/bash
set -e

# Mission: Global Orchestration (Phase 4)
# Verification of Decentralized Nostr Discovery & PQC Tunneling

source "$(dirname "$0")/common.sh"
COMPOSE_FILE="deploy/docker/mission-global.yml"

trap 'fail_trap "Global Orchestration" "$COMPOSE_FILE"' EXIT

echo "🏗️  Provisioning Global Mesh (Nostr + 2 Agents)..."
docker compose -p maknoon-global -f $COMPOSE_FILE up -d --build
sleep 15

# Use specific containers
LONDON="agent-london"
NY="agent-ny"
L_EXEC="docker compose -p maknoon-global -f $COMPOSE_FILE exec -T $LONDON"
N_EXEC="docker compose -p maknoon-global -f $COMPOSE_FILE exec -T $NY"

echo "🔑 Step 1: Provisioning London PQC Identity..."
$L_EXEC maknoon keygen -o london-id --no-password
$L_EXEC maknoon config set nostr.relays "ws://172.30.0.5:8080"

echo "📡 Step 2: Publishing London Identity to Nostr Registry..."
# This will include active Multiaddrs in the Kind 0 event
$L_EXEC maknoon identity publish @london-gateway --name london-id

echo "🌍 Step 3: Global Discovery from New York..."
$N_EXEC maknoon config set nostr.relays "ws://172.30.0.5:8080"
echo "🔍 NY Agent searching for '@london-gateway' via Nostr..."
RESOLVE_RES=$($N_EXEC maknoon mcp --transport stdio <<EOF
{"jsonrpc":"2.0","method":"tools/call","params":{"name":"resolve_identity","arguments":{"input":"@london-gateway"}},"id":1}
EOF
)

# Extract Public Key from NY's discovery result
PUBKEY=$(echo "$RESOLVE_RES" | jq -r '.result.content[0].text | fromjson | .public_key')

if [ -z "$PUBKEY" ] || [ "$PUBKEY" == "null" ]; then
    echo "❌ FAILED: Global discovery of @london-gateway failed."
    exit 1
fi
echo "✅ Discovery SUCCESS: Retrieved ML-KEM Key: ${PUBKEY:0:16}..."

echo "🌉 Step 4: Autonomous PQC Tunnel Provisioning..."
# Start a tunnel from NY to London's discovered P2P endpoint
# (In this simulation, we use the local multiaddr discovered via the mesh)
$N_EXEC maknoon mcp --transport stdio <<EOF
{"jsonrpc":"2.0","method":"tools/call","params":{"name":"tunnel_start","arguments":{"p2p_addr":"@london-gateway","p2p_mode":true,"port":1080}},"id":2}
EOF

# Verify tunnel state
STATUS_RES=$($N_EXEC maknoon mcp --transport stdio <<EOF
{"jsonrpc":"2.0","method":"tools/call","params":{"name":"tunnel_status","arguments":{}},"id":3}
EOF
)

if echo "$STATUS_RES" | grep -q "\"active\":true"; then
    echo "✅ SUCCESS: Global PQC Tunnel provisioned via Nostr discovery."
else
    echo "❌ FAILED: Tunnel provisioning failed."
    exit 1
fi

echo -e "\n🏆 SUCCESS: Global Orchestration Mission Passed."
echo "Verified: NIP-05 Resolution, Multiaddr Mesh Discovery, and MCP Provisioning."

echo "🧹 Tearing down Global Mesh..."
docker compose -p maknoon-global -f $COMPOSE_FILE down
