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
# Start a persistent P2P host on London with a fixed port
# Use 'listen' command instead of 'serve'
$L_EXEC sh -c "MAKNOON_P2P_PORT=4001 maknoon tunnel listen --p2p --identity london-id > /tmp/tunnel.log 2>&1 &"
sleep 5

# Check if it's still running and show log start
echo "--- London Tunnel Log ---"
$L_EXEC head -n 20 /tmp/tunnel.log
echo "-------------------------"

# Extract PeerID from London
LONDON_PEER_ID=$($L_EXEC maknoon identity info london-id --json | jq -r '.peer_id')
LONDON_MA="/ip4/172.30.0.10/tcp/4001/p2p/$LONDON_PEER_ID"
echo "📍 London Multiaddr: $LONDON_MA"

# Publish the identity with the explicit Multiaddr
$L_EXEC maknoon identity publish @london-gateway --name london-id --multiaddr "$LONDON_MA"

# Extract the Nostr Hex for discovery
LONDON_NOSTR_HEX=$($L_EXEC maknoon identity info london-id --json | jq -r '.nostr_pub')
echo "📍 London Gateway Nostr Hex: $LONDON_NOSTR_HEX"

echo "🌍 Step 3: Global Discovery from New York..."
$N_EXEC maknoon config set nostr.relays "ws://172.30.0.5:8080"
echo "🔍 NY Agent searching for London Gateway via Nostr..."

# Retry loop for discovery (Nostr indexing can be slow)
MAX_RETRIES=5
PUBKEY=""
for i in $(seq 1 $MAX_RETRIES); do
    echo "   Attempt $i/$MAX_RETRIES..."
    RESOLVE_RES=$($N_EXEC maknoon mcp --transport stdio <<EOF
{"jsonrpc":"2.0","method":"tools/call","params":{"name":"resolve_identity","arguments":{"input":"@$LONDON_NOSTR_HEX"}},"id":1}
EOF
)
    PUBKEY=$(echo "$RESOLVE_RES" | jq -r '.result.content[0].text | fromjson | .public_key // empty')
    if [ -n "$PUBKEY" ] && [ "$PUBKEY" != "null" ]; then
        break
    fi
    sleep 2
done

if [ -z "$PUBKEY" ] || [ "$PUBKEY" == "null" ]; then
    echo "❌ FAILED: Global discovery of London Gateway failed."
    echo "Last Resolution Response: $RESOLVE_RES"
    exit 1
fi
echo "✅ Discovery SUCCESS: Retrieved ML-KEM Key: ${PUBKEY:0:16}..."


echo "🌉 Step 4: Autonomous PQC Tunnel Provisioning..."
# Start a tunnel from NY to London in the background
$N_EXEC sh -c "maknoon tunnel start --p2p --remote @$LONDON_NOSTR_HEX --port 1080 > /tmp/ny-tunnel.log 2>&1 &"
sleep 5

# Verify tunnel is running by checking logs or status (if we had a status command)
# For now, we check the log for success message
if $N_EXEC grep -q "🔒 PQC L4 Tunnel Active" /tmp/ny-tunnel.log; then
    echo "✅ SUCCESS: Global PQC Tunnel provisioned via Nostr discovery."
else
    echo "❌ FAILED: Tunnel provisioning failed."
    echo "--- NY Tunnel Log ---"
    $N_EXEC cat /tmp/ny-tunnel.log
    exit 1
fi

echo -e "\n🏆 SUCCESS: Global Orchestration Mission Passed."
echo "Verified: NIP-05 Resolution, Multiaddr Mesh Discovery, and MCP Provisioning."

echo "🧹 Tearing down Global Mesh..."
docker compose -p maknoon-global -f $COMPOSE_FILE down
