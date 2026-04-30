#!/bin/bash
set -e

# Mission 3: Threshold-Authorized "Dead Man's Switch"
# Verification of Distributed Multi-Sig Governance via P2P.

source "$(dirname "$0")/common.sh"
COMPOSE_FILE="deploy/docker/mission-deadmans.yml"

trap 'fail_trap "Dead Man Switch" "$COMPOSE_FILE"' EXIT

echo "🏗️  Provisioning Dead Man's Switch Infrastructure..."
docker compose -f $COMPOSE_FILE up -d --build

# Wait for nodes to generate IDs
echo "⏳ Waiting for nodes to initialize..."
sleep 5

# Step 1: Initialize SSS Shards on Recovery Node
echo "🎲 Recovery Node: Generating Master Secret and Shards (3-of-4)..."
RECOVERY_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q recovery-node)
MASTER_SECRET="deadbeefdeadbeefdeadbeefdeadbeef"

# Split secret into 4 shards, threshold 3 using the NEW 'identity shard' command
docker exec $RECOVERY_CONTAINER maknoon identity shard "$MASTER_SECRET" -n 4 -m 3 --json > shards.json

# Extract shards
SHARD1=$(jq -r '.shares[0]' shards.json)
SHARD2=$(jq -r '.shares[1]' shards.json)
SHARD3=$(jq -r '.shares[2]' shards.json)
SHARD4=$(jq -r '.shares[3]' shards.json)

# Step 2: Distribute shards to Guardians
echo "📡 Distributing Shards to Guardians..."
for i in 1 2 3 4; do
    eval "S=\$SHARD$i"
    G_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q guardian-$i)
    echo -n "$S" > g$i.shard
    docker cp g$i.shard $G_CONTAINER:/home/maknoon/shard.txt
done

# Step 3: Secure the High-Value Asset on Recovery Node
echo "🔒 Recovery Node: Securing high-value asset in vault..."
docker exec -e MAKNOON_PASSWORD="maknoon-gold-access" $RECOVERY_CONTAINER \
    maknoon vault set "SECRET_SERVICE" --user "ADMIN" --vault "top-secret" --passphrase "$MASTER_SECRET"

# Step 4: Trigger Recovery (3 guardians send shards)
echo "🆘 TRIGGERING RECOVERY: 3 Guardians transmitting shards..."
NET_QUORUM=$(docker network ls --filter name=quorum-net --format "{{.Name}}")
RECOVERY_IP=$(docker inspect -f "{{with index .NetworkSettings.Networks \"$NET_QUORUM\"}}{{.IPAddress}}{{end}}" $RECOVERY_CONTAINER)

for i in 1 2 3; do
    echo "📡 Guardian $i sending shard..."
    docker exec $RECOVERY_CONTAINER sh -c "timeout 30 maknoon receive --p2p --identity recovery-id --trace > shard_recv.log 2>&1" &
    RECV_PID=$!
    
    sleep 3
    RECOVERY_ADDR=$(docker exec $RECOVERY_CONTAINER grep "/ip4/$RECOVERY_IP" shard_recv.log | grep "/tcp/" | tail -n 1 | awk '{print $NF}' | tr -d '\r')
    
    if [ -z "$RECOVERY_ADDR" ]; then echo "❌ FAILED: Could not find Recovery Multiaddr."; exit 1; fi
    
    G_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q guardian-$i)
    docker cp $RECOVERY_CONTAINER:/home/maknoon/.maknoon/keys/recovery-id.kem.pub recovery.pub
    docker cp recovery.pub $G_CONTAINER:/home/maknoon/recovery.pub
    
    docker exec $G_CONTAINER maknoon send shard.txt --to "$RECOVERY_ADDR" --public-key recovery.pub --identity g$i-id --trace
    
    wait $RECV_PID || true
    
    RECEIVED_FILE=$(docker exec $RECOVERY_CONTAINER sh -c "ls shard.txt* | head -n 1")
    if [ -z "$RECEIVED_FILE" ]; then echo "❌ FAILED: Shard $i did not arrive."; exit 1; fi
    docker exec $RECOVERY_CONTAINER mv "$RECEIVED_FILE" shard$i.txt
done

# Step 5: Recovery Node combines shards and unlocks
echo "🧩 Recovery Node: Reconstructing Master Secret..."
RECOV_SHARD1=$(docker exec $RECOVERY_CONTAINER cat shard1.txt)
RECOV_SHARD2=$(docker exec $RECOVERY_CONTAINER cat shard2.txt)
RECOV_SHARD3=$(docker exec $RECOVERY_CONTAINER cat shard3.txt)

# Use the NEW 'identity reconstruct' command
RECOV_MASTER_JSON=$(docker exec $RECOVERY_CONTAINER sh -c "maknoon identity reconstruct \"$RECOV_SHARD1\" \"$RECOV_SHARD2\" \"$RECOV_SHARD3\" --json")
RECOVERED_MASTER=$(echo "$RECOV_MASTER_JSON" | jq -r '.secret' | tr -d '\r' | xargs)

if [ "$MASTER_SECRET" != "$RECOVERED_MASTER" ]; then
    echo "❌ FAILED: Master secret reconstruction failed!"
    echo "Expected: $MASTER_SECRET"
    echo "Recovered: $RECOVERED_MASTER"
    exit 1
fi

echo "🔓 Recovery Node: Unlocking vault with recovered secret..."
FINAL_VAL=$(docker exec $RECOVERY_CONTAINER maknoon vault get "SECRET_SERVICE" --vault "top-secret" --passphrase "$RECOVERED_MASTER" --json | jq -r '.password')

if [ "$FINAL_VAL" == "maknoon-gold-access" ]; then
    echo "✅ SUCCESS: Dead Man's Switch verified! Master secret recovered via 3-of-4 P2P shards."
else
    echo "❌ FAILED: Final secret mismatch."
    echo "Got: $FINAL_VAL"
    exit 1
fi

echo "🧹 Tearing down Quorum..."
docker compose -f $COMPOSE_FILE down
rm shards.json g1.shard g2.shard g3.shard g4.shard recovery.pub
