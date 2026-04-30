#!/bin/bash
set -e

# Mission 1: The "Blind" Cryptographic Proxy
# Verification of Double-Ciphertext hygiene and secure routing.

source "$(dirname "$0")/common.sh"
COMPOSE_FILE="deploy/docker/mission-blind-proxy.yml"

trap 'fail_trap "Blind Cryptographic Proxy" "$COMPOSE_FILE"' EXIT

echo "🏗️  Provisioning Blind Proxy Infrastructure..."
docker compose -f $COMPOSE_FILE up -d --build

# Wait for nodes to generate IDs
echo "⏳ Waiting for nodes to initialize..."
sleep 5

# Step 1: Discover Public Keys
echo "🔑 Discovering Public Keys..."
SINK_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q sink)
RELAY_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q relay)
PRODUCER_CONTAINER=$(docker compose -f $COMPOSE_FILE ps -q producer)

docker cp $SINK_CONTAINER:/home/maknoon/.maknoon/keys/sink-id.kem.pub sink.pub
docker cp $RELAY_CONTAINER:/home/maknoon/.maknoon/keys/relay-id.kem.pub relay.pub

# Step 2: Producer generates and encrypts L1 for Sink
echo "🚀 Producer: Generating and Encrypting Layer 1 (for Sink)..."
docker cp sink.pub $PRODUCER_CONTAINER:/home/maknoon/sink.pub
docker exec $PRODUCER_CONTAINER sh -c "dd if=/dev/urandom of=data.bin bs=1M count=5"
ORIG_HASH=$(docker exec $PRODUCER_CONTAINER sha256sum data.bin | awk '{print $1}')

# Encrypt for Sink (Layer 1)
docker exec $PRODUCER_CONTAINER maknoon encrypt data.bin -o L1.makn --public-key sink.pub --trace

# Step 3: Producer sends L1 to Relay (encrypted for Relay)
echo "📡 Producer -> Relay: Transmitting L1..."
docker cp relay.pub $PRODUCER_CONTAINER:/home/maknoon/relay.pub

NET_PRODUCER=$(docker network ls --filter name=net-producer --format "{{.Name}}")
RELAY_IP_PROD=$(docker inspect -f "{{with index .NetworkSettings.Networks \"$NET_PRODUCER\"}}{{.IPAddress}}{{end}}" $RELAY_CONTAINER)
RELAY_ADDR=$(docker compose -f $COMPOSE_FILE logs relay | grep "/ip4/$RELAY_IP_PROD" | grep "/tcp/" | head -n 1 | awk '{print $NF}' | tr -d '\r')

echo "🆔 Relay Multiaddr: $RELAY_ADDR"

# Send L1 to Relay
docker exec $PRODUCER_CONTAINER maknoon send L1.makn --to "$RELAY_ADDR" --public-key relay.pub --identity producer-id --trace

# Step 4: Relay receives L1, Encrypts as L2, and sends to Sink
echo "⏳ Waiting for Relay to receive L1..."
MAX_RETRIES=30
COUNT=0
while ! docker exec $RELAY_CONTAINER find /home/maknoon -name "L1.makn" >/dev/null 2>&1; do
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ TIMEOUT: Relay failed to receive L1."
        docker exec $RELAY_CONTAINER ls -R /home/maknoon
        exit 1
    fi
    sleep 1
    COUNT=$((COUNT + 1))
done
RELAY_L1_PATH=$(docker exec $RELAY_CONTAINER find /home/maknoon -name "L1.makn" | head -n 1)

echo "🔄 Relay: Wrapping in Layer 2 (Blind Encryption)..."
docker cp sink.pub $RELAY_CONTAINER:/home/maknoon/sink.pub
docker exec $RELAY_CONTAINER maknoon encrypt "$RELAY_L1_PATH" -o L2.makn --public-key sink.pub --trace

echo "📡 Relay -> Sink: Transmitting L2..."
NET_SINK=$(docker network ls --filter name=net-sink --format "{{.Name}}")
SINK_IP_SINK=$(docker inspect -f "{{with index .NetworkSettings.Networks \"$NET_SINK\"}}{{.IPAddress}}{{end}}" $SINK_CONTAINER)
SINK_ADDR=$(docker compose -f $COMPOSE_FILE logs sink | grep "/ip4/$SINK_IP_SINK" | grep "/tcp/" | head -n 1 | awk '{print $NF}' | tr -d '\r')
echo "🆔 Sink Multiaddr: $SINK_ADDR"

# Send L2 to Sink
docker exec $RELAY_CONTAINER maknoon send L2.makn --to "$SINK_ADDR" --public-key sink.pub --identity relay-id --trace

# Step 5: Sink receives and decrypts twice
echo "⏳ Waiting for Sink to receive L2..."
COUNT=0
while ! docker exec $SINK_CONTAINER find /home/maknoon -name "L2.makn" >/dev/null 2>&1; do
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ TIMEOUT: Sink failed to receive L2."
        docker exec $SINK_CONTAINER ls -R /home/maknoon
        exit 1
    fi
    sleep 1
    COUNT=$((COUNT + 1))
done
SINK_L2_PATH=$(docker exec $SINK_CONTAINER find /home/maknoon -name "L2.makn" | head -n 1)

echo "🔓 Sink: Decrypting Layer 2..."
# Use path to private key
docker exec $SINK_CONTAINER maknoon decrypt "$SINK_L2_PATH" -o L1_recovered.makn -k sink-id.kem.key --overwrite --trace

echo "🔓 Sink: Decrypting Layer 1..."
docker exec $SINK_CONTAINER maknoon decrypt L1_recovered.makn -o recovered.bin -k sink-id.kem.key --overwrite --trace

# Step 6: Verification
echo "🧪 Verifying E2E Integrity..."
RECOV_HASH=$(docker exec $SINK_CONTAINER sha256sum recovered.bin | awk '{print $1}')

if [ "$ORIG_HASH" == "$RECOV_HASH" ]; then
    echo "✅ SUCCESS: Blind Proxy verified! Original data recovered through double-encryption."
else
    echo "❌ FAILED: Data corruption."
    echo "Original: $ORIG_HASH"
    echo "Recovered: $RECOV_HASH"
    exit 1
fi

echo "🛡️  Verifying Relay Zero-Knowledge State..."
if docker exec $RELAY_CONTAINER find /home/maknoon -name "recovered.bin" | grep "." >/dev/null 2>&1; then
    echo "❌ FAILED: Relay has access to raw data!"
    exit 1
fi
echo "✅ Relay Integrity: No unauthorized decryptions detected."

echo "🧹 Tearing down Blind Proxy..."
docker compose -f $COMPOSE_FILE down
rm sink.pub relay.pub
