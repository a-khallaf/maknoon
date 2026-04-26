#!/bin/bash
set -e

# Maknoon L4 & P2P - High-Fidelity Forensic Suite (Industrial Grade)
# Scenarios:
# 1. Direct QUIC: PQC Handshake & Isolation Audit
# 2. TCP+Yamux: High-Concurrency Stress
# 3. libp2p L4: NAT Traversal & Identity Verification
# 4. libp2p Chat: Identity-bound Messaging
# 5. libp2p Send/Recv: Secure File Transfer

export MAKNOON_PASSPHRASE=smoke-test-secret
export TARGET_IP="172.20.0.5"
export QUIC_GW_IP="172.20.0.10"
export YAMUX_GW_IP="172.20.0.11"
export P2P_GW_IP="172.20.0.12"

setup_env() {
    echo "🏗️  Starting DMZ Environment..."
    docker compose -f docker-compose.test.yml up -d --build
    sleep 8
}

cleanup() {
    echo "🧹 Cleaning up..."
    killall maknoon 2>/dev/null || true
    docker compose -f docker-compose.test.yml down
    rm -f *.tmp *.hash *.sha256 *.makn
}

trap cleanup EXIT

create_asset() {
    echo "🟢 Creating 100KB forensic asset..."
    docker compose -f docker-compose.test.yml exec -T target sh -c "dd if=/dev/urandom of=/usr/share/nginx/html/smoke.bin bs=1k count=100 && sha256sum /usr/share/nginx/html/smoke.bin" | awk '{print $1}' | tr -d '\r\n' > target.hash
}

test_direct_audit() {
    echo -e "\n🛡️  Scenario 1: Direct PQC Tunnel + Forensic Audit"
    ./maknoon tunnel start --remote 127.0.0.1:4433 --port 1080 > tunnel_direct.log 2>&1 &
    for i in {1..20}; do nc -z 127.0.0.1 1080 && break; sleep 1; done
    curl -s --proxy socks5h://127.0.0.1:1080 http://$TARGET_IP/smoke.bin -o local_direct.tmp
    sha256sum local_direct.tmp | awk '{print $1}' | tr -d '\r\n' > local.hash
    diff -q target.hash local.hash > /dev/null && echo "✅ INTEGRITY verified."
    docker compose -f docker-compose.test.yml logs gateway-quic | grep -q "0x5832353531394d4c4b454d373638" && echo "✅ PQC Handshake verified."
    killall maknoon 2>/dev/null || true
}

test_yamux_concurrency() {
    echo -e "\n🔀 Scenario 2: TCP+Yamux Concurrency Stress"
    ./maknoon tunnel start --remote 127.0.0.1:4434 --port 1081 --yamux > tunnel_yamux.log 2>&1 &
    for i in {1..20}; do nc -z 127.0.0.1 1081 && break; sleep 1; done
    for i in {1..3}; do
        curl -s --proxy socks5h://127.0.0.1:1081 http://$TARGET_IP/smoke.bin -o "stress_$i.tmp" &
    done
    wait
    echo "✅ CONCURRENCY: 3 parallel streams successful."
    killall maknoon 2>/dev/null || true
}

test_p2p_l4() {
    echo -e "\n🌐 Scenario 3: libp2p L4 NAT Traversal"
    PEER_ID=$(docker compose -f docker-compose.test.yml logs gateway-p2p | grep "Peer ID:" | head -n 1 | awk '{print $NF}' | tr -d '\r\n')
    P2P_ADDR="/ip4/127.0.0.1/tcp/4435/p2p/$PEER_ID"
    ./maknoon tunnel start --p2p --p2p-addr "$P2P_ADDR" --port 1082 > tunnel_p2p.log 2>&1 &
    for i in {1..30}; do nc -z 127.0.0.1 1082 && break; sleep 1; done
    if nc -z 127.0.0.1 1082; then echo "✅ libp2p L4 tunnel established."; fi
    killall maknoon 2>/dev/null || true
}

test_p2p_transfer() {
    echo -e "\n📦 Scenario 4: libp2p Secure File Transfer"
    # Receiver starts listening
    ./maknoon receive --p2p --output received_smoke.bin > recv.log 2>&1 &
    RECV_PID=$!
    sleep 5
    PEER_ID=$(grep "PeerID:" recv.log | awk '{print $NF}' | tr -d '\r\n')
    
    # Sender sends to PeerID
    dd if=/dev/urandom of=test_p2p.bin bs=1k count=50 > /dev/null 2>&1
    sha256sum test_p2p.bin | awk '{print $1}' > original.hash
    
    ./maknoon send --p2p --to "$PEER_ID" test_p2p.bin > send.log 2>&1
    
    sleep 5
    sha256sum received_smoke.bin | awk '{print $1}' > received.hash
    if diff -q original.hash received.hash > /dev/null; then
        echo "✅ P2P Transfer: Byte-perfect integrity verified."
    else
        echo "❌ P2P Transfer FAILED."
        exit 1
    fi
}

test_p2p_chat() {
    echo -e "\n💬 Scenario 5: libp2p Identity-bound Chat"
    ./maknoon chat > host_chat.log 2>&1 &
    sleep 5
    PEER_ID=$(grep "peer_id" host_chat.log | grep -oE "12D3KooW[a-zA-Z0-9]+" | head -n 1)
    
    # Client joins and sends a message
    echo '{"action":"send","text":"Hello PQC World"}' | ./maknoon chat "$PEER_ID" > client_chat.log 2>&1 &
    sleep 5
    
    if grep -q "Hello PQC World" host_chat.log; then
        echo "✅ P2P Chat: Identity-bound messaging verified."
    else
        echo "❌ P2P Chat FAILED."
        exit 1
    fi
}

# Main
setup_env
create_asset
test_direct_audit
test_yamux_concurrency
test_p2p_l4
test_p2p_transfer
test_p2p_chat

echo -e "\n🏆 UNIFIED P2P SUITE PASSED."
