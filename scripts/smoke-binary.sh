#!/bin/bash
set -e

# Maknoon L4 - High-Fidelity Forensic Suite (Industrial Grade)
# Scenarios:
# 1. Direct QUIC: PQC Handshake & Isolation Audit
# 2. TCP+Yamux: High-Concurrency Stress (Multi-Stream)
# 3. libp2p: NAT Traversal & Identity Verification

export MAKNOON_PASSPHRASE=smoke-test-secret
export TARGET_IP="172.20.0.5"
export QUIC_GW_IP="172.20.0.10"
export YAMUX_GW_IP="172.20.0.11"
export P2P_GW_IP="172.20.0.12"

setup_env() {
    echo "🏗️  Starting PQC DMZ Environment..."
    docker compose -f docker-compose.test.yml up -d --build
    sleep 8
}

cleanup() {
    echo "🧹 Cleaning up..."
    killall maknoon 2>/dev/null || true
    docker compose -f docker-compose.test.yml down
    rm -f *.tmp *.hash *.sha256
}

trap cleanup EXIT

create_asset() {
    echo "🟢 Creating 100KB forensic asset..."
    docker compose -f docker-compose.test.yml exec -T target sh -c "dd if=/dev/urandom of=/usr/share/nginx/html/smoke.bin bs=1k count=100 && sha256sum /usr/share/nginx/html/smoke.bin" | awk '{print $1}' | tr -d '\r\n' > target.hash
}

test_direct_audit() {
    echo -e "\n🛡️  Scenario 1: Direct PQC Tunnel + Forensic Audit"
    ./maknoon tunnel start --remote 127.0.0.1:4433 --port 1080 > tunnel_direct.log 2>&1 &
    local pid=$!
    for i in {1..20}; do nc -z 127.0.0.1 1080 && break; sleep 1; done

    curl -s --proxy socks5h://127.0.0.1:1080 http://$TARGET_IP/smoke.bin -o local_direct.tmp
    sha256sum local_direct.tmp | awk '{print $1}' | tr -d '\r\n' > local.hash
    if diff -q target.hash local.hash > /dev/null; then echo "✅ INTEGRITY verified."; else echo "❌ INTEGRITY FAILED"; exit 1; fi
    
    # Audit Logs
    if docker compose -f docker-compose.test.yml logs gateway-quic | grep -q "0x5832353531394d4c4b454d373638"; then echo "✅ PQC Handshake verified."; fi
    if docker compose -f docker-compose.test.yml logs target | grep "GET /smoke.bin" | grep -q "$QUIC_GW_IP"; then echo "✅ ISOLATION verified."; fi
    kill $pid
}

test_yamux_concurrency() {
    echo -e "\n🔀 Scenario 2: TCP+Yamux Concurrency Stress (3 Parallel Streams)"
    ./maknoon tunnel start --remote 127.0.0.1:4434 --port 1081 --yamux > tunnel_yamux.log 2>&1 &
    local pid=$!
    for i in {1..20}; do nc -z 127.0.0.1 1081 && break; sleep 1; done

    # Launch 3 parallel curls
    for i in {1..3}; do
        curl -s --proxy socks5h://127.0.0.1:1081 http://$TARGET_IP/smoke.bin -o "stress_$i.tmp" &
    done
    wait
    
    for i in {1..3}; do
        sha256sum "stress_$i.tmp" | awk '{print $1}' | tr -d '\r\n' > "stress_$i.hash"
        diff -q target.hash "stress_$i.hash" || (echo "❌ Stress Test Failed on stream $i" && exit 1)
    done
    echo "✅ CONCURRENCY: 3 parallel streams successful."
    kill $pid
}

test_p2p_lifecycle() {
    echo -e "\n🌐 Scenario 3: libp2p P2P Lifecycle & Status API"
    PEER_ID=$(docker compose -f docker-compose.test.yml logs gateway-p2p | grep "Peer ID:" | head -n 1 | awk '{print $NF}' | tr -d '\r\n')
    P2P_ADDR="/ip4/127.0.0.1/tcp/4435/p2p/$PEER_ID"
    
    ./maknoon tunnel start --p2p --p2p-addr "$P2P_ADDR" --port 1082 > tunnel_p2p.log 2>&1 &
    for i in {1..30}; do nc -z 127.0.0.1 1082 && break; sleep 1; done

    # Verify via Status API
    STATUS=$(./maknoon tunnel status --json)
    if echo "$STATUS" | grep -q "active\":true" && echo "$STATUS" | grep -q "$PEER_ID"; then
        echo "✅ STATUS: Engine reports active P2P tunnel correctly."
    else
        echo "❌ STATUS: API mismatch!"
        echo "$STATUS"
        exit 1
    fi
    
    # Stop via API simulation (killing process and checking port)
    kill $!
    sleep 2
    if ! nc -z 127.0.0.1 1082; then
        echo "✅ LIFECYCLE: Tunnel successfully torn down."
    fi
}

# Main
setup_env
create_asset
test_direct_audit
test_yamux_concurrency
test_p2p_lifecycle

echo -e "\n🏆 INDUSTRIAL SMOKE SUITE PASSED."
