#!/bin/bash
set -e

# Maknoon L4 Gateway - High-Fidelity Smoke Test
# This script executes a 3-layer forensic audit:
# 1. Byte-perfect data integrity (SHA256)
# 2. Cryptographic PQC enforcement (ML-KEM)
# 3. Network isolation (Source-IP attribution)

echo "🏗️  Starting PQC DMZ Environment..."
docker compose -f docker-compose.test.yml up -d --build
trap "docker compose -f docker-compose.test.yml down; kill -9 \$TUNNEL_PID 2>/dev/null || true; rm -f *.tmp *.hash *.sha256" EXIT

sleep 5

echo "🟢 Step 1: Creating 20MB forensic asset..."
docker compose -f docker-compose.test.yml exec -T target sh -c "dd if=/dev/urandom of=/usr/share/nginx/html/smoke.bin bs=1M count=20 && sha256sum /usr/share/nginx/html/smoke.bin" | awk '{print $1}' | tr -d '\r\n' > target.hash
TARGET_HASH=$(cat target.hash)

echo "🔒 Step 2: Initializing PQC Tunnel..."
export MAKNOON_PASSPHRASE=smoke-test-secret
./maknoon tunnel start --remote 127.0.0.1:4433 --port 1080 > tunnel_smoke.log 2>&1 &
TUNNEL_PID=$!

# Wait for local SOCKS5 gateway
for i in {1..20}; do nc -z 127.0.0.1 1080 && break; sleep 1; done

echo "🔵 Step 3: Executing PQC-Secured Transfer (20MB)..."
curl -s --proxy socks5h://127.0.0.1:1080 http://172.20.0.5/smoke.bin -o local_smoke.tmp

echo "📊 Step 4: Integrity Verification..."
sha256sum local_smoke.tmp | awk '{print $1}' | tr -d '\r\n' > local.hash
if diff -q target.hash local.hash > /dev/null; then
    echo "✅ INTEGRITY: Byte-perfect transfer confirmed."
else
    echo "❌ INTEGRITY: SHA256 mismatch detected!"
    exit 1
fi

echo "🛡️  Step 5: Cryptographic Enforcement Audit..."
# Check logs for X25519MLKEM768 (Curve ID in hex string form from our server diagnostics)
if docker compose -f docker-compose.test.yml logs gateway | grep -q "curve_id=0x5832353531394d4c4b454d373638"; then
    echo "✅ CRYPTO: Strict PQC Handshake (ML-KEM-768) verified."
else
    echo "❌ CRYPTO: PQC Enforcement log not found!"
    exit 1
fi

echo "🌍 Step 6: Network Isolation Audit..."
if docker compose -f docker-compose.test.yml logs target | grep "GET /smoke.bin" | grep -q "172.20.0.10"; then
    echo "✅ ISOLATION: Host identity hidden. Gateway attribution confirmed."
else
    echo "❌ ISOLATION: Source-IP attribution failed!"
    exit 1
fi

echo -e "\n🏆 SMOKE TEST PASSED: Maknoon L4 Gateway is Mission-Ready."
