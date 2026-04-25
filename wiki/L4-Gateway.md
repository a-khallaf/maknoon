# Post-Quantum L4 Tunnel Gateway
> **Programmable User-Space Secure Perimeters**

## Overview
The Maknoon L4 Gateway provides a mechanism for AI Agents and human operators to establish secure, post-quantum network tunnels without administrative privileges. By utilizing a user-space network stack (QUIC) and a SOCKS5 proxy, Maknoon can route application traffic through a post-quantum-secured transport layer.

---

## Technical Architecture

### Component Stack
1.  **SOCKS5 Gateway**: A concurrent proxy listener that accepts local TCP connections.
2.  **QUIC Transport**: A multiplexed transport layer powered by `quic-go`, providing native congestion control and stream-based isolation.
3.  **PQ-TLS 1.3 Handshake**: A secure handshake utilizing **ML-KEM-1024 + X25519** hybrid key exchange.
4.  **Enclave Buffer Pool**: A hardware-locked memory arena (`memguard`) for high-performance, zero-leak packet processing.

### Data Flow (Encapsulation)
When a connection is initiated through the SOCKS5 proxy:
*   The **Gateway** reads the destination address from the SOCKS5 handshake.
*   A new **QUIC Stream** is opened within the existing PQC tunnel.
*   The **Destination Header** (1-byte length + address string) is transmitted as the first payload on the stream.
*   The **Remote Peer** accepts the stream, dials the target address, and begins bi-directional proxying.

---

## Security Posture

### Post-Quantum Resistance
The tunnel handshake is secured via **PQ-TLS 1.3**, prioritizing the `X25519MLKEM768` hybrid curve. This ensures that the network traffic is resilient against future decryption by quantum computers.

### Memory Hygiene
To prevent plaintext network data from leaking into the Go heap (where it could be exposed via swap or cold-boot attacks), Maknoon utilizes a specialized memory pipeline:
*   **Locked Buffers**: All network I/O is performed using `memguard.LockedBuffer` instances.
*   **Deterministic Zeroization**: Every buffer is explicitly wiped using `b.Wipe()` before being returned to the pool, ensuring no residual data persists between streams.

### Zero-Privilege Operation
The entire stack operates in user-space using standard UDP sockets. This allows Maknoon to provision secure network perimeters in restricted environments such as:
*   Enterprise CI/CD runners.
*   Sandboxed Docker containers (`scratch` images).
*   Non-root cloud instances.

---

## AI Agent Integration

### MCP Tools
Agents can dynamically manage the tunnel lifecycle using the following tools:

| Tool | Action | Parameters |
| :--- | :--- | :--- |
| `mcp_maknoon_tunnel_start` | Initializes the tunnel and proxy. | `remote` (string), `port` (int) |
| `mcp_maknoon_tunnel_status` | Retrieves performance and state data. | None |
| `mcp_maknoon_tunnel_stop` | Tears down the tunnel and clears memory. | None |

### Mission Scenario
1.  **Agent** receives a task to fetch data from a restricted internal server.
2.  **Agent** provisions an identity and starts a tunnel to a trusted Maknoon gateway.
3.  **Agent** configures its internal HTTP client to use the local SOCKS5 proxy (`127.0.0.1:1080`).
4.  **Agent** performs the task securely and terminates the tunnel.

---

## Operational Configuration (Viper)
The tunnel is governed by the following environment variables:

*   `MAKNOON_PASSPHRASE`: Master key for tunnel identity unlocking.
*   `MAKNOON_AGENT_MODE`: Enforces path validation and resource clamps.
*   `MAKNOON_PERFORMANCE_CONCURRENCY`: Controls the parallel worker pool for packet processing.
