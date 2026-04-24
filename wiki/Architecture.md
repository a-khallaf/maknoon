# Architectural Specification
> **Streaming-First Post-Quantum Cryptographic Engine**

## Executive Summary
Maknoon is architected as a high-performance, constant-memory cryptographic engine designed to mitigate both classical and quantum computational risks. The system utilizes a modular streaming pipeline that decouples I/O operations from cryptographic transformations, ensuring linear scalability and predictable resource utilization across diverse hardware environments.

---

## Core Streaming Pipeline
The architecture centers on a **Parallel Sequencer Model** that processes data in discrete segments, allowing for high-throughput operations without the memory overhead associated with traditional buffer-and-process models.

| Component | Technical Function |
| :--- | :--- |
| **I/O Reader** | Ingests data in 64KB atomic blocks to maintain $O(1)$ memory complexity. |
| **Worker Pool** | Executes cryptographic transformations in parallel across available CPU cores. |
| **Sequencer** | Reassembles processed segments in deterministic order, ensuring strict file integrity. |
| **Transformer Middleware** | Modular layer for on-the-fly archival (TAR), compression, and encryption. |

---

## Cryptographic Design
Maknoon implements a hybrid cryptographic stack that combines NIST-standardized lattice-based algorithms with battle-tested elliptic curve cryptography.

### Hybrid Key Encapsulation (HPKE)
The system utilizes **HPKE (RFC 9180)** to wrap File Encryption Keys (FEKs). This implementation employs a composite KEM:
*   **Lattice Component**: ML-KEM-1024 (Kyber) providing quantum resistance.
*   **Elliptic Curve Component**: X25519 for classical security and performance.

### Context-Aware Security
All cryptographic operations are bound to the file's metadata via the HPKE `info` parameter. This binding includes `ProfileID` and `Header Flags`, effectively mitigating "Recipient Transplantation" and metadata-tampering attacks.

> **Security Compliance:** The engine enforces strict memory hygiene. All sensitive buffers are zeroed out using `SafeClear` patterns immediately after use to prevent leakage via memory-scraping or swap-file persistence.

---

## Directory and Stream Processing
Maknoon processes directory structures as continuous streams rather than discrete file operations.

*   **Internal TAR Encoding**: Directory trees are converted to a TAR stream on-the-fly within the pipeline.
*   **Zero-Disk Footprint**: Archival and encryption occur in a single pass, eliminating the need for temporary storage or intermediate archive files.
*   **Indistinguishability**: In "Stealth Mode," header magic bytes are omitted, rendering the output cryptographically indistinguishable from high-entropy random noise.

---

## Identity and Discovery
The architecture supports a decentralized identity layer designed for serverless environments.

| Feature | Specification |
| :--- | :--- |
| **Identity Registry** | Abstract interface supporting local petnames and decentralized discovery. |
| **Signature Scheme** | ML-DSA-87 (Dilithium) for all self-signed identity records. |
| **Nostr Integration** | Utilizes the global Nostr relay network for public key discovery and transport. |
| **Bridge Design** | Pluggable registry factory allowing integration with LDAP, DNS, or private keybases. |

---

## Technical Philosophy
Maknoon adheres to a "Modern Unix" philosophy, balancing traditional modularity with contemporary requirements for automation and quantum security.

### 1. Atomic Composition
Every core function within `pkg/crypto` is designed to accept `io.Reader` and `io.Writer` interfaces. This ensures that the engine can be integrated into larger pipelines while maintaining a 64KB RAM window for files of any scale.

### 2. Structured Orchestration
While maintaining CLI utility, Maknoon prioritizes **JSON-based observability**. The `maknoon schema` command provides machine-readable capability descriptions, enabling seamless integration with AI agents and CI/CD pipelines.

### 3. Scalable Security
Acknowledging the increased size of PQC payloads (~5KB for keys and signatures), the architecture prioritizes cryptographic robustness over minimal byte-count, ensuring long-term viability against advancing computational threats.
