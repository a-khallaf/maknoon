# CLI Command Reference
> **Technical Specification for the Maknoon Command-Line Interface**

## Executive Summary
The Maknoon CLI provides a unified interface for post-quantum cryptographic operations, identity management, and secure secret storage. Designed for both interactive use and automated integration, the interface supports structured JSON output and follows a deterministic command hierarchy.

---

## Cryptographic Operations
These commands manage the primary data protection pipeline, including hybrid encryption, digital signatures, and metadata analysis.

| Command | Functionality | Key Parameters |
| :--- | :--- | :--- |
| `encrypt` | Protects files/directories using the streaming engine. | `--public-key`, `--sign-key`, `--stealth`, `--compress` |
| `decrypt` | Restores encrypted assets using private identities. | `--private-key`, `--output`, `--overwrite` |
| `info` | Provides deep technical metadata for encrypted files. | `--json` |
| `sign` | Generates a standalone ML-DSA-87 signature. | `--identity`, `--output` |
| `verify` | Validates data integrity and provenance. | `--public-key`, `--signature` |

---

## Identity Management
Maknoon utilizes a decentralized identity model based on NIST-standardized PQC algorithms.

| Command | Functionality | Key Parameters |
| :--- | :--- | :--- |
| `keygen` | Provisions a new hybrid PQC identity pair. | `--profile`, `--output` |
| `identity list` | Enumerates locally managed cryptographic identities. | `--json` |
| `identity publish` | Announces public keys to decentralized registries. | `--nostr`, `--dns`, `--desec` |
| `identity split` | Shards a private identity via Shamir's Secret Sharing. | `--threshold`, `--shares` |
| `identity combine` | Reconstructs a private identity from M-of-N mnemonics. | `--output` |

---

## Secret Vault Operations
The vault provides an authenticated, quantum-resistant storage layer for credentials and sensitive configuration data.

| Command | Functionality | Key Parameters |
| :--- | :--- | :--- |
| `vault set` | Stores a credential with optional username association. | `--user`, `--vault` |
| `vault get` | Retrieves a managed secret from the storage layer. | `--vault` |
| `vault list` | Displays services stored within the active vault. | `--json` |
| `vault split` | Shards the master vault access material. | `--threshold`, `--shares` |
| `vault recover` | Restores vault access using reconstructed material. | `--output` |

---

## Communication and P2P
Secure transport commands facilitate ephemeral data exchange and real-time communication across disparate networks.

| Command | Functionality | Key Parameters |
| :--- | :--- | :--- |
| `send` | Initiates a secure ephemeral P2P file transfer. | `--text`, `--public-key`, `--stealth` |
| `receive` | Ingests data from a remote peer via wormhole code. | `--output`, `--private-key` |
| `chat` | Establishes a real-time, end-to-end encrypted session. | `--jsonl` (for agents) |

---

## System Utilities
Administrative commands for system configuration, capability discovery, and cryptographic profiling.

| Command | Functionality | Key Parameters |
| :--- | :--- | :--- |
| `config` | Manages global security and performance settings. | `list`, `set`, `init` |
| `profiles` | Manages cryptographic parameter sets (ciphers, KDF). | `list`, `gen`, `rm` |
| `schema` | Generates a recursive JSON-Schema of the CLI. | `(No parameters)` |
| `man` | Generates technical manual pages (roff/man format). | `(No parameters)` |

---

## Configuration Management
Maknoon behavior is governed by a centralized configuration that dictates resource allocation and security parameters.

> **Performance Optimization:** Users can configure `perf.concurrency` to align with host CPU resources. The default streaming chunk size is fixed at 64KB to ensure stable memory profiles.

### Security Parameters (Argon2id)
| Key | Default | Description |
| :--- | :--- | :--- |
| `security.time` | `3` | Number of KDF iterations. |
| `security.memory` | `64MB` | Memory allocation for the Argon2id process. |
| `security.threads` | `4` | Parallelism factor for key derivation. |

> **Compliance Notice:** Modifying security parameters may impact compatibility with existing encrypted assets. It is recommended to maintain standardized profiles across organizational deployments.
