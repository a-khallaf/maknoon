# Maknoon (مكنون) - Project Context

Maknoon is an industrial-grade, post-quantum CLI encryption engine and Model Context Protocol (MCP) server. It focuses on functional density, cryptographic integrity, and secure AI agent orchestration.

## 🏗 Project Architecture (v4.0 Alpha - Autonomous Orchestration)

- **`Unified Binary`**: A single statically linked binary hosts both the CLI and the native MCP server. Mode of operation is determined by the `mcp` command.
- **`Pure Engine (DI)`**: The central Engine is fully decoupled from the environment via **Dependency Injection**. All storage logic is abstracted into `KeyStore`, `ConfigStore`, and `VaultStore` interfaces.
- **`Presenter Pattern`**: All user-facing output is managed via the `Presenter` interface. Logic layer returns structured `Result` objects; UI layer renders them as pretty tables (CLI) or JSON (MCP/Agent).
- **`Transformer Pipeline`**: Data streaming is organized into a modular pipeline defined in `pkg/crypto/transformer.go`. Pluggable stages (Compressor, Encryptor, Archiver) can be chained dynamically.
- **`Dual-Transport MCP`**: Supports local `stdio` and remote `sse`. Remote sessions are secured via **Post-Quantum TLS 1.3** (prioritizing ML-KEM).
- **`Container Sandbox`**: Multi-stage `scratch` build (~13MB) with zero OS attack surface. Runs as a non-privileged user (`1000:1000`).

## 🛡 Cryptographic Stack

- **Symmetric Cipher**: XChaCha20-Poly1305 (AEAD) with 192-bit nonces.
- **Asymmetric Encryption (KEM)**: ML-KEM / Kyber1024 (NIST Standard) wrapped in standard **HPKE Seal/Open** (RFC 9180).
- **Digital Signatures**: ML-DSA-87 / Dilithium (NIST Standard).
- **Key Derivation (KDF)**: Argon2id (Standard: 3 iterations, 64MB memory).
- **Transport Security**: TLS 1.3 with native X25519MLKEM768 hybrid key exchange.

## 🚀 P2P & Identity Lessons

- **Identity Collision**: Never use `libp2p.FallbackDefaults` when providing a custom identity. This triggers a "cannot specify multiple identities" error.
- **Explicit Identity**: All P2P operations (`send`, `receive`, `chat`) support explicit identity selection via the `--identity` flag.
- **Transport Agnosticism**: The Maknoon P2P Wire Protocol (defined in `p2p_message.go`) is isolated from the `libp2p` transport.
- **MCP-over-SSE**: Tool responses are pushed through the long-lived SSE stream (`/sse`), not the POST body.

## 🏗 Mission & Docker Infrastructure Lessons

- **ENTRYPOINT vs. COMMAND Conflict**: When using `ENTRYPOINT ["maknoon"]` in a Dockerfile, Docker Compose `command: ["sh", "-c", "..."]` passes `sh` as an argument to `maknoon`, leading to errors. For mission-ready images, use a bare image and define the full execution logic in the Compose file or use a shell-based `ENTRYPOINT`.
- **Volume Permission Shadowing**: Mounted volumes often default to root ownership. Use the `su-exec` pattern: start as `root`, `chown` the mount point, and then drop privileges using `su-exec maknoon ...`.
- **Shell Quoting in YAML**: Avoid double-quoting shell command blocks in YAML (e.g., `command: "sh -c '...'"`). Use the literal block scalar `>` or a simple string to prevent argument misparsing.
-   **Verification Robustness**: Integration scripts MUST implement explicit timeouts and log capturing for failing services to prevent infinite "wait" loops in CI.
-   **Test Environmental Isolation**: Unit tests that interact with the filesystem (Vaults, Config) MUST override the `HOME` environment variable and call `commands.ResetGlobalConfig()` to ensure a clean state and prevent contamination from the developer's real environment.


## 🤖 Agent Sandbox & Governance

1.  **Logical Isolation**: `AgentPolicy` restricts the engine to the user's workspace and temp directories.
2.  **Physical Isolation**: Containerized deployment removes shells and utilities.
3.  **Governance**: All operations are logged with masked metadata via the `AuditEngine` decorator using the `ConsoleAuditLogger` (verbose) or `JSONFileLogger` (audit).

## 📋 Engineering & Documentation Standards

### 1. The Skeptical Engineering Persona
- **Empirical Rigor**: Never assume a feature works just because it compiles or passed a shallow test. Demand high-fidelity E2E verification for all critical paths.
- **Dependency Suspicion**: Treat all third-party libraries (even core ones like libp2p) as potential sources of bloat, complexity, and failure.
- **Proof of Failure**: Before applying a fix, you MUST empirically reproduce the failure.

### 2. The Engine Pattern
All business logic must be invoked via the `Engine` struct. UI layers (CLI/MCP) must remain strictly as controllers. **Mandatory DI**: New services must accept their dependencies in the constructor.

### 3. UI-Agnostic Design (Presenter)
NEVER use `fmt.Print` or `json.Marshal` directly in business logic. Use the `Presenter` interface to maintain consistency across CLI and Agent modes.

### 4. Testing Mandates
- **Universal Missions**: All integration tools must be verified by a transport-agnostic mission suite.
- **Isolation**: Use `testing.Short()` to skip network-dependent tests. 
- **Integrity**: Every new feature requires a functional smoke test and a policy-violation test.

## 🛠 Building and Running

### Key Commands (Makefile)
- **Build**: `make build` (Produces optimized 13MB stripped binary)
- **Test**: `make test` (Runs industry-standard fast suite)
- **Docker**: `make docker-build` (Generates OCI-compliant secure sandbox)

## 🧪 Current Status
- **Architecture**: V4.0 Alpha (DI & Presenter complete).
- **Testing**: Passed all 45+ cases and high-fidelity P2P smoke test.
- **Security**: Post-Quantum TLS 1.3 verified.
