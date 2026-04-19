# Maknoon (مكنون) - Project Context

Maknoon is a high-performance, post-quantum CLI encryption tool. It focuses on efficiency, security, and future-proofing against quantum computing threats.

## 🏗 Project Architecture

- **`cmd/maknoon/`**: Entry point (`main.go`) and CLI command definitions using Cobra. CLI commands should remain "thin," delegating logic to the service layer.
- **`pkg/crypto/`**: Core library implementing the cryptographic pipeline, streaming logic, and FIDO2 integration.
- **`integrations/`**: Third-party wrappers and tools (e.g., Python/LangChain, MCP Server).

## 🛡 Cryptographic Stack

- **Symmetric Cipher**: XChaCha20-Poly1305 (AEAD) with 192-bit nonces.
- **Asymmetric Encryption (KEM)**: ML-KEM / Kyber1024 (NIST Standard).
- **Digital Signatures**: ML-DSA-87 / Dilithium (NIST Standard).
- **Key Derivation (KDF)**: Argon2id (Time: 3, Memory: 64MB).
- **P2P Transport**: Magic Wormhole (SPAKE2 PAKE) layered with Maknoon Symmetric PQC. Supports **Identity-Based** (Asymmetric) handshakes and **Zero-Disk** text transport.

## 📋 Engineering Standards & Design Patterns

### 1. Service Layer Pattern
Core business logic (handshakes, multi-stage pipelines) MUST reside in `pkg/crypto` and be exposed as reusable service functions.
- **`crypto.Protect`**: Orchestrates "Archive -> Compress -> Encrypt."
- **`crypto.Unprotect`**: Orchestrates "Decrypt -> Decompress -> Extract."
- **Goal**: Allow the core library to be used by TUIs, APIs, or Agents without CLI dependency.

### 2. Context Encapsulation
Avoid global variables for execution state. Use the `commands.Context` struct to manage `JSONOutput`, `JSONWriter`, and other session-specific states.
- Always use `commands.SetJSONOutput(bool)` to keep state synchronized.

### 3. Centralized Security Validation
All file system operations MUST be validated using `crypto.ValidatePath(path, restricted)`.
- **Restricted Mode**: (Triggered in Agent/JSON mode) Limits access to the user's Home and System Temp directories to prevent path traversal.

### 4. Identity Management API
Use the `crypto.IdentityManager` struct for key discovery and resolution. Avoid manual path concatenation in CLI commands.

### 5. Memory Hygiene
Use `crypto.SafeClear` (aliased to `memguard.WipeBytes`) immediately after sensitive data use.
- **Critical**: Go's GC does not guarantee immediate erasure; deterministic wiping is mandatory for FEKs and passphrases.

### 6. Streaming & Pipes
All cryptographic operations MUST support `io.Reader` and `io.Writer` to allow processing of files larger than available RAM.

## 🛠 Building and Running

### Prerequisites
- Go 1.25 or higher.
- `uv` for Python integration tasks.

### Key Commands
- **Build**: `go build -o maknoon ./cmd/maknoon`
- **Test**: `go test ./...`
- **Python Tests**: `uv run python3 integrations/langchain/test_maknoon_agent_tool.py`

## 🚀 Development Workflow

1.  **Dedicated Branching**: ALWAYS create a new feature branch (`feat/...`, `fix/...`) BEFORE committing.
2.  **Pre-Push Requirements**:
    *   **Update Documentation**: Sync `README.md`, `maknoon.1`, and `wiki/`.
    *   **Verify Tests**: Ensure 100% pass rate on all Go and Python integration tests.
    *   **Formatting**: Run `gofmt -w .` project-wide.
3.  **Mandatory Pull Requests**: Direct commits to `main` are restricted. Changes must be verified via CI/CD before merging.

## 🚀 Pre-Release Checklist

1.  **Security Audit**:
    *   Verify **Path Traversal** protection in `validatePath`.
    *   Confirm **Zip Slip** protection in `ExtractArchive`.
    *   Ensure **JSON Redirection**: In `stdout` decryption, success metadata MUST go to `stderr`.
2.  **Quality Check**:
    *   Run `staticcheck ./...` and `go vet ./...`.
    *   Verify CI/CD success at [Actions](https://github.com/al-Zamakhshari/maknoon/actions).

## 🧪 Testing Practices

- **Unit Tests**: Alongside source in `*_test.go`.
- **Integration Tests**: End-to-end scenarios in `cmd/maknoon/main_test.go` and `integrations/`.
- **P2P Verification**: Use `pkg/crypto/p2p_test.go` to verify race-free header handling, zero-disk text transfers, and asymmetric identity-based handshakes.
