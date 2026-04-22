# Roadmap: Post-Quantum Cryptographic Hardening
*Derived from the Architectural and Security Analysis of the Maknoon Suite (April 2026)*

## 🎯 Executive Summary
This roadmap outlines the strategic remediation and hardening steps required to transition Maknoon from a robust prototype into an enterprise-grade, zero-trust security suite. The focus is on securing the AI-Agent boundary, enhancing hardware integration, and mitigating side-channel risks.

---

## 🏗 Strategic Remediation Matrix

| Domain | Engineering Enhancement | Threat Mitigation | Priority |
| :--- | :--- | :--- | :--- |
| **Autonomous AI** | Human-in-the-Loop (HITL) for MCP | Prevents prompt-injection-based exfiltration. | **Critical** |
| **Supply Chain** | SLSA Level 4 & Reproducible Builds | Mitigates repository/dependency compromise. | **Critical** |
| **Hardware** | TPM 2.0 & PKCS#11 Offloading | Keys never touch host OS memory. | **High** |
| **Agility** | SPHINCS+ & Classic McEliece | Hedge against lattice-based cryptanalysis. | **High** |
| **OPSEC** | Decoy Vaults (Plausible Deniability) | Defense against physical coercion. | **Medium** |
| **Side-Channel** | Constant-Time CI/CD Testing | Prevents compiler-induced timing leaks. | **Medium** |

---

## 🛠 Detailed Implementation Workstreams

### 1. Hardening the MCP Boundary (AI Security)
- **HITL Authorization:** Modify the MCP server to require manual approval or a hardware token tap for `decrypt_file`, `vault_get`, and `identity_active`.
- **Taint Tracking:** Flag data from untrusted sources in memory to prevent its use in sensitive execution paths.
- **Network Gapping:** In `MAKNOON_AGENT_MODE`, strip the process of network sockets to prevent data exfiltration.

### 2. Side-Channel & Execution Verification
- **Binary Instrumentation:** Use Valgrind/specialized tools in CI/CD to verify constant-time execution of ML-KEM and ML-DSA.
- **Fault Countermeasures:** Implement self-verification of signatures (verify before output) to detect transient hardware faults.

### 3. Supply Chain Integrity
- **SLSA Level 4:** Generate unforgeable provenance attestations for all releases.
- **Deterministic Builds:** Ensure SHA256 hashes match regardless of build environment.
- **Dependency Pinning:** Audit and pin all Go modules to specific commit hashes.

### 4. Hardware Security Integration
- **PKCS#11 Support:** Offload KEM/SIG operations to YubiKey/Nitrokey tokens.
- **TPM 2.0 Sealing:** Bind vault passphrases to Platform Configuration Registers (PCRs) to detect boot-level tampering.

### 5. Advanced Cryptographic Agility
- [x] **Algorithm Diversity:** Expanded the Profile system to include non-lattice algorithms (FrodoKEM-640 + SLH-DSA-SHA2-128s).
- [ ] **Hybrid Extensions:** Support tri-hybrid modes (e.g., X25519 + ML-KEM + FrodoKEM).

### 6. Steganography & Deniability
- **LSB Embedding:** Hide ciphertexts within innocuous carrier files (WAV, RAW images).
- **Decoy Vaults:** Support secondary passphrases that open functional but innocuous "decoy" vaults to satisfy physical coercion.

---

## 📈 Status & Progress
- [x] **Initial Audit Received:** April 23, 2026.
- [ ] **Workstream 1 - HITL for MCP:** *Pending*
- [ ] **Workstream 2 - Hardware Binding:** *Pending*
