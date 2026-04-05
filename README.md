# Maknoon (مكنون) 🛡️

**Maknoon** (Arabic: مكنون) translates to *the hidden*, *the concealed*, or *that which is carefully preserved*. 

Maknoon is a versatile, ultra-efficient CLI encryption tool designed for a post-quantum world. It combines bleeding-edge cryptographic standards with a high-performance streaming architecture to protect your data with absolute care.

## ✨ Core Philosophies

1.  **Bleeding-Edge Security:** Uses hybrid cryptographic schemes, preparing your data for the future of quantum computing.
2.  **Hyper-Efficiency:** Processes massive files (100GB+) with a **constant memory footprint** (~64KB) using advanced streaming I/O.
3.  **Memory Hygiene:** Strictly adheres to the "Carefully Preserved" ethos by zeroing out all sensitive data (passphrases, keys, secrets) from RAM immediately after use.
4.  **Modern DX:** Intuitive CLI with real-time progress feedback and automatic header detection.

---

## 🛠 Technical Stack

*   **Symmetric Encryption:** XChaCha20-Poly1305 (Fast, authenticated encryption).
*   **Key Derivation:** Argon2id (Memory-hard, GPU-resistant).
*   **Post-Quantum KEM:** ML-KEM / Kyber1024 (NIST-standardized quantum resistance).
*   **Streaming:** Chunked AEAD (64KB blocks) using `io.Reader` / `io.Writer`.

---

## 🚀 Getting Started

### Installation

Requires Go 1.21+

```bash
git clone https://github.com/a-khallaf/maknoon.git
cd maknoon
go build -o maknoon ./cmd/maknoon
```

### 1. Key Generation (Post-Quantum)
Generate a Kyber1024 keypair. By default, your private key is protected with an Argon2id-derived passphrase.

```bash
./maknoon keygen -o id_identity
```

### 2. Encryption

**Passphrase Mode:**
```bash
./maknoon encrypt sensitive_report.pdf
```

**Asymmetric (Public Key) Mode:**
```bash
./maknoon encrypt massive_data.iso --pubkey id_identity.pub
```

### 3. Decryption
Maknoon automatically detects the encryption type and handles key-unlocking seamlessly.

```bash
./maknoon decrypt massive_data.iso.makn
```

---

## 🏗 Architecture & Security

### AEAD Streaming
To ensure high performance and integrity, Maknoon uses a chunked STREAM construction. Each 64KB chunk is encrypted with a unique nonce derived from a per-file random base and a 64-bit counter. This prevents nonce-reuse vulnerabilities while allowing for bit-perfect restoration.

### "Carefully Preserved" RAM
Sensitive data is never left to the garbage collector. Maknoon uses `defer` blocks to explicitly overwrite byte slices containing passphrases and raw keys with zeros as soon as the cryptographic operations (Seal, Open, Encapsulate, Decapsulate) are finished.

### Performance
Verified on modern hardware to exceed **1.0 GB/s** throughput for both encryption and decryption, saturating most high-speed SSDs.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
