package crypto

import (
	"crypto/cipher"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"runtime"
	"sync"

	"golang.org/x/crypto/chacha20poly1305"
)

// DecryptStream decrypts data from r to w using a passphrase.
func DecryptStream(r io.Reader, w io.Writer, password []byte, concurrency int) (byte, error) {
	// 1. Read Fixed Header (Magic, Version, Flags)
	fixedHeader := make([]byte, 6)
	if _, err := io.ReadFull(r, fixedHeader); err != nil {
		return 0, err
	}

	if string(fixedHeader[:4]) != MagicHeader {
		return 0, errors.New("invalid file format: missing MAKN magic header")
	}

	profile, err := GetProfile(fixedHeader[4], r)
	if err != nil {
		return 0, err
	}
	flags := fixedHeader[5]

	// 2. Read Salt & Base Nonce
	salt := make([]byte, profile.SaltSize())
	if _, err := io.ReadFull(r, salt); err != nil {
		return 0, err
	}
	baseNonce := make([]byte, profile.NonceSize())
	if _, err := io.ReadFull(r, baseNonce); err != nil {
		return 0, err
	}

	// 3. Derive Key
	key := profile.DeriveKey(password, salt)
	defer SafeClear(key)

	// 4. Setup AEAD
	aead, err := profile.NewAEAD(key)
	if err != nil {
		return 0, err
	}

	// 5. Stream Decrypt Chunks
	return flags, streamDecrypt(r, w, aead, baseNonce, concurrency)
}

// DecryptStreamWithPrivateKey decrypts data from r to w using a Post-Quantum Private Key.
// If the file is signed (FlagSigned), it verifies the integrated signature.
func DecryptStreamWithPrivateKey(r io.Reader, w io.Writer, privKeyBytes []byte, concurrency int) (byte, error) {
	return DecryptStreamWithPrivateKeyAndVerifier(r, w, privKeyBytes, nil, concurrency)
}

// DecryptStreamWithPrivateKeyAndVerifier is the internal implementation supporting optional integrated verification.
func DecryptStreamWithPrivateKeyAndVerifier(r io.Reader, w io.Writer, privKeyBytes []byte, pubKeyBytes []byte, concurrency int) (byte, error) {
	// 1. Read Fixed Header (Magic, Version, Flags, RecipientCount)
	header := make([]byte, 7)
	if _, err := io.ReadFull(r, header); err != nil {
		return 0, err
	}

	if string(header[:4]) != MagicHeaderAsym {
		return 0, errors.New("invalid file format: missing MAKA magic header")
	}

	profile, err := GetProfile(header[4], r)
	if err != nil {
		return 0, err
	}
	flags := header[5]
	recipientCount := header[6]

	// 2. Optional: Read integrated signature
	var signature []byte
	if flags&FlagSigned != 0 {
		// ML-DSA-87 signatures are fixed size (mldsa87.SignatureSize = 4608)
		// Since we don't want to hardcode sizes, we should probably have a SIGSize in Profile
		// For now, using a standard large buffer if we can't get it from profile.
		signature = make([]byte, 4608) // Default for ProfileV1
		if _, err := io.ReadFull(r, signature); err != nil {
			return 0, fmt.Errorf("failed to read integrated signature: %w", err)
		}
	}

	// 3. Load ALL recipient blocks
	kemSize := profile.KEMCiphertextSize()
	wrappedSize := 32 + 16 // FEK(32) + Poly1305 Tag(16)
	blockSize := 4 + kemSize + wrappedSize

	blocks := make([]byte, int(recipientCount)*blockSize)
	if _, err := io.ReadFull(r, blocks); err != nil {
		return 0, fmt.Errorf("failed to read recipient blocks: %w", err)
	}

	// 4. Read Base Nonce
	baseNonce := make([]byte, profile.NonceSize())
	if _, err := io.ReadFull(r, baseNonce); err != nil {
		return 0, err
	}

	// 5. Find and Uncapsulate the FEK
	var fek []byte
	for i := 0; i < int(recipientCount); i++ {
		offset := i * blockSize
		ct := blocks[offset+4 : offset+4+kemSize]
		wrappedFEK := blocks[offset+4+kemSize : offset+blockSize]

		ss, err := profile.KEMDecapsulate(privKeyBytes, ct)
		if err != nil {
			continue // Not for this key
		}

		// Try to unwrap FEK
		unwrapped, err := unwrapFEK(ss, wrappedFEK)
		SafeClear(ss)
		if err == nil {
			fek = unwrapped
			break
		}
	}

	if fek == nil {
		return 0, fmt.Errorf("decryption failed: no recipient block matches the provided private key")
	}
	defer SafeClear(fek)

	// 6. Verify Integrated Signature if present
	if flags&FlagSigned != 0 {
		if len(pubKeyBytes) == 0 {
			// Without the sender's public key, we can't verify the signature.
			// This is an architectural choice: integrated signing REQUIRES the sender's identity.
			return 0, fmt.Errorf("file is signed but sender public key not provided")
		}
		
		commitment := append([]byte(MagicHeaderAsym), profile.ID(), flags)
		commitment = append(commitment, fek...)
		commitment = append(commitment, baseNonce...)
		
		if !profile.Verify(commitment, signature, pubKeyBytes) {
			return 0, fmt.Errorf("integrated signature verification FAILED: header might be tampered or sender unknown")
		}
	}

	// 7. Setup AEAD
	aead, err := profile.NewAEAD(fek)
	if err != nil {
		return 0, err
	}

	// 8. Stream Decrypt Chunks
	return flags, streamDecrypt(r, w, aead, baseNonce, concurrency)
}

// ... rest of parallel streaming logic ...

type decryptJob struct {
	index uint64
	data  []byte
}

type decryptResult struct {
	index uint64
	data  []byte
	err   error
}

func streamDecrypt(r io.Reader, w io.Writer, aead cipher.AEAD, baseNonce []byte, concurrency int) error {
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}

	if concurrency == 1 {
		return streamDecryptSequential(r, w, aead, baseNonce)
	}

	sem := make(chan struct{}, concurrency*2)
	jobs := make(chan decryptJob, concurrency*2)
	results := make(chan decryptResult, concurrency*2)
	var wg sync.WaitGroup

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go decryptionWorker(&wg, jobs, results, aead, baseNonce)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	errChan := make(chan error, 1)
	go decryptionReader(r, jobs, errChan, sem)

	return decryptionSequencer(w, results, errChan, sem)
}

func decryptionWorker(wg *sync.WaitGroup, jobs <-chan decryptJob, results chan<- decryptResult, aead cipher.AEAD, baseNonce []byte) {
	defer wg.Done()
	for job := range jobs {
		nonce := make([]byte, aead.NonceSize())
		copy(nonce, baseNonce)
		counterBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(counterBytes, job.index)

		offset := len(nonce) - 8
		for i := 0; i < 8; i++ {
			nonce[offset+i] ^= counterBytes[i]
		}

		plaintext, err := aead.Open(nil, nonce, job.data, nil)
		if err != nil {
			SafeClear(job.data)
			bufferPool.Put(job.data)
			results <- decryptResult{err: errors.New("authentication failed: incorrect key or corrupted data")}
			return
		}
		
		SafeClear(job.data)
		bufferPool.Put(job.data)

		results <- decryptResult{index: job.index, data: plaintext}
	}
}

func decryptionReader(r io.Reader, jobs chan<- decryptJob, errChan chan<- error, sem chan struct{}) {
	defer close(jobs)
	chunkIndex := uint64(0)
	lenBuf := make([]byte, 4)
	for {
		sem <- struct{}{}

		if _, err := io.ReadFull(r, lenBuf); err != nil {
			<-sem
			if err == io.EOF {
				break
			}
			errChan <- err
			return
		}

		chunkLen := binary.LittleEndian.Uint32(lenBuf)
		if chunkLen > uint32(ChunkSize+16) {
			<-sem
			errChan <- errors.New("corrupted payload: chunk size exceeds maximum")
			return
		}

		ciphertext := bufferPool.Get().([]byte)
		if cap(ciphertext) < int(chunkLen) {
			ciphertext = make([]byte, chunkLen)
		} else {
			ciphertext = ciphertext[:chunkLen]
		}

		if _, err := io.ReadFull(r, ciphertext); err != nil {
			<-sem
			errChan <- err
			return
		}

		jobs <- decryptJob{index: chunkIndex, data: ciphertext}
		chunkIndex++
	}
}

func decryptionSequencer(w io.Writer, results <-chan decryptResult, errChan <-chan error, sem chan struct{}) error {
	nextIndex := uint64(0)
	pending := make(map[uint64][]byte)
	for {
		select {
		case err := <-errChan:
			return err
		case res, ok := <-results:
			if !ok {
				if len(pending) > 0 {
					return fmt.Errorf("decryption pipeline failed: missing chunks")
				}
				return nil
			}
			if res.err != nil {
				return res.err
			}

			pending[res.index] = res.data

			for {
				data, exists := pending[nextIndex]
				if !exists {
					break
				}

				if _, err := w.Write(data); err != nil {
					return err
				}

				SafeClear(data)
				<-sem
				delete(pending, nextIndex)
				nextIndex++
			}
		}
	}
}

func streamDecryptSequential(r io.Reader, w io.Writer, aead cipher.AEAD, baseNonce []byte) error {
	chunkIndex := uint64(0)
	nonce := make([]byte, aead.NonceSize())
	lenBuf := make([]byte, 4)

	buf := bufferPool.Get().([]byte)
	defer bufferPool.Put(buf)

	for {
		if _, err := io.ReadFull(r, lenBuf); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		chunkLen := binary.LittleEndian.Uint32(lenBuf)
		if chunkLen > uint32(ChunkSize+16) {
			return errors.New("corrupted payload: chunk size exceeds maximum")
		}

		ciphertext := buf[:chunkLen]
		if _, err := io.ReadFull(r, ciphertext); err != nil {
			return err
		}

		copy(nonce, baseNonce)
		counterBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(counterBytes, chunkIndex)
		offset := len(nonce) - 8
		for i := 0; i < 8; i++ {
			nonce[offset+i] ^= counterBytes[i]
		}

		plaintext, err := aead.Open(nil, nonce, ciphertext, nil)
		if err != nil {
			return errors.New("authentication failed: incorrect key or corrupted data")
		}

		if _, err := w.Write(plaintext); err != nil {
			SafeClear(plaintext)
			return err
		}
		SafeClear(plaintext)
		chunkIndex++
	}
	return nil
}
