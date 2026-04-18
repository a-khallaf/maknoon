package crypto

import (
	"crypto/cipher"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"runtime"
	"sync"
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

	id := fixedHeader[4]
	flags := fixedHeader[5]

	profile, err := GetProfile(id, r)
	if err != nil {
		return 0, err
	}

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

	id := header[4]
	flags := header[5]
	recipientCount := header[6]

	profile, err := GetProfile(id, r)
	if err != nil {
		return 0, err
	}

	// 2. Load ALL recipient blocks
	kemSize := profile.KEMCiphertextSize()
	wrappedSize := 32 + 16
	blockSize := 4 + kemSize + wrappedSize

	blocks := make([]byte, int(recipientCount)*blockSize)
	if _, err := io.ReadFull(r, blocks); err != nil {
		return 0, fmt.Errorf("failed to read recipient blocks: %w", err)
	}

	// 3. Optional: Read integrated signature (Moved AFTER recipient blocks in header)
	var signature []byte
	if flags&FlagSigned != 0 {
		signature = make([]byte, profile.SIGSize())
		if _, err := io.ReadFull(r, signature); err != nil {
			return 0, fmt.Errorf("failed to read integrated signature: %w", err)
		}
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
			return 0, fmt.Errorf("file is signed but sender public key not provided")
		}

		commitment := make([]byte, 0, 4+1+1+len(fek)+len(baseNonce))
		commitment = append(commitment, []byte(MagicHeaderAsym)...)
		commitment = append(commitment, profile.ID(), flags)
		commitment = append(commitment, fek...)
		commitment = append(commitment, baseNonce...)

		if !profile.Verify(commitment, signature, pubKeyBytes) {
			return 0, fmt.Errorf("integrated signature verification FAILED")
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

	sem := make(chan struct{}, concurrency*4)
	jobs := make(chan decryptJob, concurrency*2)
	results := make(chan decryptResult, concurrency*2)
	var wg sync.WaitGroup

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go decryptionWorker(&wg, jobs, results, aead, baseNonce, sem)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	errChan := make(chan error, 1)
	go decryptionReader(r, jobs, errChan, sem)

	return decryptionSequencer(w, results, errChan)
}

func decryptionWorker(wg *sync.WaitGroup, jobs <-chan decryptJob, results chan<- decryptResult, aead cipher.AEAD, baseNonce []byte, sem chan struct{}) {
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
			ptr := &job.data
			bufferPool.Put(ptr)
			<-sem
			results <- decryptResult{err: errors.New("authentication failed: incorrect key or corrupted data")}
			return
		}

		SafeClear(job.data)
		ptr := &job.data
		bufferPool.Put(ptr)
		<-sem

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

		workerBufPtr := bufferPool.Get().(*[]byte)
		workerBuf := *workerBufPtr
		if cap(workerBuf) < int(chunkLen) {
			workerBuf = make([]byte, chunkLen)
		} else {
			workerBuf = workerBuf[:chunkLen]
		}

		if _, err := io.ReadFull(r, workerBuf); err != nil {
			<-sem
			errChan <- err
			return
		}

		jobs <- decryptJob{index: chunkIndex, data: workerBuf}
		chunkIndex++
	}
}

func decryptionSequencer(w io.Writer, results <-chan decryptResult, errChan <-chan error) error {
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

	bufPtr := bufferPool.Get().(*[]byte)
	buf := *bufPtr
	defer bufferPool.Put(bufPtr)

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
