package crypto

import (
	"crypto/cipher"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"runtime"
	"sync"
)

// bufferPool reuses 64KB buffers to reduce GC pressure.
var bufferPool = sync.Pool{
	New: func() interface{} {
		return make([]byte, ChunkSize)
	},
}

// EncryptStream symmetrically encrypts data from r to w using a passphrase and specified profile.
func EncryptStream(r io.Reader, w io.Writer, password []byte, flags byte, concurrency int, profileID byte) error {
	profile := DefaultProfile()
	if profileID != 0 {
		var err error
		profile, err = GetProfile(profileID, nil)
		if err != nil {
			return err
		}
	}

	// 1. Generate random Salt for KDF
	salt := make([]byte, profile.SaltSize())
	if _, err := io.ReadFull(rand.Reader, salt); err != nil {
		return err
	}

	// 2. Derive Key
	key := profile.DeriveKey(password, salt)
	defer SafeClear(key)

	// 3. Setup AEAD & Random Base Nonce
	aead, err := profile.NewAEAD(key)
	if err != nil {
		return err
	}
	baseNonce := make([]byte, aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, baseNonce); err != nil {
		return err
	}

	// 4. Write Header: Magic (4) | Version/ProfileID (1) | Flags (1) | Salt (N) | BaseNonce (N)
	if _, err := w.Write([]byte(MagicHeader)); err != nil {
		return err
	}
	if _, err := w.Write([]byte{profile.ID(), flags}); err != nil {
		return err
	}
	if _, err := w.Write(salt); err != nil {
		return err
	}
	if _, err := w.Write(baseNonce); err != nil {
		return err
	}

	// 5. Stream Encrypt Chunks
	return streamEncrypt(r, w, aead, baseNonce, concurrency)
}

// EncryptStreamWithPublicKeys encrypts data from r to w for one or more recipients.
func EncryptStreamWithPublicKeys(r io.Reader, w io.Writer, pubKeys [][]byte, flags byte, concurrency int, profileID byte) error {
	return EncryptStreamWithPublicKeysAndSigner(r, w, pubKeys, nil, flags, concurrency, profileID)
}

// EncryptStreamWithPublicKeysAndSigner is the internal implementation supporting optional integrated signing.
func EncryptStreamWithPublicKeysAndSigner(r io.Reader, w io.Writer, pubKeys [][]byte, signingKey []byte, flags byte, concurrency int, profileID byte) error {
	profile := DefaultProfile()
	if profileID != 0 {
		var err error
		profile, err = GetProfile(profileID, nil)
		if err != nil {
			return err
		}
	}

	if len(pubKeys) == 0 {
		return fmt.Errorf("at least one public key is required")
	}
	if len(pubKeys) > 255 {
		return fmt.Errorf("too many recipients (max 255)")
	}

	fek := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, fek); err != nil {
		return err
	}
	defer SafeClear(fek)

	type recipientHeader struct {
		pubKeyHash []byte // 4 bytes of SHA256(pubKey)
		ciphertext []byte
	}
	var recs []recipientHeader

	for _, pk := range pubKeys {
		ct, ss, err := profile.KEMEncapsulate(pk)
		if err != nil {
			return fmt.Errorf("failed to encapsulate for a recipient: %w", err)
		}

		wrappedFEK, err := wrapFEK(ss, fek)
		if err != nil {
			return err
		}
		SafeClear(ss)

		h := sha256Sum(pk)[:4]
		recs = append(recs, recipientHeader{
			pubKeyHash: h,
			ciphertext: append(ct, wrappedFEK...),
		})
	}

	aead, err := profile.NewAEAD(fek)
	if err != nil {
		return err
	}
	baseNonce := make([]byte, aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, baseNonce); err != nil {
		return err
	}

	// Integrated Signing: If signingKey is provided, sign the FEK + metadata commitment
	var signature []byte
	if len(signingKey) > 0 {
		commitment := append([]byte(MagicHeaderAsym), profile.ID(), flags)
		commitment = append(commitment, fek...)
		commitment = append(commitment, baseNonce...)
		
		sig, err := profile.Sign(commitment, signingKey)
		if err != nil {
			return fmt.Errorf("failed to generate integrated signature: %w", err)
		}
		signature = sig
		flags |= FlagSigned // Mark that this file has an integrated signature
	}

	// 4. Write Header: Magic (4) | ProfileID (1) | Flags (1) | RecipientCount (1) | [PackedProfile (7) if ID >= 128] | [Signature if FlagSigned] | RecipientBlock... | BaseNonce (N)
	if _, err := w.Write([]byte(MagicHeaderAsym)); err != nil {
		return err
	}
	if _, err := w.Write([]byte{profile.ID(), flags, byte(len(recs))}); err != nil {
		return err
	}

	if profile.ID() >= 128 {
		if dp, ok := profile.(*DynamicProfile); ok {
			if _, err := w.Write(dp.Pack()); err != nil {
				return err
			}
		}
	}
	
	if len(signature) > 0 {
		if _, err := w.Write(signature); err != nil {
			return err
		}
	}

	for _, r := range recs {
		if _, err := w.Write(r.pubKeyHash); err != nil {
			return err
		}
		if _, err := w.Write(r.ciphertext); err != nil {
			return err
		}
	}

	if _, err := w.Write(baseNonce); err != nil {
		return err
	}

	return streamEncrypt(r, w, aead, baseNonce, concurrency)
}

// Deprecated: Use EncryptStreamWithPublicKeys
func EncryptStreamWithPublicKey(r io.Reader, w io.Writer, pubKeyBytes []byte, flags byte, concurrency int, profileID byte) error {
	return EncryptStreamWithPublicKeys(r, w, [][]byte{pubKeyBytes}, flags, concurrency, profileID)
}

// ... rest of parallel streaming logic ...

type encryptJob struct {
	index uint64
	data  []byte
}

type encryptResult struct {
	index uint64
	data  []byte
	err   error
}

func streamEncrypt(r io.Reader, w io.Writer, aead cipher.AEAD, baseNonce []byte, concurrency int) error {
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}

	if concurrency == 1 {
		return streamEncryptSequential(r, w, aead, baseNonce)
	}

	sem := make(chan struct{}, concurrency*2)
	jobs := make(chan encryptJob, concurrency*2)
	results := make(chan encryptResult, concurrency*2)
	var wg sync.WaitGroup

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go encryptionWorker(&wg, jobs, results, aead, baseNonce)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	errChan := make(chan error, 1)
	go encryptionReader(r, jobs, errChan, sem)

	return encryptionSequencer(w, results, errChan, sem)
}

func encryptionWorker(wg *sync.WaitGroup, jobs <-chan encryptJob, results chan<- encryptResult, aead cipher.AEAD, baseNonce []byte) {
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

		ciphertext := aead.Seal(nil, nonce, job.data, nil)
		
		SafeClear(job.data)
		bufferPool.Put(job.data)

		results <- encryptResult{index: job.index, data: ciphertext}
	}
}

func encryptionReader(r io.Reader, jobs chan<- encryptJob, errChan chan<- error, sem chan struct{}) {
	defer close(jobs)
	chunkIndex := uint64(0)
	for {
		sem <- struct{}{}

		buf := bufferPool.Get().([]byte)
		n, err := r.Read(buf)
		if n > 0 {
			payload := buf[:n]
			jobs <- encryptJob{index: chunkIndex, data: payload}
			chunkIndex++
		} else {
			<-sem
			bufferPool.Put(buf)
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			errChan <- err
			return
		}
	}
}

func encryptionSequencer(w io.Writer, results <-chan encryptResult, errChan <-chan error, sem chan struct{}) error {
	nextIndex := uint64(0)
	pending := make(map[uint64][]byte)
	for {
		select {
		case err := <-errChan:
			return err
		case res, ok := <-results:
			if !ok {
				if len(pending) > 0 {
					return fmt.Errorf("encryption pipeline failed: missing chunks")
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

				if err := writeChunk(w, data); err != nil {
					return err
				}

				<-sem
				delete(pending, nextIndex)
				nextIndex++
			}
		}
	}
}

func writeChunk(w io.Writer, data []byte) error {
	lenBuf := make([]byte, 4)
	binary.LittleEndian.PutUint32(lenBuf, uint32(len(data)))
	if _, err := w.Write(lenBuf); err != nil {
		return err
	}
	if _, err := w.Write(data); err != nil {
		return err
	}
	return nil
}

func streamEncryptSequential(r io.Reader, w io.Writer, aead cipher.AEAD, baseNonce []byte) error {
	buf := bufferPool.Get().([]byte)
	defer bufferPool.Put(buf)
	
	chunkIndex := uint64(0)
	nonce := make([]byte, aead.NonceSize())

	for {
		n, err := r.Read(buf)
		if n > 0 {
			copy(nonce, baseNonce)
			counterBytes := make([]byte, 8)
			binary.LittleEndian.PutUint64(counterBytes, chunkIndex)

			offset := len(nonce) - 8
			for i := 0; i < 8; i++ {
				nonce[offset+i] ^= counterBytes[i]
			}

			ciphertext := aead.Seal(nil, nonce, buf[:n], nil)

			if err := writeChunk(w, ciphertext); err != nil {
				return err
			}
			chunkIndex++
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	return nil
}
