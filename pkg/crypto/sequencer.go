package crypto

import (
	"fmt"
	"io"
)

type sequencerResult struct {
	index uint64
	data  []byte
	err   error
}

func runSequencer(w io.Writer, results <-chan sequencerResult, errChan <-chan error, writeFunc func(io.Writer, []byte) error) error {
	nextIndex := uint64(0)
	pending := make(map[uint64][]byte)
	for {
		select {
		case err := <-errChan:
			return err
		case res, ok := <-results:
			if !ok {
				if len(pending) > 0 {
					return fmt.Errorf("pipeline failed: missing chunks")
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

				if err := writeFunc(w, data); err != nil {
					return err
				}

				delete(pending, nextIndex)
				nextIndex++
			}
		}
	}
}
