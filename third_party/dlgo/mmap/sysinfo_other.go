//go:build !linux && !windows && !darwin

package mmap

func GetSystemMemInfo() (SystemMemInfo, error) { return SystemMemInfo{}, nil }
func TrimWorkingSet() {}
func SetWorkingSetLimit(maxBytes uint64) {}
