//go:build darwin

package mmap

func GetSystemMemInfo() (SystemMemInfo, error) {
	// macOS implementation: returning conservative values for now
	// In a real fork we would use sysctl
	return SystemMemInfo{TotalPhysical: 8 * 1024 * 1024 * 1024, AvailablePhysical: 4 * 1024 * 1024 * 1024}, nil
}

func TrimWorkingSet() {}
func SetWorkingSetLimit(maxBytes uint64) {}
