//go:build linux

package mmap

import "syscall"

func GetSystemMemInfo() (SystemMemInfo, error) {
	var si syscall.Sysinfo_t
	if err := syscall.Sysinfo(&si); err != nil { return SystemMemInfo{}, err }
	unit := uint64(si.Unit)
	return SystemMemInfo{TotalPhysical: uint64(si.Totalram) * unit, AvailablePhysical: uint64(si.Freeram) * unit}, nil
}

func TrimWorkingSet() {}
func SetWorkingSetLimit(maxBytes uint64) {}
