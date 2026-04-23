package crypto

import (
	"testing"
)

type MockRegistry struct {
	IdentityRegistry
}

func TestRegistryFactory(t *testing.T) {
	// Register a mock registry
	RegisterRegistry("mock", func() IdentityRegistry {
		return &MockRegistry{}
	})

	// Override config for test
	ResetGlobalConfig()
	conf := GetGlobalConfig()
	conf.IdentityRegistries = []string{"mock"}

	reg := NewIdentityRegistry()
	mr, ok := reg.(*MultiRegistry)
	if !ok {
		t.Fatal("expected MultiRegistry")
	}

	found := false
	for _, r := range mr.Registries {
		if _, ok := r.(*MockRegistry); ok {
			found = true
			break
		}
	}

	if !found {
		t.Error("MockRegistry not found in active registries")
	}
}
