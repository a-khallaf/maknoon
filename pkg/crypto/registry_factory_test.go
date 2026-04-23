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

	if len(mr.Registries) != 1 {
		t.Errorf("expected 1 registry, got %d", len(mr.Registries))
	}

	if _, ok := mr.Registries[0].(*MockRegistry); !ok {
		t.Error("expected MockRegistry as first registry")
	}
}
