package agent

import (
	"fmt"
	"sync"

	"github.com/Harshal1000/ago"
)

// ---------------------------------------------------------------------------
// Backend Registry
// ---------------------------------------------------------------------------

// Backend identifies which LLM provider to use.
type Backend string

const (
	BackendGenAI  Backend = "genai"
	BackendOpenAI Backend = "openai"
)

var backends = map[Backend]func() (ago.LLM, error){}

// RegisterBackend registers a factory function for a named backend.
// LLM provider packages call this from their init() function.
func RegisterBackend(b Backend, factory func() (ago.LLM, error)) {
	backends[b] = factory
}

// NewLLMFromBackend creates an LLM instance from a registered backend.
func NewLLMFromBackend(b Backend) (ago.LLM, error) {
	factory, ok := backends[b]
	if !ok {
		return nil, fmt.Errorf("ago: unknown backend %q", b)
	}
	return factory()
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

// Agent is a configuration holder describing an LLM-powered agent.
// Execution is handled by the ago.Run / ago.RunSSE functions.
type Agent struct {
	Name          string              // required identifier
	Backend       Backend             // which LLM to create (e.g. BackendGenAI)
	Model         string              // model name passed to LLM per-call
	SystemPrompt  string              // optional system instruction
	Config        *ago.GenerateConfig // optional default generation parameters
	Tools         []ago.Tool          // tools available to the agent
	MaxIterations int                 // max agentic loop iterations (default 10)

	// LLM can be set directly to skip Backend-based creation (useful for testing).
	LLM ago.LLM
}

// ---------------------------------------------------------------------------
// Agent Registry
// ---------------------------------------------------------------------------

var (
	registry   = map[string]*Agent{}
	registryMu sync.RWMutex
)

// Register stores an agent in the global registry. Name must be non-empty and unique.
func Register(a *Agent) error {
	if a.Name == "" {
		return fmt.Errorf("ago: agent Name is required")
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, exists := registry[a.Name]; exists {
		return fmt.Errorf("ago: agent %q already registered", a.Name)
	}
	registry[a.Name] = a
	return nil
}

// Get retrieves an agent by name from the global registry.
func Get(name string) (*Agent, error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	a, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("ago: agent %q not found", name)
	}
	return a, nil
}

// ResetRegistry clears all registered agents. Intended for testing only.
func ResetRegistry() {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry = map[string]*Agent{}
}

// InitLLM lazily creates the LLM from Backend if LLM isn't already set.
func (a *Agent) InitLLM() error {
	if a.LLM != nil {
		return nil
	}
	llm, err := NewLLMFromBackend(a.Backend)
	if err != nil {
		return err
	}
	a.LLM = llm
	return nil
}

// ---------------------------------------------------------------------------
// AgentConfig interface implementation (for ago.Run / ago.RunSSE)
// ---------------------------------------------------------------------------

func (a *Agent) GetName() string                        { return a.Name }
func (a *Agent) GetModel() string                       { return a.Model }
func (a *Agent) GetLLM() ago.LLM                        { return a.LLM }
func (a *Agent) GetTools() []ago.Tool                   { return a.Tools }
func (a *Agent) GetMaxIterations() int                  { return a.MaxIterations }
func (a *Agent) GetGenerateConfig() *ago.GenerateConfig { return a.copyConfig() }

// GetSystemInstruction returns the agent's system prompt as a Content value,
// or nil if no system prompt is set.
func (a *Agent) GetSystemInstruction() *ago.Content {
	if a.SystemPrompt == "" {
		return nil
	}
	return ago.NewTextContent(ago.RoleSystem, a.SystemPrompt)
}

// Compile-time check.
var _ ago.AgentConfig = (*Agent)(nil)

// copyConfig returns a copy of the agent's Config (or an empty config if nil),
// so the executor never mutates the original.
func (a *Agent) copyConfig() *ago.GenerateConfig {
	cfg := a.Config
	if cfg == nil {
		cfg = &ago.GenerateConfig{}
	} else {
		copy := *cfg
		cfg = &copy
	}
	return cfg
}
