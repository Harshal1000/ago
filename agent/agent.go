package agent

import (
	"context"
	"fmt"
	"iter"

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
// Implements ago.Runner, ago.Streamer, and ago.Named.
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
// Runner / Streamer / Named implementation
// ---------------------------------------------------------------------------

// Run implements ago.Runner.
func (a *Agent) Run(ctx context.Context, contents []*ago.Content) (*ago.RunResult, error) {
	return ago.AgentLoop(ctx, a, contents)
}

// RunStream implements ago.Streamer.
func (a *Agent) RunStream(ctx context.Context, contents []*ago.Content) iter.Seq2[*ago.StreamChunk, error] {
	return ago.AgentLoopStream(ctx, a, contents)
}

// RunnerName implements ago.Named.
func (a *Agent) RunnerName() string { return a.Name }

// Compile-time checks.
var _ ago.Runner = (*Agent)(nil)
var _ ago.Streamer = (*Agent)(nil)
var _ ago.Named = (*Agent)(nil)

// ---------------------------------------------------------------------------
// AgentConfig interface implementation (for ago.AgentLoop / ago.AgentLoopStream)
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
