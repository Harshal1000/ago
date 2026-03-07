package ago

import (
	"context"
	"iter"

	"github.com/Harshal1000/ago/storage"
)

// Hooks holds optional callbacks that fire at key points in the agentic loop.
// All fields are optional; nil fields are silently skipped.
type Hooks struct {
	// BeforeLLMCall fires before each LLM call in the agentic loop.
	// Return a non-nil error to abort the loop immediately as an infrastructure error.
	BeforeLLMCall func(ctx context.Context, params *GenerateParams) error

	// AfterLLMCall fires after each blocking LLM call in run().
	// Not called during RunSSE (streaming has no single *Response object).
	AfterLLMCall func(ctx context.Context, resp *Response)

	// BeforeToolCall fires before each tool is executed (in parallel goroutines).
	// Return a non-nil error to abort the agentic loop immediately as an infrastructure error.
	BeforeToolCall func(ctx context.Context, call *FunctionCall) error

	// AfterToolCall fires after each tool finishes, including tools that returned errors.
	// result.Error is set if the tool returned a tool-level error.
	AfterToolCall func(ctx context.Context, call *FunctionCall, result *ToolResult)

	// OnComplete fires once when the agentic loop finishes successfully.
	// Called in both run() (with full RunResult) and runSSE() (with partial RunResult: history + nil Response).
	OnComplete func(ctx context.Context, result *RunResult)
}

// App holds app-level infrastructure shared across all agents and sessions.
// Create one App per application; pass it to Run, RunSSE, and Compact.
type App struct {
	// Storage is the persistence backend for sessions and events.
	// nil disables all persistence.
	Storage storage.Service

	// HistoryLimit is the maximum number of stored events loaded into LLM
	// context per turn. 0 means unlimited.
	HistoryLimit int

	// IncludeHistory, when true, prepends stored conversation history into
	// the LLM context on every turn. Requires Storage to be set.
	IncludeHistory bool

	// Hooks is an optional set of callbacks for observing the agentic loop.
	Hooks *Hooks
}

// RunOptions carries per-turn session identity.
// Pass nil to run without storage (ephemeral, no session).
type RunOptions struct {
	// SessionID resumes an existing session. Empty creates a new one.
	SessionID string

	// UserID identifies the user for this turn.
	UserID string

	// Author labels who owns this session (agent name, service name, etc.).
	Author string
}

// RunResult is the outcome of App.Run or the final state of App.RunSSE.
type RunResult struct {
	// Response is the final LLM response (or synthetic response for SkipSynthesis tools).
	Response *Response

	// History is the full conversation including tool calls and responses.
	History []*Content

	// SessionID is the session used for this run. Empty when storage is not configured.
	SessionID string
}

// Run executes the agentic loop synchronously and returns the final result.
// Pass opts to enable session persistence; nil opts runs ephemerally.
func (app *App) Run(ctx context.Context, agent AgentConfig, contents []*Content, opts *RunOptions) (*RunResult, error) {
	return run(ctx, app, agent, contents, opts)
}

// RunSSE executes the agentic loop with streaming, yielding chunks to the caller.
// Pass opts to enable session persistence; nil opts runs ephemerally.
func (app *App) RunSSE(ctx context.Context, agent AgentConfig, contents []*Content, opts *RunOptions) iter.Seq2[*StreamChunk, error] {
	return runSSE(ctx, app, agent, contents, opts)
}

// RunEphemeral runs an agent for a single turn with no storage and no history.
// Use this for sub-agent calls, one-off queries, or any context where
// persistence and history are not needed.
func RunEphemeral(ctx context.Context, agent AgentConfig, contents []*Content) (*RunResult, error) {
	return run(ctx, nil, agent, contents, nil)
}

// Compact summarizes old events for a session into a single summary event,
// keeping the most recent turns intact. This bounds the cost of LoadHistory
// as conversations grow long.
//
// Not yet implemented — returns nil immediately.
func (app *App) Compact(ctx context.Context, agent AgentConfig, sessionID string) error {
	// TODO: load all events, ask LLM to summarize, replace old events with summary event
	return nil
}
