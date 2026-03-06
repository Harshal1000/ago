package ago

import (
	"context"
	"iter"

	"github.com/Harshal1000/ago/storage"
)

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
