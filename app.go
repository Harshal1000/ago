package ago

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"time"

	"github.com/Harshal1000/ago/storage"
	"github.com/google/uuid"
)

// ---------------------------------------------------------------------------
// Runner / Streamer / Named
// ---------------------------------------------------------------------------

// Runner executes an agentic task. Runners are pure — they don't touch storage.
type Runner interface {
	Run(ctx context.Context, contents []*Content) (*RunResult, error)
}

// Streamer is optionally implemented by Runners that support streaming.
type Streamer interface {
	RunStream(ctx context.Context, contents []*Content) iter.Seq2[*StreamChunk, error]
}

// Named is optionally implemented by Runners that have a name.
// Used by Orchestrate to register workers as named tools, and by logging hooks.
type Named interface {
	RunnerName() string
}

// ---------------------------------------------------------------------------
// RunContext
// ---------------------------------------------------------------------------

// RunContext carries immutable per-run metadata through context.Context.
// Hooks, agent name, session ID, and user ID flow through here.
type RunContext struct {
	AgentName string
	SessionID string
	UserID    string
	Hooks     *Hooks
}

type runContextKey struct{}

// WithRunContext returns a new context carrying the given RunContext.
func WithRunContext(ctx context.Context, rc *RunContext) context.Context {
	return context.WithValue(ctx, runContextKey{}, rc)
}

// GetRunContext retrieves the RunContext from ctx, or returns an empty RunContext if none is set.
func GetRunContext(ctx context.Context) *RunContext {
	rc, _ := ctx.Value(runContextKey{}).(*RunContext)
	if rc == nil {
		return &RunContext{}
	}
	return rc
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

// Hooks holds optional callbacks that fire at key points in the agentic loop.
// All fields are optional; nil fields are silently skipped.
type Hooks struct {
	// BeforeLLMCall fires before each LLM call in the agentic loop.
	BeforeLLMCall func(ctx context.Context, params *GenerateParams) error

	// AfterLLMCall fires after each blocking LLM call.
	AfterLLMCall func(ctx context.Context, resp *Response)

	// BeforeToolCall fires before each tool is executed (in parallel goroutines).
	BeforeToolCall func(ctx context.Context, call *FunctionCall) error

	// AfterToolCall fires after each tool finishes, including tools that returned errors.
	AfterToolCall func(ctx context.Context, call *FunctionCall, result *ToolResult)

	// OnComplete fires once when the agentic loop finishes successfully.
	OnComplete func(ctx context.Context, result *RunResult)
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

// App holds app-level infrastructure shared across all agents and sessions.
type App struct {
	// Name identifies this application.
	Name string

	// Description is a human-readable description of this application.
	Description string

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

	// Runner is the default runner used when Run/RunSSE are called with nil runner.
	Runner Runner
}

// ---------------------------------------------------------------------------
// RunOptions / RunResult
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// App.Run
// ---------------------------------------------------------------------------

// Run executes a runner synchronously and returns the final result.
// Pass runner=nil to use app.Runner; pass a Runner to override for that call.
// Pass opts to enable session persistence; nil opts runs ephemerally.
func (app *App) Run(ctx context.Context, runner Runner, contents []*Content, opts *RunOptions) (*RunResult, error) {
	r := runner
	if r == nil {
		r = app.Runner
	}
	if r == nil {
		return nil, fmt.Errorf("ago: no runner configured")
	}

	rc := &RunContext{Hooks: app.Hooks}
	if opts != nil {
		rc.UserID = opts.UserID
	}

	inputLen := len(contents)

	// Storage: create/resume session, store user input, load history.
	if app.Storage != nil && opts != nil {
		sid, isNew, err := resolveSession(ctx, app, opts)
		if err != nil {
			return nil, err
		}
		rc.SessionID = sid

		// Store user input events immediately.
		storeEvents(ctx, app.Storage, sid, rc.UserID, contents)

		// Load history if configured and session is not brand new.
		if app.IncludeHistory && !isNew {
			prior, err := loadHistory(ctx, app.Storage, sid, app.HistoryLimit)
			if err == nil && len(prior) > 0 {
				contents = append(prior, contents...)
			}
		}
	}

	ctx = WithRunContext(ctx, rc)
	result, err := r.Run(ctx, contents)
	if err != nil {
		return nil, err
	}

	// Store model events (everything after input contents).
	if app.Storage != nil && opts != nil && rc.SessionID != "" {
		newEvents := result.History[inputLen:]
		if len(newEvents) > 0 {
			storeEvents(ctx, app.Storage, rc.SessionID, rc.UserID, newEvents)
		}
	}

	result.SessionID = rc.SessionID
	return result, nil
}

// ---------------------------------------------------------------------------
// App.RunSSE
// ---------------------------------------------------------------------------

// RunSSE executes a runner with streaming, yielding chunks to the caller.
// If the runner implements Streamer, true streaming is used.
// Otherwise the runner runs synchronously and a single final chunk is yielded.
func (app *App) RunSSE(ctx context.Context, runner Runner, contents []*Content, opts *RunOptions) iter.Seq2[*StreamChunk, error] {
	r := runner
	if r == nil {
		r = app.Runner
	}
	if r == nil {
		return func(yield func(*StreamChunk, error) bool) {
			yield(nil, fmt.Errorf("ago: no runner configured"))
		}
	}

	rc := &RunContext{Hooks: app.Hooks}
	if opts != nil {
		rc.UserID = opts.UserID
	}

	inputLen := len(contents)

	// Storage: create/resume session, store user input, load history.
	if app.Storage != nil && opts != nil {
		sid, isNew, err := resolveSession(ctx, app, opts)
		if err != nil {
			return func(yield func(*StreamChunk, error) bool) {
				yield(nil, err)
			}
		}
		rc.SessionID = sid
		storeEvents(ctx, app.Storage, sid, rc.UserID, contents)
		if app.IncludeHistory && !isNew {
			prior, err := loadHistory(ctx, app.Storage, sid, app.HistoryLimit)
			if err == nil && len(prior) > 0 {
				contents = append(prior, contents...)
			}
		}
	}

	ctx = WithRunContext(ctx, rc)

	// If runner supports streaming, use it.
	if s, ok := r.(Streamer); ok {
		return func(yield func(*StreamChunk, error) bool) {
			var lastResult *RunResult
			for chunk, err := range s.RunStream(ctx, contents) {
				if err != nil {
					yield(nil, err)
					return
				}
				if chunk.Complete {
					// Capture for storage.
					lastResult = &RunResult{
						History:   contents, // approximate; stream doesn't return full history
						SessionID: rc.SessionID,
					}
				}
				if !yield(chunk, nil) {
					return
				}
			}
			// Store model events after stream completes.
			if app.Storage != nil && opts != nil && rc.SessionID != "" && lastResult != nil {
				newEvents := lastResult.History[inputLen:]
				if len(newEvents) > 0 {
					storeEvents(ctx, app.Storage, rc.SessionID, rc.UserID, newEvents)
				}
			}
		}
	}

	// Fallback: run synchronously and emit a single final chunk.
	return func(yield func(*StreamChunk, error) bool) {
		result, err := r.Run(ctx, contents)
		if err != nil {
			yield(nil, err)
			return
		}

		// Store model events.
		if app.Storage != nil && opts != nil && rc.SessionID != "" {
			newEvents := result.History[inputLen:]
			if len(newEvents) > 0 {
				storeEvents(ctx, app.Storage, rc.SessionID, rc.UserID, newEvents)
			}
		}

		chunk := &StreamChunk{Complete: true}
		if result.Response != nil && len(result.Response.Candidates) > 0 {
			chunk.Candidates = result.Response.Candidates
			usage := result.Response.Usage
			chunk.Usage = &usage
		}
		yield(chunk, nil)
	}
}

// ---------------------------------------------------------------------------
// RunEphemeral
// ---------------------------------------------------------------------------

// RunEphemeral runs a runner for a single turn with no storage and no history.
func RunEphemeral(ctx context.Context, runner Runner, contents []*Content) (*RunResult, error) {
	return runner.Run(ctx, contents)
}

// ---------------------------------------------------------------------------
// Storage helpers (replaces recorder.go)
// ---------------------------------------------------------------------------

// resolveSession creates or resumes a session. Returns (sessionID, isNew, error).
func resolveSession(ctx context.Context, app *App, opts *RunOptions) (string, bool, error) {
	if opts.SessionID != "" {
		return opts.SessionID, false, nil
	}
	sid := newV7()
	author := opts.Author
	if author == "" && app.Name != "" {
		author = app.Name
	}
	if err := app.Storage.CreateSession(ctx, &storage.Session{
		ID:     sid,
		UserID: opts.UserID,
		Author: author,
	}); err != nil {
		return "", false, fmt.Errorf("ago: storage: %w", err)
	}
	return sid, true, nil
}

// loadHistory retrieves stored events and converts them back to Content slices.
func loadHistory(ctx context.Context, svc storage.Service, sessionID string, limit int) ([]*Content, error) {
	events, err := svc.GetRecentEvents(ctx, sessionID, limit)
	if err != nil {
		return nil, err
	}
	contents := make([]*Content, 0, len(events))
	for _, e := range events {
		var c Content
		if err := json.Unmarshal(e.Content, &c); err != nil {
			continue
		}
		contents = append(contents, &c)
	}
	return contents, nil
}

// storeEvents marshals Content to storage events and batch-inserts them.
// Errors are silently ignored so storage never breaks the agentic loop.
func storeEvents(ctx context.Context, svc storage.Service, sessionID, userID string, contents []*Content) {
	if len(contents) == 0 {
		return
	}
	msgID := newV7()
	events := make([]*storage.Event, 0, len(contents))
	for _, c := range contents {
		data, err := json.Marshal(c)
		if err != nil {
			continue
		}
		events = append(events, &storage.Event{
			ID:        newV7(),
			SessionID: sessionID,
			MessageID: msgID,
			UserID:    userID,
			Content:   data,
			CreatedAt: time.Now(),
		})
	}
	_ = svc.CreateEvents(ctx, events)
}

// newV7 generates a UUIDv7 string.
func newV7() string {
	return uuid.Must(uuid.NewV7()).String()
}

// ---------------------------------------------------------------------------
// Runner name helper
// ---------------------------------------------------------------------------

// runnerName returns the name of a Runner if it implements Named, or fallback otherwise.
func runnerName(r Runner, fallback string) string {
	if n, ok := r.(Named); ok {
		return n.RunnerName()
	}
	return fallback
}
