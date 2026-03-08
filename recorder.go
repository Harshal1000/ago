package ago

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Harshal1000/ago/storage"
	"github.com/google/uuid"
)

// recorder bridges ago types and the storage layer.
// One recorder is created per Run/RunSSE invocation.
// Events are buffered in memory and flushed to storage in a single batch on Flush.
type recorder struct {
	svc             storage.Service
	limit           int // HistoryLimit from App (0 = unlimited)
	sessionID       string
	messageID       string // UUIDv7, unique per Run invocation
	userID          string
	author          string
	agentName       string
	isNew           bool
	skipInputBuffer bool             // true for sub-agents; strategy buffers input once at top level
	buf             []*storage.Event // buffered events, flushed in one batch
}

// newRecorder builds a recorder from App config and per-turn RunOptions.
// Returns nil when storage is not configured or opts is nil (ephemeral run).
func newRecorder(app *App, agentName string, opts *RunOptions) *recorder {
	if app == nil || app.Storage == nil || opts == nil {
		return nil
	}

	sid := opts.SessionID
	isNew := sid == ""
	if isNew {
		sid = newV7()
	}

	author := opts.Author
	if author == "" && app.Name != "" {
		author = app.Name
	}
	if author == "" {
		author = agentName
	}

	return &recorder{
		svc:             app.Storage,
		limit:           app.HistoryLimit,
		sessionID:       sid,
		messageID:       newV7(),
		userID:          opts.UserID,
		author:          author,
		agentName:       agentName,
		isNew:           isNew,
		skipInputBuffer: app.isSubAgent,
	}
}

// ensureStrategySession creates a storage session for a strategy run if storage
// is configured and opts has no existing SessionID. Returns updated opts with
// SessionID set, or the original opts unchanged when nothing needs to be done.
func ensureStrategySession(ctx context.Context, app *App, strategyName string, opts *RunOptions) (*RunOptions, error) {
	if app == nil || app.Storage == nil || opts == nil || opts.SessionID != "" {
		return opts, nil
	}
	newOpts := *opts
	newOpts.SessionID = newV7()
	author := newOpts.Author
	if author == "" && app.Name != "" {
		author = app.Name
	}
	if author == "" {
		author = strategyName
	}
	if err := app.Storage.CreateSession(ctx, &storage.Session{
		ID:     newOpts.SessionID,
		UserID: newOpts.UserID,
		Author: author,
	}); err != nil {
		return nil, fmt.Errorf("ago: storage: %w", err)
	}
	return &newOpts, nil
}

// SessionID returns the session identifier (auto-generated or resumed).
func (r *recorder) SessionID() string {
	if r == nil {
		return ""
	}
	return r.sessionID
}

// EnsureSession creates the session when isNew=true.
// For resumed sessions the SELECT is skipped entirely — an invalid session ID
// will surface as an FK violation on the first Flush.
func (r *recorder) EnsureSession(ctx context.Context) error {
	if !r.isNew {
		return nil
	}
	return r.svc.CreateSession(ctx, &storage.Session{
		ID:     r.sessionID,
		UserID: r.userID,
		Author: r.author,
	})
}

// Buffer queues a Content turn to be persisted on the next Flush.
func (r *recorder) Buffer(c *Content) {
	r.buffer(c, nil)
}

// BufferWithUsage queues a Content turn together with token usage data.
func (r *recorder) BufferWithUsage(c *Content, usage *TokenUsage) {
	r.buffer(c, usage)
}

func (r *recorder) buffer(c *Content, usage *TokenUsage) {
	contentData, err := json.Marshal(c)
	if err != nil {
		return
	}
	// User-role content is always stored without agent attribution —
	// it belongs to the session, not to any particular agent.
	agent := r.agentName
	if c.Role == RoleUser {
		agent = ""
	}
	event := &storage.Event{
		ID:        newV7(),
		SessionID: r.sessionID,
		MessageID: r.messageID,
		UserID:    r.userID,
		Agent:     agent,
		Content:   contentData,
		CreatedAt: time.Now(),
	}
	if usage != nil {
		if data, err := json.Marshal(usage); err == nil {
			event.Usage = data
		}
	}
	r.buf = append(r.buf, event)
}

// Flush persists all buffered events in a single batch INSERT.
// Errors are silently ignored so storage never breaks the agentic loop.
func (r *recorder) Flush(ctx context.Context) {
	if len(r.buf) == 0 {
		return
	}
	_ = r.svc.CreateEvents(ctx, r.buf)
	r.buf = nil
}

// LoadHistory retrieves stored events and converts them back to Content slices.
// Returns nil immediately for new sessions (nothing in storage yet).
// Respects the recorder's limit (HistoryLimit from App).
func (r *recorder) LoadHistory(ctx context.Context) ([]*Content, error) {
	if r.isNew {
		return nil, nil
	}
	events, err := r.svc.GetRecentEvents(ctx, r.sessionID, r.limit)
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

// bufferStrategyInput stores the incoming user contents once at the strategy level.
// Called by Sequential, Parallel, and Loop before dispatching sub-agents, so that
// the shared session has exactly one copy of the user input (not one per sub-agent).
// Sub-agents skip buffering their own input via skipInputBuffer=true (set by subApp).
func bufferStrategyInput(ctx context.Context, app *App, opts *RunOptions, contents []*Content) {
	if app == nil || app.Storage == nil || opts == nil || opts.SessionID == "" {
		return
	}
	rec := newRecorder(app, "", opts)
	if rec == nil {
		return
	}
	for _, c := range contents {
		rec.Buffer(c)
	}
	rec.Flush(ctx)
}

// newV7 generates a UUIDv7 string.
func newV7() string {
	return uuid.Must(uuid.NewV7()).String()
}
