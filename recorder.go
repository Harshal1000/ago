package ago

import (
	"context"
	"encoding/json"
	"time"

	"github.com/Harshal1000/ago/storage"
	"github.com/google/uuid"
)

// recorder bridges ago types and the storage layer.
// One recorder is created per Run/RunSSE invocation.
// Events are buffered in memory and flushed to storage in a single batch on Flush.
type recorder struct {
	svc       storage.Service
	limit     int // HistoryLimit from App (0 = unlimited)
	sessionID string
	messageID string // UUIDv7, unique per Run invocation
	userID    string
	author    string
	isNew     bool
	buf       []*storage.Event // buffered events, flushed in one batch
}

// newRecorder builds a recorder from App config and per-turn RunOptions.
// Returns nil when storage is not configured.
func newRecorder(app *App, agentName string, opts *RunOptions) *recorder {
	if app == nil || app.Storage == nil {
		return nil
	}
	if opts == nil {
		opts = &RunOptions{}
	}

	sid := opts.SessionID
	isNew := sid == ""
	if isNew {
		sid = newV7()
	}

	author := opts.Author
	if author == "" {
		author = agentName
	}

	return &recorder{
		svc:       app.Storage,
		limit:     app.HistoryLimit,
		sessionID: sid,
		messageID: newV7(),
		userID:    opts.UserID,
		author:    author,
		isNew:     isNew,
	}
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
	event := &storage.Event{
		ID:        newV7(),
		SessionID: r.sessionID,
		MessageID: r.messageID,
		UserID:    r.userID,
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

// newV7 generates a UUIDv7 string.
func newV7() string {
	return uuid.Must(uuid.NewV7()).String()
}
