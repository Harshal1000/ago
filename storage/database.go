package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type database struct {
	pool *pgxpool.Pool
}

func newDatabase(ctx context.Context, cfg *Config) (*database, error) {
	poolCfg, err := pgxpool.ParseConfig(cfg.DatabaseURL)
	if err != nil {
		return nil, fmt.Errorf("ago: storage: parse config: %w", err)
	}

	if cfg.MaxConns > 0 {
		poolCfg.MaxConns = cfg.MaxConns
	}
	if cfg.MinConns > 0 {
		poolCfg.MinConns = cfg.MinConns
	}
	if cfg.Schema != "" {
		poolCfg.ConnConfig.RuntimeParams["search_path"] = cfg.Schema
	}

	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		return nil, fmt.Errorf("ago: storage: connect: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ago: storage: ping: %w", err)
	}

	db := &database{pool: pool}
	if cfg.autoMigrate() {
		if err := db.migrate(ctx); err != nil {
			pool.Close()
			return nil, err
		}
	}
	return db, nil
}

var _ Service = (*database)(nil)

func (db *database) migrate(ctx context.Context) error {
	ddl := `
CREATE TABLE IF NOT EXISTS sessions (
    id         UUID PRIMARY KEY,
    user_id    UUID NOT NULL,
    author     TEXT NOT NULL DEFAULT '',
    title      TEXT NOT NULL DEFAULT '',
    metadata   JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

CREATE TABLE IF NOT EXISTS events (
    id         UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id UUID NOT NULL,
    user_id    UUID NOT NULL,
    content    JSONB NOT NULL,
    error      TEXT NOT NULL DEFAULT '',
    usage      JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata   JSONB
);
CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);
`
	if _, err := db.pool.Exec(ctx, ddl); err != nil {
		return fmt.Errorf("ago: storage: migrate: %w", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

func (db *database) CreateSession(ctx context.Context, session *Session) error {
	now := time.Now()
	if session.CreatedAt.IsZero() {
		session.CreatedAt = now
	}
	if session.UpdatedAt.IsZero() {
		session.UpdatedAt = now
	}
	_, err := db.pool.Exec(ctx,
		`INSERT INTO sessions (id, user_id, author, title, metadata, created_at, updated_at)
		 VALUES ($1, $2, $3, $4, $5, $6, $7)`,
		session.ID, session.UserID, session.Author, session.Title,
		nullJSON(session.Metadata), session.CreatedAt, session.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("ago: storage: create session: %w", err)
	}
	return nil
}

func (db *database) GetSession(ctx context.Context, id string) (*Session, error) {
	s := &Session{}
	err := db.pool.QueryRow(ctx,
		`SELECT id, user_id, author, title, metadata, created_at, updated_at, deleted_at
		 FROM sessions WHERE id = $1 AND deleted_at IS NULL`, id,
	).Scan(&s.ID, &s.UserID, &s.Author, &s.Title,
		&s.Metadata, &s.CreatedAt, &s.UpdatedAt, &s.DeletedAt)
	if err != nil {
		return nil, fmt.Errorf("ago: storage: get session: %w", err)
	}
	return s, nil
}

func (db *database) UpdateSession(ctx context.Context, session *Session) error {
	session.UpdatedAt = time.Now()
	_, err := db.pool.Exec(ctx,
		`UPDATE sessions SET user_id=$2, author=$3, title=$4, metadata=$5, updated_at=$6
		 WHERE id = $1`,
		session.ID, session.UserID, session.Author, session.Title,
		nullJSON(session.Metadata), session.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("ago: storage: update session: %w", err)
	}
	return nil
}

func (db *database) DeleteSession(ctx context.Context, id string) error {
	_, err := db.pool.Exec(ctx,
		`UPDATE sessions SET deleted_at = NOW() WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("ago: storage: delete session: %w", err)
	}
	return nil
}

func (db *database) ListSessions(ctx context.Context, userID string, limit, offset int) ([]*Session, error) {
	query := `SELECT id, user_id, author, title, metadata, created_at, updated_at
	          FROM sessions WHERE deleted_at IS NULL`
	args := []any{}
	idx := 1

	if userID != "" {
		query += fmt.Sprintf(" AND user_id = $%d", idx)
		args = append(args, userID)
		idx++
	}
	query += " ORDER BY created_at DESC"
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT $%d", idx)
		args = append(args, limit)
		idx++
	}
	if offset > 0 {
		query += fmt.Sprintf(" OFFSET $%d", idx)
		args = append(args, offset)
	}

	rows, err := db.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("ago: storage: list sessions: %w", err)
	}
	defer rows.Close()

	var sessions []*Session
	for rows.Next() {
		s := &Session{}
		if err := rows.Scan(&s.ID, &s.UserID, &s.Author, &s.Title,
			&s.Metadata, &s.CreatedAt, &s.UpdatedAt); err != nil {
			return nil, fmt.Errorf("ago: storage: scan session: %w", err)
		}
		sessions = append(sessions, s)
	}
	return sessions, rows.Err()
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

func (db *database) CreateEvent(ctx context.Context, event *Event) error {
	if event.CreatedAt.IsZero() {
		event.CreatedAt = time.Now()
	}
	_, err := db.pool.Exec(ctx,
		`INSERT INTO events (id, session_id, message_id, user_id, content, error, usage, created_at, metadata)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
		event.ID, event.SessionID, event.MessageID, event.UserID,
		event.Content, event.Error, nullJSON(event.Usage),
		event.CreatedAt, nullJSON(event.Metadata),
	)
	if err != nil {
		return fmt.Errorf("ago: storage: create event: %w", err)
	}
	return nil
}

// CreateEvents inserts multiple events in a single pgx batch — one network round trip.
func (db *database) CreateEvents(ctx context.Context, events []*Event) error {
	if len(events) == 0 {
		return nil
	}
	batch := &pgx.Batch{}
	now := time.Now()
	for _, event := range events {
		if event.CreatedAt.IsZero() {
			event.CreatedAt = now
		}
		batch.Queue(
			`INSERT INTO events (id, session_id, message_id, user_id, content, error, usage, created_at, metadata)
			 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
			event.ID, event.SessionID, event.MessageID, event.UserID,
			event.Content, event.Error, nullJSON(event.Usage),
			event.CreatedAt, nullJSON(event.Metadata),
		)
	}
	br := db.pool.SendBatch(ctx, batch)
	defer br.Close()
	for range events {
		if _, err := br.Exec(); err != nil {
			return fmt.Errorf("ago: storage: batch create events: %w", err)
		}
	}
	return nil
}

func (db *database) GetEvents(ctx context.Context, sessionID string) ([]*Event, error) {
	rows, err := db.pool.Query(ctx,
		`SELECT id, session_id, message_id, user_id, content, error, usage, created_at, metadata
		 FROM events WHERE session_id = $1 ORDER BY created_at ASC`, sessionID,
	)
	if err != nil {
		return nil, fmt.Errorf("ago: storage: get events: %w", err)
	}
	defer rows.Close()

	var events []*Event
	for rows.Next() {
		e := &Event{}
		if err := rows.Scan(&e.ID, &e.SessionID, &e.MessageID, &e.UserID,
			&e.Content, &e.Error, &e.Usage, &e.CreatedAt, &e.Metadata); err != nil {
			return nil, fmt.Errorf("ago: storage: scan event: %w", err)
		}
		events = append(events, e)
	}
	return events, rows.Err()
}

// GetRecentEvents returns the most recent N events ordered oldest→newest.
// limit <= 0 returns all events.
func (db *database) GetRecentEvents(ctx context.Context, sessionID string, limit int) ([]*Event, error) {
	var (
		rows pgx.Rows
		err  error
	)
	if limit > 0 {
		rows, err = db.pool.Query(ctx,
			`SELECT id, session_id, message_id, user_id, content, error, usage, created_at, metadata
			 FROM (
			   SELECT id, session_id, message_id, user_id, content, error, usage, created_at, metadata
			   FROM events WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2
			 ) sub ORDER BY created_at ASC`,
			sessionID, limit,
		)
	} else {
		rows, err = db.pool.Query(ctx,
			`SELECT id, session_id, message_id, user_id, content, error, usage, created_at, metadata
			 FROM events WHERE session_id = $1 ORDER BY created_at ASC`, sessionID,
		)
	}
	if err != nil {
		return nil, fmt.Errorf("ago: storage: get recent events: %w", err)
	}
	defer rows.Close()

	var events []*Event
	for rows.Next() {
		e := &Event{}
		if err := rows.Scan(&e.ID, &e.SessionID, &e.MessageID, &e.UserID,
			&e.Content, &e.Error, &e.Usage, &e.CreatedAt, &e.Metadata); err != nil {
			return nil, fmt.Errorf("ago: storage: scan event: %w", err)
		}
		events = append(events, e)
	}
	return events, rows.Err()
}

func (db *database) GetEvent(ctx context.Context, id string) (*Event, error) {
	e := &Event{}
	err := db.pool.QueryRow(ctx,
		`SELECT id, session_id, message_id, user_id, content, error, usage, created_at, metadata
		 FROM events WHERE id = $1`, id,
	).Scan(&e.ID, &e.SessionID, &e.MessageID, &e.UserID,
		&e.Content, &e.Error, &e.Usage, &e.CreatedAt, &e.Metadata)
	if err != nil {
		return nil, fmt.Errorf("ago: storage: get event: %w", err)
	}
	return e, nil
}

func (db *database) Close() error {
	db.pool.Close()
	return nil
}

func nullJSON(data json.RawMessage) any {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	return []byte(data)
}
