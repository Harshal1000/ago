// Package storage provides persistent session and event storage for the Ago framework.
//
// Use New to create a Service. Pass nil or a Config with empty DatabaseURL
// to get in-memory storage; provide a DatabaseURL to connect to PostgreSQL.
//
//	svc, err := storage.New(ctx, nil)                          // in-memory
//	svc, err := storage.New(ctx, &storage.Config{              // postgres
//	    DatabaseURL: "postgres://localhost/ago",
//	    MaxConns:    20,
//	})
package storage

import (
	"context"
	"encoding/json"
	"time"
)

// Service is the interface for persisting sessions and events.
type Service interface {
	CreateSession(ctx context.Context, session *Session) error
	GetSession(ctx context.Context, id string) (*Session, error)
	UpdateSession(ctx context.Context, session *Session) error
	DeleteSession(ctx context.Context, id string) error
	ListSessions(ctx context.Context, userID string, limit, offset int) ([]*Session, error)

	CreateEvent(ctx context.Context, event *Event) error
	// CreateEvents inserts multiple events in a single round trip.
	CreateEvents(ctx context.Context, events []*Event) error
	GetEvents(ctx context.Context, sessionID string) ([]*Event, error)
	// GetRecentEvents returns the most recent N events ordered oldest→newest.
	// limit <= 0 returns all events.
	GetRecentEvents(ctx context.Context, sessionID string, limit int) ([]*Event, error)
	GetEvent(ctx context.Context, id string) (*Event, error)

	Close() error
}

// Config controls how the storage backend is initialized.
// All fields are optional — zero values use sensible defaults.
type Config struct {
	// DatabaseURL is the PostgreSQL connection string.
	// Empty means in-memory storage.
	DatabaseURL string

	// MaxConns is the maximum number of connections in the pool (default 10).
	MaxConns int32

	// MinConns is the minimum idle connections kept open (default 2).
	MinConns int32

	// Schema sets the PostgreSQL search_path (default "public").
	Schema string

	// AutoMigrate creates tables on startup (default true).
	AutoMigrate *bool
}

// autoMigrate returns the effective AutoMigrate value (defaults to true).
func (c *Config) autoMigrate() bool {
	if c == nil || c.AutoMigrate == nil {
		return true
	}
	return *c.AutoMigrate
}

// New creates a Service based on the given Config.
// If cfg is nil or cfg.DatabaseURL is empty, returns an in-memory Service.
func New(ctx context.Context, cfg *Config) (Service, error) {
	if cfg == nil || cfg.DatabaseURL == "" {
		return newInMemory(), nil
	}
	return newDatabase(ctx, cfg)
}

// Session represents a conversation or workflow.
type Session struct {
	ID        string          `json:"id"`
	UserID    string          `json:"user_id"`
	Author    string          `json:"author"`
	Title     string          `json:"title"`
	Metadata  json.RawMessage `json:"metadata,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
	UpdatedAt time.Time       `json:"updated_at"`
	DeletedAt *time.Time      `json:"deleted_at,omitempty"`
}

// Event represents a single turn within a session.
// Content stores the full message object (role + parts) as JSON.
type Event struct {
	ID        string          `json:"id"`
	SessionID string          `json:"session_id"`
	MessageID string          `json:"message_id"`
	UserID    string          `json:"user_id"`
	Content   json.RawMessage `json:"content"`
	Error     string          `json:"error,omitempty"`
	Usage     json.RawMessage `json:"usage,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
	Metadata  json.RawMessage `json:"metadata,omitempty"`
}
