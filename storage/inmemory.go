package storage

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type inMemory struct {
	mu       sync.RWMutex
	sessions map[string]*Session
	events   map[string][]*Event // sessionID → events
	eventIdx map[string]*Event   // eventID → event
}

func newInMemory() *inMemory {
	return &inMemory{
		sessions: make(map[string]*Session),
		events:   make(map[string][]*Event),
		eventIdx: make(map[string]*Event),
	}
}

var _ Service = (*inMemory)(nil)

func (s *inMemory) CreateSession(ctx context.Context, session *Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.sessions[session.ID]; exists {
		return fmt.Errorf("ago: session %q already exists", session.ID)
	}
	now := time.Now()
	if session.CreatedAt.IsZero() {
		session.CreatedAt = now
	}
	if session.UpdatedAt.IsZero() {
		session.UpdatedAt = now
	}
	cp := *session
	s.sessions[session.ID] = &cp
	return nil
}

func (s *inMemory) GetSession(ctx context.Context, id string) (*Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	sess, ok := s.sessions[id]
	if !ok || sess.DeletedAt != nil {
		return nil, fmt.Errorf("ago: session %q not found", id)
	}
	cp := *sess
	return &cp, nil
}

func (s *inMemory) UpdateSession(ctx context.Context, session *Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.sessions[session.ID]; !ok {
		return fmt.Errorf("ago: session %q not found", session.ID)
	}
	session.UpdatedAt = time.Now()
	cp := *session
	s.sessions[session.ID] = &cp
	return nil
}

func (s *inMemory) DeleteSession(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[id]
	if !ok {
		return fmt.Errorf("ago: session %q not found", id)
	}
	now := time.Now()
	sess.DeletedAt = &now
	return nil
}

func (s *inMemory) ListSessions(ctx context.Context, userID string, limit, offset int) ([]*Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var result []*Session
	for _, sess := range s.sessions {
		if sess.DeletedAt != nil {
			continue
		}
		if userID != "" && sess.UserID != userID {
			continue
		}
		cp := *sess
		result = append(result, &cp)
	}

	if offset >= len(result) {
		return nil, nil
	}
	result = result[offset:]
	if limit > 0 && limit < len(result) {
		result = result[:limit]
	}
	return result, nil
}

func (s *inMemory) CreateEvent(ctx context.Context, event *Event) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if event.CreatedAt.IsZero() {
		event.CreatedAt = time.Now()
	}
	cp := *event
	s.events[event.SessionID] = append(s.events[event.SessionID], &cp)
	s.eventIdx[event.ID] = &cp
	return nil
}

func (s *inMemory) CreateEvents(ctx context.Context, events []*Event) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for _, event := range events {
		if event.CreatedAt.IsZero() {
			event.CreatedAt = now
		}
		cp := *event
		s.events[event.SessionID] = append(s.events[event.SessionID], &cp)
		s.eventIdx[event.ID] = &cp
	}
	return nil
}

func (s *inMemory) GetEvents(ctx context.Context, sessionID string) ([]*Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	events := s.events[sessionID]
	result := make([]*Event, len(events))
	for i, e := range events {
		cp := *e
		result[i] = &cp
	}
	return result, nil
}

func (s *inMemory) GetRecentEvents(ctx context.Context, sessionID string, limit int) ([]*Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	events := s.events[sessionID]
	if limit > 0 && len(events) > limit {
		events = events[len(events)-limit:]
	}
	result := make([]*Event, len(events))
	for i, e := range events {
		cp := *e
		result[i] = &cp
	}
	return result, nil
}

func (s *inMemory) GetEvent(ctx context.Context, id string) (*Event, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.eventIdx[id]
	if !ok {
		return nil, fmt.Errorf("ago: event %q not found", id)
	}
	cp := *e
	return &cp, nil
}

func (s *inMemory) Close() error { return nil }
