// Package plugins provides ready-made Hooks implementations for the ago
// agentic loop.
package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/Harshal1000/ago"
)

// LoggingHooks returns a *ago.Hooks that prints a structured, visual log of
// every agent's execution to w. Pass nil to write to os.Stderr.
func LoggingHooks(w io.Writer) *ago.Hooks {
	if w == nil {
		w = os.Stderr
	}
	p := &loggingPlugin{
		w:      w,
		states: make(map[string]*agentState),
	}
	return &ago.Hooks{
		BeforeLLMCall:  p.beforeLLM,
		AfterLLMCall:   p.afterLLM,
		BeforeToolCall: p.beforeTool,
		AfterToolCall:  p.afterTool,
		OnComplete:     p.onComplete,
	}
}

// ----------- internal -----------

// agentState holds per-agent timing and iteration state.
type agentState struct {
	iter     int
	runStart time.Time
	llmStart time.Time
}

type loggingPlugin struct {
	w      io.Writer
	mu     sync.Mutex             // serialises writes and protects states map
	states map[string]*agentState // keyed by agent name
	starts sync.Map               // map[callID]time.Time for tool timing
}

const (
	prefix      = "[ago] "
	maxArgWidth = 60
)

func (p *loggingPlugin) writef(format string, args ...any) {
	p.mu.Lock()
	fmt.Fprintf(p.w, prefix+format+"\n", args...)
	p.mu.Unlock()
}

// state returns (or lazily creates) the agentState for name. Must be called with mu held.
func (p *loggingPlugin) state(name string) *agentState {
	s := p.states[name]
	if s == nil {
		s = &agentState{}
		p.states[name] = s
	}
	return s
}

func agentName(ctx context.Context) string {
	rc := ago.GetRunContext(ctx)
	if rc.AgentName != "" {
		return rc.AgentName
	}
	return "agent"
}

func (p *loggingPlugin) beforeLLM(ctx context.Context, params *ago.GenerateParams) error {
	name := agentName(ctx)

	p.mu.Lock()
	s := p.state(name)
	s.iter++
	iter := s.iter
	if iter == 1 {
		s.runStart = time.Now()
	}
	s.llmStart = time.Now()
	p.mu.Unlock()

	bar := strings.Repeat("─", max(0, 44-len(name)-len(fmt.Sprintf("%d", iter))))
	p.writef("┌─ %s · turn %d %s", name, iter, bar)
	if iter == 1 {
		p.writef("│ User: %s", extractUserText(params.Contents))
	}
	p.writef("│ ▶  llm   %d messages", len(params.Contents))
	return nil
}

func (p *loggingPlugin) afterLLM(ctx context.Context, resp *ago.Response) {
	name := agentName(ctx)

	p.mu.Lock()
	var elapsed time.Duration
	if s := p.states[name]; s != nil {
		elapsed = time.Since(s.llmStart)
	}
	p.mu.Unlock()

	total := resp.Usage.TotalTokenCount
	in := resp.Usage.PromptTokenCount
	out := resp.Usage.CandidatesTokenCount
	p.writef("│ ◀  llm   %d tokens · %d in · %d out%s%s",
		total, in, out, strings.Repeat(" ", 5), fmtDuration(elapsed))
}

func (p *loggingPlugin) beforeTool(ctx context.Context, call *ago.FunctionCall) error {
	name := agentName(ctx)
	p.starts.Store(call.ID, [2]any{time.Now(), name})
	p.writef("│ [%s]   ◆ tool   %-12s %s", name, call.Name, fmtArgs(call.Args))
	return nil
}

func (p *loggingPlugin) afterTool(ctx context.Context, call *ago.FunctionCall, result *ago.ToolResult) {
	var elapsed time.Duration
	var name string
	if v, ok := p.starts.LoadAndDelete(call.ID); ok {
		pair := v.([2]any)
		elapsed = time.Since(pair[0].(time.Time))
		name, _ = pair[1].(string)
	}
	if name == "" {
		name = agentName(ctx)
	}
	if result.Error != nil {
		p.writef("│ [%s]   ✗ tool   %-12s %-*s  %s",
			name, call.Name, maxArgWidth, "error: "+result.Error.Error(), fmtDuration(elapsed))
		return
	}
	p.writef("│ [%s]   ✓ tool   %-12s %-*s  %s",
		name, call.Name, maxArgWidth, fmtResult(result.Response), fmtDuration(elapsed))
}

func (p *loggingPlugin) onComplete(ctx context.Context, result *ago.RunResult) {
	name := agentName(ctx)

	p.mu.Lock()
	var total time.Duration
	if s := p.states[name]; s != nil {
		total = time.Since(s.runStart)
		delete(p.states, name)
	}
	p.mu.Unlock()

	if text := ago.ExtractResponseText(result.Response); text != "" {
		p.writef("│ %s", truncateLines(text, 5, 120))
	}
	sessionPart := ""
	if result.SessionID != "" {
		sessionPart = fmt.Sprintf("session=%s · ", result.SessionID)
	}
	p.writef("└─ %s ✓  %shistory=%d · %s", name, sessionPart, len(result.History), fmtDuration(total))
	p.writef("")
}

// ----------- content helpers -----------

func extractUserText(contents []*ago.Content) string {
	for i := len(contents) - 1; i >= 0; i-- {
		c := contents[i]
		if c.Role != ago.RoleUser {
			continue
		}
		var parts []string
		for _, p := range c.Parts {
			if p.Text != "" {
				parts = append(parts, p.Text)
			}
		}
		if len(parts) > 0 {
			return truncate(strings.Join(parts, " "), 120)
		}
	}
	return "(no user message)"
}

// ----------- formatting helpers -----------

func fmtArgs(args map[string]any) string {
	if len(args) == 0 {
		return "(no args)"
	}
	parts := make([]string, 0, len(args))
	for k, v := range args {
		parts = append(parts, fmt.Sprintf("%s=%s", k, fmtValue(v)))
	}
	return truncate(strings.Join(parts, " · "), maxArgWidth)
}

func fmtResult(resp map[string]any) string {
	if len(resp) == 0 {
		return "{}"
	}
	b, err := json.Marshal(resp)
	if err != nil {
		return fmt.Sprintf("%v", resp)
	}
	return truncate(string(b), maxArgWidth)
}

func fmtValue(v any) string {
	switch val := v.(type) {
	case string:
		if len(val) > 30 {
			return fmt.Sprintf("%q", val[:30]+"...")
		}
		return fmt.Sprintf("%q", val)
	case float64:
		if val == float64(int64(val)) {
			return fmt.Sprintf("%d", int64(val))
		}
		return fmt.Sprintf("%g", val)
	default:
		return fmt.Sprintf("%v", val)
	}
}

func fmtDuration(d time.Duration) string {
	switch {
	case d < time.Millisecond:
		return fmt.Sprintf("%dµs", d.Microseconds())
	case d < time.Second:
		return fmt.Sprintf("%dms", d.Milliseconds())
	default:
		return fmt.Sprintf("%.2fs", d.Seconds())
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

// truncateLines returns the first maxLines non-empty lines of s, with each
// line capped at maxWidth characters.
func truncateLines(s string, maxLines, maxWidth int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	var out []string
	for _, l := range lines {
		if len(out) >= maxLines {
			out = append(out, "…")
			break
		}
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, truncate(l, maxWidth))
		}
	}
	return strings.Join(out, "\n│ ")
}
