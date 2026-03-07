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
// the entire agentic loop to w. Pass nil to write to os.Stderr.
//
// It covers every observable event in one place:
//
//	[ago] User: My name is Harshal. What is 25 * 4?
//	[ago]
//	[ago] ─── turn 1 ───────────────────────────────────
//	[ago] ▶  llm      3 messages
//	[ago]    ◆ tool   calculator   operation="multiply" · a=25 · b=4
//	[ago]    ✓ tool   calculator   {"result":100}                       12ms
//	[ago] ◀  llm      142 tokens · 89 in · 53 out                     234ms
//	[ago]
//	[ago] ─── turn 2 ───────────────────────────────────
//	[ago] ▶  llm      5 messages
//	[ago] ◀  llm      87 tokens · 67 in · 20 out                      189ms
//	[ago]
//	[ago] Agent: 25 × 4 = 100
//	[ago] ✓  done     session=abc123 · history=9 · 0.42s
func LoggingHooks(w io.Writer) *ago.Hooks {
	if w == nil {
		w = os.Stderr
	}
	p := &loggingPlugin{w: w}
	return &ago.Hooks{
		BeforeLLMCall:  p.beforeLLM,
		AfterLLMCall:   p.afterLLM,
		BeforeToolCall: p.beforeTool,
		AfterToolCall:  p.afterTool,
		OnComplete:     p.onComplete,
	}
}

// ----------- internal -----------

type loggingPlugin struct {
	w        io.Writer
	mu       sync.Mutex // serialises writes and protects timing fields
	iter     int
	runStart time.Time
	llmStart time.Time
	starts   sync.Map // map[string]time.Time keyed by FunctionCall.ID (tool timing)
}

const (
	prefix      = "[ago] "
	separator   = "────────────────────────────────"
	maxArgWidth = 60
)

func (p *loggingPlugin) writef(format string, args ...any) {
	p.mu.Lock()
	fmt.Fprintf(p.w, prefix+format+"\n", args...)
	p.mu.Unlock()
}

func (p *loggingPlugin) beforeLLM(_ context.Context, params *ago.GenerateParams) error {
	p.mu.Lock()
	p.iter++
	iter := p.iter
	if iter == 1 {
		p.runStart = time.Now()
	}
	p.llmStart = time.Now()
	p.mu.Unlock()

	if iter == 1 {
		p.writef("User: %s", extractUserText(params.Contents))
		p.writef("")
	}
	p.writef("─── turn %d %s", iter, separator)
	p.writef("▶  llm      %d messages", len(params.Contents))
	return nil
}

func (p *loggingPlugin) afterLLM(_ context.Context, resp *ago.Response) {
	p.mu.Lock()
	elapsed := time.Since(p.llmStart)
	p.mu.Unlock()

	total := resp.Usage.TotalTokenCount
	in := resp.Usage.PromptTokenCount
	out := resp.Usage.CandidatesTokenCount
	p.writef("◀  llm      %d tokens · %d in · %d out%s%s",
		total, in, out, strings.Repeat(" ", 5), fmtDuration(elapsed))
	p.writef("")
}

func (p *loggingPlugin) beforeTool(_ context.Context, call *ago.FunctionCall) error {
	p.starts.Store(call.ID, time.Now())
	p.writef("   ◆ tool   %-12s %s", call.Name, fmtArgs(call.Args))
	return nil
}

func (p *loggingPlugin) afterTool(_ context.Context, call *ago.FunctionCall, result *ago.ToolResult) {
	var elapsed time.Duration
	if t, ok := p.starts.LoadAndDelete(call.ID); ok {
		elapsed = time.Since(t.(time.Time))
	}
	if result.Error != nil {
		p.writef("   ✗ tool   %-12s %-*s  %s",
			call.Name, maxArgWidth, "error: "+result.Error.Error(), fmtDuration(elapsed))
		return
	}
	p.writef("   ✓ tool   %-12s %-*s  %s",
		call.Name, maxArgWidth, fmtResult(result.Response), fmtDuration(elapsed))
}

func (p *loggingPlugin) onComplete(_ context.Context, result *ago.RunResult) {
	p.mu.Lock()
	total := time.Since(p.runStart)
	p.iter = 0 // reset for next run on this plugin instance
	p.mu.Unlock()

	if text := extractResponseText(result.Response); text != "" {
		p.writef("Agent: %s", text)
	}
	p.writef("✓  done     session=%s · history=%d · %s",
		result.SessionID, len(result.History), fmtDuration(total))
	p.writef("═══════════════════════════════════════════════════")
}

// ----------- content helpers -----------

// extractUserText returns the text of the last user-role content in contents.
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

// extractResponseText returns the plain text from a Response, or empty string.
func extractResponseText(resp *ago.Response) string {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return ""
	}
	var parts []string
	for _, p := range resp.Candidates[0].Content.Parts {
		if p.Text != "" {
			parts = append(parts, p.Text)
		}
	}
	return strings.Join(parts, "")
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
