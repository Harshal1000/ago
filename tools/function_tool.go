package tools

import (
	"context"

	"github.com/Harshal1000/ago"
)

// ToolFunc is the signature for a function that implements tool logic.
type ToolFunc func(ctx context.Context, args map[string]any) (map[string]any, error)

// FunctionTool wraps a Go function as an ago.Tool.
type FunctionTool struct {
	ToolName    string
	Description string
	Parameters  *ago.Schema
	Fn          ToolFunc
	ToolOptions ago.ToolOptions
}

// Name returns the tool name.
func (t *FunctionTool) Name() string { return t.ToolName }

// Declaration returns the function declaration for LLM tool registration.
func (t *FunctionTool) Declaration() *ago.FunctionDeclaration {
	return &ago.FunctionDeclaration{
		Name:        t.ToolName,
		Description: t.Description,
		Parameters:  t.Parameters,
	}
}

// Execute runs the wrapped function and returns a ToolResult.
// If Fn returns an error, it becomes ToolResult.Error (sent to model, loop continues).
// A nil Fn returns an empty response.
func (t *FunctionTool) Execute(ctx context.Context, args map[string]any) (*ago.ToolResult, error) {
	if t.Fn == nil {
		return &ago.ToolResult{Response: map[string]any{}}, nil
	}
	resp, err := t.Fn(ctx, args)
	if err != nil {
		return &ago.ToolResult{Error: err}, nil
	}
	return &ago.ToolResult{Response: resp}, nil
}

// Options returns the tool's configuration options.
func (t *FunctionTool) Options() ago.ToolOptions { return t.ToolOptions }

// Compile-time check.
var _ ago.Tool = (*FunctionTool)(nil)
