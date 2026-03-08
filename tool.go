package ago

import "context"

// ToolResult holds the outcome of a tool execution.
// Response is sent back to the model as FunctionResponse data.
// Error is a tool-level error (sent to the model as {"error": "..."}, loop continues).
type ToolResult struct {
	Response map[string]any
	Error    error
}

// ToolOptions configures tool behavior within the executor loop.
type ToolOptions struct {
	// SkipSynthesis, when true, causes the executor to return the tool result
	// directly without an additional LLM synthesis turn.
	SkipSynthesis bool

	// IsAgentCall, when true, marks this tool as an agent-transfer call.
	// The executor stores its call/response as RoleAgent events in storage
	// while keeping RoleTool in LLM history (the LLM always sees role=tool).
	IsAgentCall bool
}

// Tool is the interface for all tools usable by the executor.
type Tool interface {
	Name() string
	Declaration() *FunctionDeclaration
	Execute(ctx context.Context, args map[string]any) (*ToolResult, error)
	Options() ToolOptions
}
