package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/Harshal1000/ago"
)

// AgentTool wraps a Runner as a tool, enabling multi-agent architectures.
// The runner executes ephemerally — no storage, no history, current turn only.
type AgentTool struct {
	ToolName    string
	Description string
	Parameters  *ago.Schema // nil defaults to {"input": string}
	Runner      ago.Runner  // runner to execute
	ToolOptions ago.ToolOptions
}

// Name returns the tool name.
func (t *AgentTool) Name() string { return t.ToolName }

// Declaration returns the function declaration for LLM tool registration.
func (t *AgentTool) Declaration() *ago.FunctionDeclaration {
	params := t.Parameters
	if params == nil {
		params = &ago.Schema{
			Type: ago.TypeObject,
			Properties: map[string]*ago.Schema{
				"input": {Type: ago.TypeString, Description: "Input for the agent"},
			},
			Required: []string{"input"},
		}
	}
	return &ago.FunctionDeclaration{
		Name:        t.ToolName,
		Description: t.Description,
		Parameters:  params,
	}
}

// Execute runs the runner ephemerally (no storage, no history) and returns its text response.
func (t *AgentTool) Execute(ctx context.Context, args map[string]any) (*ago.ToolResult, error) {
	if t.Runner == nil {
		return nil, fmt.Errorf("ago: AgentTool %q has no Runner configured", t.ToolName)
	}

	var inputText string
	if input, ok := args["input"]; ok {
		inputText = fmt.Sprintf("%v", input)
	} else {
		var parts []string
		for k, v := range args {
			parts = append(parts, fmt.Sprintf("%s: %v", k, v))
		}
		inputText = strings.Join(parts, "\n")
	}

	result, err := ago.RunEphemeral(ctx, t.Runner, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, inputText),
	})
	if err != nil {
		return &ago.ToolResult{Error: err}, nil
	}

	return &ago.ToolResult{
		Response: map[string]any{"result": ago.ExtractText(result)},
	}, nil
}

// Options returns the tool's configuration options.
func (t *AgentTool) Options() ago.ToolOptions { return t.ToolOptions }

// Compile-time check.
var _ ago.Tool = (*AgentTool)(nil)
