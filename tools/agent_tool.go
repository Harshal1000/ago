package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/Harshal1000/ago"
)

// AgentTool wraps another agent as a tool, enabling multi-agent architectures.
// The sub-agent runs ephemerally — no storage, no history, current turn only.
// Assign a cheaper/smaller model to Agent to reduce cost for delegatable subtasks.
type AgentTool struct {
	ToolName    string
	Description string
	Parameters  *ago.Schema     // nil defaults to {"input": string}
	Agent       ago.AgentConfig // sub-agent to run
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

// Execute runs the sub-agent ephemerally (no storage, no history) and returns its text response.
func (t *AgentTool) Execute(ctx context.Context, args map[string]any) (*ago.ToolResult, error) {
	if t.Agent == nil {
		return nil, fmt.Errorf("ago: AgentTool %q has no Agent configured", t.ToolName)
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

	result, err := ago.RunEphemeral(ctx, t.Agent, []*ago.Content{
		ago.NewTextContent(ago.RoleUser, inputText),
	})
	if err != nil {
		// Tool-level error — sent to model as {"error": "..."}, loop continues.
		return &ago.ToolResult{Error: err}, nil
	}

	return &ago.ToolResult{
		Response: map[string]any{"result": extractResponseText(result.Response)},
	}, nil
}

// Options returns the tool's configuration options.
func (t *AgentTool) Options() ago.ToolOptions { return t.ToolOptions }

func extractResponseText(resp *ago.Response) string {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return ""
	}
	var texts []string
	for _, p := range resp.Candidates[0].Content.Parts {
		if p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// Compile-time check.
var _ ago.Tool = (*AgentTool)(nil)
