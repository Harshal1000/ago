package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/Harshal1000/ago"
)

// AgentTool wraps another agent as a tool. The sub-agent is looked up
// from the registry by AgentName at execution time, avoiding import cycles.
type AgentTool struct {
	ToolName    string
	Description string
	Parameters  *ago.Schema // nil defaults to {"input": string}
	AgentName   string      // agent name to look up at execution time
	ToolOptions ago.ToolOptions

	// RunFunc is the function used to run the sub-agent. Defaults to nil;
	// must be set before execution (typically to ago.Run via the wiring layer).
	RunFunc func(ctx context.Context, agentName string, contents []*ago.Content) (*ago.ExecutorResult, error)
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

// Execute runs the sub-agent by looking it up from the registry and calling RunFunc.
func (t *AgentTool) Execute(ctx context.Context, args map[string]any) (*ago.ToolResult, error) {
	if t.RunFunc == nil {
		return nil, fmt.Errorf("ago: AgentTool %q has no RunFunc configured", t.ToolName)
	}

	// Build user content from args.
	var inputText string
	if input, ok := args["input"]; ok {
		inputText = fmt.Sprintf("%v", input)
	} else {
		// Serialize all args as the input.
		var parts []string
		for k, v := range args {
			parts = append(parts, fmt.Sprintf("%s: %v", k, v))
		}
		inputText = strings.Join(parts, "\n")
	}

	contents := []*ago.Content{
		ago.NewTextContent(ago.RoleUser, inputText),
	}

	result, err := t.RunFunc(ctx, t.AgentName, contents)
	if err != nil {
		// Tool-level error — send to model, loop continues.
		return &ago.ToolResult{Error: err}, nil
	}

	// Extract text from the response.
	responseText := extractResponseText(result.Response)
	return &ago.ToolResult{
		Response: map[string]any{"result": responseText},
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
