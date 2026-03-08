package ago

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

func maxIterations(agent AgentConfig) int {
	n := agent.GetMaxIterations()
	if n <= 0 {
		return DefaultMaxIterations
	}
	return n
}

func buildToolMap(tools []Tool) map[string]Tool {
	m := make(map[string]Tool, len(tools))
	for _, t := range tools {
		m[t.Name()] = t
	}
	return m
}

func buildToolDecls(tools []Tool) []*FunctionDeclaration {
	if len(tools) == 0 {
		return nil
	}
	decls := make([]*FunctionDeclaration, len(tools))
	for i, t := range tools {
		decls[i] = t.Declaration()
	}
	return decls
}

func extractFunctionCalls(resp *Response) []*FunctionCall {
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return nil
	}
	var calls []*FunctionCall
	for _, p := range resp.Candidates[0].Content.Parts {
		if p.FunctionCall != nil {
			calls = append(calls, p.FunctionCall)
		}
	}
	return calls
}

func extractFunctionCallsFromChunk(chunk *StreamChunk) []*FunctionCall {
	if len(chunk.Candidates) == 0 || chunk.Candidates[0].Content == nil {
		return nil
	}
	var calls []*FunctionCall
	for _, p := range chunk.Candidates[0].Content.Parts {
		if p.FunctionCall != nil {
			calls = append(calls, p.FunctionCall)
		}
	}
	return calls
}

func buildFuncResponses(calls []*FunctionCall, results []*ToolResult, toolMap map[string]Tool) ([]*FunctionResponse, bool) {
	funcResponses := make([]*FunctionResponse, 0, len(calls))
	allSkip := true
	for j, call := range calls {
		tr := results[j]
		respData := tr.Response
		if tr.Error != nil {
			respData = map[string]any{"error": tr.Error.Error()}
		}
		funcResponses = append(funcResponses, &FunctionResponse{
			ID:       call.ID,
			Name:     call.Name,
			Response: respData,
		})
		if t, ok := toolMap[call.Name]; ok {
			if !t.Options().SkipSynthesis {
				allSkip = false
			}
		}
	}
	return funcResponses, allSkip
}

type toolExecResult struct {
	index  int
	result *ToolResult
	err    error
}

func executeToolsParallel(ctx context.Context, calls []*FunctionCall, toolMap map[string]Tool, hooks *Hooks) ([]*ToolResult, error) {
	results := make([]*ToolResult, len(calls))
	ch := make(chan toolExecResult, len(calls))
	var wg sync.WaitGroup

	for i, call := range calls {
		tool, ok := toolMap[call.Name]
		if !ok {
			results[i] = &ToolResult{Error: fmt.Errorf("unknown tool %q", call.Name)}
			continue
		}
		wg.Add(1)
		go func(idx int, t Tool, c *FunctionCall) {
			defer wg.Done()
			if hooks != nil && hooks.BeforeToolCall != nil {
				if err := hooks.BeforeToolCall(ctx, c); err != nil {
					ch <- toolExecResult{index: idx, err: err}
					return
				}
			}
			tr, err := t.Execute(ctx, c.Args)
			if err == nil && hooks != nil && hooks.AfterToolCall != nil {
				hooks.AfterToolCall(ctx, c, tr)
			}
			ch <- toolExecResult{index: idx, result: tr, err: err}
		}(i, tool, call)
	}

	go func() {
		wg.Wait()
		close(ch)
	}()

	for res := range ch {
		if res.err != nil {
			return nil, fmt.Errorf("ago: tool %q infrastructure error: %w", calls[res.index].Name, res.err)
		}
		results[res.index] = res.result
	}
	return results, nil
}

func synthesizeToolResponse(responses []*FunctionResponse) *Response {
	var parts []string
	for _, r := range responses {
		data, _ := json.Marshal(r.Response)
		parts = append(parts, fmt.Sprintf("[%s]: %s", r.Name, string(data)))
	}
	return &Response{
		Candidates: []*Candidate{{
			Content:      NewTextContent(RoleModel, strings.Join(parts, "\n")),
			FinishReason: FinishReasonStop,
		}},
	}
}
