package ago

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"
	"sync"
)

// DefaultMaxIterations is the default maximum number of agentic loop iterations.
const DefaultMaxIterations = 10

// ExecutorResult holds the outcome of an executor run.
type ExecutorResult struct {
	Response *Response  // final LLM response (or synthetic response for SkipSynthesis)
	History  []*Content // full conversation including tool calls/responses
}

// AgentConfig is the interface the executor needs from an agent.
// This avoids importing the agent package from the root package.
type AgentConfig interface {
	GetName() string
	GetModel() string
	GetLLM() LLM
	GetTools() []Tool
	GetMaxIterations() int
	GetGenerateConfig() *GenerateConfig
	GetSystemInstruction() *Content
}

// Run executes the agentic loop for the given agent config synchronously.
func Run(ctx context.Context, agent AgentConfig, contents []*Content) (*ExecutorResult, error) {
	llm := agent.GetLLM()
	if llm == nil {
		return nil, fmt.Errorf("ago: agent %q has no LLM configured", agent.GetName())
	}

	model := agent.GetModel()
	if model == "" {
		return nil, fmt.Errorf("ago: agent %q has no Model configured", agent.GetName())
	}

	tools := agent.GetTools()
	toolMap := buildToolMap(tools)
	config := agent.GetGenerateConfig()
	sysInstruction := agent.GetSystemInstruction()

	// Build tool declarations for the LLM.
	var toolDecls []*FunctionDeclaration
	if len(tools) > 0 {
		toolDecls = make([]*FunctionDeclaration, 0, len(tools))
		for _, t := range tools {
			toolDecls = append(toolDecls, t.Declaration())
		}
	}

	maxIter := agent.GetMaxIterations()
	if maxIter <= 0 {
		maxIter = DefaultMaxIterations
	}

	history := make([]*Content, len(contents))
	copy(history, contents)

	for i := 0; i < maxIter; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		resp, err := llm.Generate(ctx, model, &GenerateParams{
			Contents:          history,
			Config:            config,
			SystemInstruction: sysInstruction,
			Tools:             toolDecls,
		})
		if err != nil {
			return nil, fmt.Errorf("ago: generate: %w", err)
		}

		// Extract function calls from response.
		calls := extractFunctionCalls(resp)
		if len(calls) == 0 {
			// No tool calls — we're done.
			if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
				history = append(history, resp.Candidates[0].Content)
			}
			return &ExecutorResult{Response: resp, History: history}, nil
		}

		// Append model's tool-call turn to history.
		history = append(history, NewFunctionCallContent(calls...))

		// Execute tools in parallel.
		results, infraErr := executeToolsParallel(ctx, calls, toolMap)
		if infraErr != nil {
			return nil, infraErr
		}

		// Build function responses and append to history.
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
			// Check SkipSynthesis.
			if t, ok := toolMap[call.Name]; ok {
				if !t.Options().SkipSynthesis {
					allSkip = false
				}
			}
		}
		history = append(history, NewFunctionResponseContent(funcResponses...))

		// If all tools have SkipSynthesis, return directly.
		if allSkip && len(calls) > 0 {
			return &ExecutorResult{
				Response: synthesizeToolResponse(funcResponses),
				History:  history,
			}, nil
		}
	}

	return nil, fmt.Errorf("ago: agent %q exceeded max iterations (%d)", agent.GetName(), maxIter)
}

// RunSSE executes the agentic loop with streaming, yielding chunks to the caller.
func RunSSE(ctx context.Context, agent AgentConfig, contents []*Content) iter.Seq2[*StreamChunk, error] {
	return func(yield func(*StreamChunk, error) bool) {
		llm := agent.GetLLM()
		if llm == nil {
			yield(nil, fmt.Errorf("ago: agent %q has no LLM configured", agent.GetName()))
			return
		}

		model := agent.GetModel()
		if model == "" {
			yield(nil, fmt.Errorf("ago: agent %q has no Model configured", agent.GetName()))
			return
		}

		tools := agent.GetTools()
		toolMap := buildToolMap(tools)
		config := agent.GetGenerateConfig()
		sysInstruction := agent.GetSystemInstruction()

		// Build tool declarations for the LLM.
		var toolDecls []*FunctionDeclaration
		if len(tools) > 0 {
			toolDecls = make([]*FunctionDeclaration, 0, len(tools))
			for _, t := range tools {
				toolDecls = append(toolDecls, t.Declaration())
			}
		}

		maxIter := agent.GetMaxIterations()
		if maxIter <= 0 {
			maxIter = DefaultMaxIterations
		}

		history := make([]*Content, len(contents))
		copy(history, contents)

		for i := 0; i < maxIter; i++ {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}

			// Stream the LLM response, collecting chunks.
			var lastChunk *StreamChunk
			var streamErr error
			for chunk, err := range llm.GenerateStream(ctx, model, &GenerateParams{
				Contents:          history,
				Config:            config,
				SystemInstruction: sysInstruction,
				Tools:             toolDecls,
			}) {
				if err != nil {
					yield(nil, fmt.Errorf("ago: generate stream: %w", err))
					return
				}
				lastChunk = chunk
				streamErr = err

				// Don't yield the "complete" chunk yet — we may need to continue the loop.
				if !chunk.Complete {
					if !yield(chunk, nil) {
						return
					}
				}
			}

			if streamErr != nil {
				yield(nil, fmt.Errorf("ago: generate stream: %w", streamErr))
				return
			}

			if lastChunk == nil {
				yield(nil, fmt.Errorf("ago: empty stream response"))
				return
			}

			// Extract function calls from the streamed response.
			calls := extractFunctionCallsFromChunk(lastChunk)
			if len(calls) == 0 {
				// No tool calls — yield final chunk and we're done.
				if !yield(lastChunk, nil) {
					return
				}
				return
			}

			// Append model's tool-call turn to history.
			history = append(history, NewFunctionCallContent(calls...))

			// Execute tools.
			results, infraErr := executeToolsParallel(ctx, calls, toolMap)
			if infraErr != nil {
				yield(nil, infraErr)
				return
			}

			// Build function responses.
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
			history = append(history, NewFunctionResponseContent(funcResponses...))

			// If all tools skip synthesis, yield a final chunk with tool results.
			if allSkip && len(calls) > 0 {
				synth := synthesizeToolResponse(funcResponses)
				finalChunk := &StreamChunk{
					Candidates: synth.Candidates,
					Complete:   true,
				}
				yield(finalChunk, nil)
				return
			}
			// Otherwise loop back for next LLM call.
		}

		yield(nil, fmt.Errorf("ago: agent %q exceeded max iterations (%d)", agent.GetName(), maxIter))
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func buildToolMap(tools []Tool) map[string]Tool {
	m := make(map[string]Tool, len(tools))
	for _, t := range tools {
		m[t.Name()] = t
	}
	return m
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

type toolExecResult struct {
	index  int
	result *ToolResult
	err    error
}

func executeToolsParallel(ctx context.Context, calls []*FunctionCall, toolMap map[string]Tool) ([]*ToolResult, error) {
	results := make([]*ToolResult, len(calls))
	ch := make(chan toolExecResult, len(calls))
	var wg sync.WaitGroup

	for i, call := range calls {
		tool, ok := toolMap[call.Name]
		if !ok {
			// Unknown tool — send error to model, don't stop the loop.
			results[i] = &ToolResult{
				Error: fmt.Errorf("unknown tool %q", call.Name),
			}
			continue
		}

		wg.Add(1)
		go func(idx int, t Tool, args map[string]any) {
			defer wg.Done()
			tr, err := t.Execute(ctx, args)
			ch <- toolExecResult{index: idx, result: tr, err: err}
		}(i, tool, call.Args)
	}

	// Close channel when all goroutines complete.
	go func() {
		wg.Wait()
		close(ch)
	}()

	for res := range ch {
		if res.err != nil {
			// Infrastructure error — stop the loop.
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
	text := strings.Join(parts, "\n")
	return &Response{
		Candidates: []*Candidate{{
			Content:      NewTextContent(RoleModel, text),
			FinishReason: FinishReasonStop,
		}},
	}
}
