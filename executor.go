package ago

import (
	"context"
	"fmt"
	"iter"
)

// DefaultMaxIterations is the default maximum number of agentic loop iterations.
const DefaultMaxIterations = 10

// AgentConfig is the interface the executor needs from an agent.
type AgentConfig interface {
	GetName() string
	GetModel() string
	GetLLM() LLM
	GetTools() []Tool
	GetMaxIterations() int
	GetGenerateConfig() *GenerateConfig
	GetSystemInstruction() *Content
}

// run executes the agentic loop synchronously.
func run(ctx context.Context, app *App, agent AgentConfig, contents []*Content, opts *RunOptions) (*RunResult, error) {
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
	toolDecls := buildToolDecls(tools)
	config := agent.GetGenerateConfig()
	sysInstruction := agent.GetSystemInstruction()
	maxIter := maxIterations(agent)

	var hooks *Hooks
	if app != nil {
		hooks = app.Hooks
	}

	rec := newRecorder(app, agent.GetName(), opts)
	includeHistory := app != nil && app.IncludeHistory
	history, err := initHistory(ctx, rec, contents, includeHistory)
	if err != nil {
		return nil, err
	}
	if rec != nil {
		defer rec.Flush(ctx)
	}

	for i := 0; i < maxIter; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		params := &GenerateParams{
			Contents:          history,
			Config:            config,
			SystemInstruction: sysInstruction,
			Tools:             toolDecls,
		}
		if hooks != nil && hooks.BeforeLLMCall != nil {
			if err := hooks.BeforeLLMCall(ctx, params); err != nil {
				return nil, err
			}
		}

		resp, err := llm.Generate(ctx, model, params)
		if err != nil {
			return nil, fmt.Errorf("ago: generate: %w", err)
		}

		if hooks != nil && hooks.AfterLLMCall != nil {
			hooks.AfterLLMCall(ctx, resp)
		}

		calls := extractFunctionCalls(resp)
		if len(calls) == 0 {
			if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
				mc := resp.Candidates[0].Content
				history = append(history, mc)
				if rec != nil {
					rec.BufferWithUsage(mc, &resp.Usage)
				}
			}
			result := &RunResult{Response: resp, History: history, SessionID: rec.SessionID()}
			if hooks != nil && hooks.OnComplete != nil {
				hooks.OnComplete(ctx, result)
			}
			return result, nil
		}

		callContent := NewFunctionCallContent(calls...)
		history = append(history, callContent)
		if rec != nil {
			rec.Buffer(callContent)
		}

		results, infraErr := executeToolsParallel(ctx, calls, toolMap, hooks)
		if infraErr != nil {
			return nil, infraErr
		}

		funcResponses, allSkip := buildFuncResponses(calls, results, toolMap)
		respContent := NewFunctionResponseContent(funcResponses...)
		history = append(history, respContent)
		if rec != nil {
			rec.Buffer(respContent)
		}

		if allSkip && len(calls) > 0 {
			result := &RunResult{
				Response:  synthesizeToolResponse(funcResponses),
				History:   history,
				SessionID: rec.SessionID(),
			}
			if hooks != nil && hooks.OnComplete != nil {
				hooks.OnComplete(ctx, result)
			}
			return result, nil
		}
	}

	return nil, fmt.Errorf("ago: agent %q exceeded max iterations (%d)", agent.GetName(), maxIter)
}

// runSSE executes the agentic loop with streaming, yielding chunks to the caller.
func runSSE(ctx context.Context, app *App, agent AgentConfig, contents []*Content, opts *RunOptions) iter.Seq2[*StreamChunk, error] {
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
		toolDecls := buildToolDecls(tools)
		config := agent.GetGenerateConfig()
		sysInstruction := agent.GetSystemInstruction()
		maxIter := maxIterations(agent)

		var hooks *Hooks
		if app != nil {
			hooks = app.Hooks
		}

		rec := newRecorder(app, agent.GetName(), opts)
		includeHistory := app != nil && app.IncludeHistory
		history, err := initHistory(ctx, rec, contents, includeHistory)
		if err != nil {
			yield(nil, err)
			return
		}
		if rec != nil {
			defer rec.Flush(ctx)
		}

		for i := 0; i < maxIter; i++ {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}

			params := &GenerateParams{
				Contents:          history,
				Config:            config,
				SystemInstruction: sysInstruction,
				Tools:             toolDecls,
			}
			if hooks != nil && hooks.BeforeLLMCall != nil {
				if err := hooks.BeforeLLMCall(ctx, params); err != nil {
					yield(nil, err)
					return
				}
			}

			var lastChunk *StreamChunk
			var streamErr error
			for chunk, err := range llm.GenerateStream(ctx, model, params) {
				if err != nil {
					yield(nil, fmt.Errorf("ago: generate stream: %w", err))
					return
				}
				lastChunk = chunk
				streamErr = err
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

			calls := extractFunctionCallsFromChunk(lastChunk)
			if len(calls) == 0 {
				if rec != nil && len(lastChunk.Candidates) > 0 && lastChunk.Candidates[0].Content != nil {
					rec.BufferWithUsage(lastChunk.Candidates[0].Content, lastChunk.Usage)
				}
				if hooks != nil && hooks.OnComplete != nil {
					hooks.OnComplete(ctx, &RunResult{History: history, SessionID: rec.SessionID()})
				}
				yield(lastChunk, nil)
				return
			}

			callContent := NewFunctionCallContent(calls...)
			history = append(history, callContent)
			if rec != nil {
				rec.Buffer(callContent)
			}

			results, infraErr := executeToolsParallel(ctx, calls, toolMap, hooks)
			if infraErr != nil {
				yield(nil, infraErr)
				return
			}

			funcResponses, allSkip := buildFuncResponses(calls, results, toolMap)
			respContent := NewFunctionResponseContent(funcResponses...)
			history = append(history, respContent)
			if rec != nil {
				rec.Buffer(respContent)
			}

			if allSkip && len(calls) > 0 {
				if hooks != nil && hooks.OnComplete != nil {
					hooks.OnComplete(ctx, &RunResult{History: history, SessionID: rec.SessionID()})
				}
				yield(&StreamChunk{
					Candidates: synthesizeToolResponse(funcResponses).Candidates,
					Complete:   true,
				}, nil)
				return
			}
		}

		yield(nil, fmt.Errorf("ago: agent %q exceeded max iterations (%d)", agent.GetName(), maxIter))
	}
}
