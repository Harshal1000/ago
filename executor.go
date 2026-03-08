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

// AgentLoop runs the agentic loop synchronously. Called by agent.Agent.Run().
func AgentLoop(ctx context.Context, agent AgentConfig, contents []*Content) (*RunResult, error) {
	rc := GetRunContext(ctx)
	subRC := &RunContext{
		AgentName: agent.GetName(),
		SessionID: rc.SessionID,
		UserID:    rc.UserID,
		Hooks:     rc.Hooks,
	}
	ctx = WithRunContext(ctx, subRC)

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
	hooks := subRC.Hooks

	history := make([]*Content, len(contents))
	copy(history, contents)

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
				history = append(history, resp.Candidates[0].Content)
			}
			result := &RunResult{Response: resp, History: history}
			if hooks != nil && hooks.OnComplete != nil {
				hooks.OnComplete(ctx, result)
			}
			return result, nil
		}

		callContent := NewFunctionCallContent(calls...)
		history = append(history, callContent)

		funcResponses, allSkip, infraErr := loopStep(ctx, calls, toolMap, hooks)
		if infraErr != nil {
			return nil, infraErr
		}

		respContent := NewFunctionResponseContent(funcResponses...)
		history = append(history, respContent)

		if allSkip && len(calls) > 0 {
			result := &RunResult{
				Response: synthesizeToolResponse(funcResponses),
				History:  history,
			}
			if hooks != nil && hooks.OnComplete != nil {
				hooks.OnComplete(ctx, result)
			}
			return result, nil
		}
	}

	return nil, fmt.Errorf("ago: agent %q exceeded max iterations (%d)", agent.GetName(), maxIter)
}

// AgentLoopStream runs the agentic loop with streaming. Called by agent.Agent.RunStream().
func AgentLoopStream(ctx context.Context, agent AgentConfig, contents []*Content) iter.Seq2[*StreamChunk, error] {
	return func(yield func(*StreamChunk, error) bool) {
		rc := GetRunContext(ctx)
		subRC := &RunContext{
			AgentName: agent.GetName(),
			SessionID: rc.SessionID,
			UserID:    rc.UserID,
			Hooks:     rc.Hooks,
		}
		ctx = WithRunContext(ctx, subRC)

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
		hooks := subRC.Hooks

		history := make([]*Content, len(contents))
		copy(history, contents)

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
				if hooks != nil && hooks.OnComplete != nil {
					hooks.OnComplete(ctx, &RunResult{History: history})
				}
				yield(lastChunk, nil)
				return
			}

			callContent := NewFunctionCallContent(calls...)
			history = append(history, callContent)

			funcResponses, allSkip, infraErr := loopStep(ctx, calls, toolMap, hooks)
			if infraErr != nil {
				yield(nil, infraErr)
				return
			}

			respContent := NewFunctionResponseContent(funcResponses...)
			history = append(history, respContent)

			if allSkip && len(calls) > 0 {
				if hooks != nil && hooks.OnComplete != nil {
					hooks.OnComplete(ctx, &RunResult{History: history})
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

// loopStep executes tool calls, fires hooks, builds responses.
// Shared between AgentLoop and AgentLoopStream.
func loopStep(ctx context.Context, calls []*FunctionCall, toolMap map[string]Tool, hooks *Hooks) (
	funcResponses []*FunctionResponse, allSkip bool, err error) {

	results, infraErr := executeToolsParallel(ctx, calls, toolMap, hooks)
	if infraErr != nil {
		return nil, false, infraErr
	}

	funcResponses, allSkip = buildFuncResponses(calls, results, toolMap)
	return funcResponses, allSkip, nil
}
