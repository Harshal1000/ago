package ago

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// ---------------------------------------------------------------------------
// Sequential
// ---------------------------------------------------------------------------

// sequentialRunner runs runners in order, threading accumulated output forward.
type sequentialRunner struct {
	runners []Runner
}

// Sequential returns a Runner that runs runners in order.
// Each runner receives the original input plus all prior runners' outputs as context.
// The final runner's result is returned.
func Sequential(runners ...Runner) Runner {
	return &sequentialRunner{runners: runners}
}

func (s *sequentialRunner) Run(ctx context.Context, contents []*Content) (*RunResult, error) {
	accumulated := make([]*Content, len(contents))
	copy(accumulated, contents)

	var lastResult *RunResult
	for i, r := range s.runners {
		result, err := r.Run(ctx, accumulated)
		if err != nil {
			return nil, fmt.Errorf("ago: sequential[%s]: %w", runnerName(r, fmt.Sprintf("step-%d", i)), err)
		}
		lastResult = result
		if out := ExtractText(result); out != "" {
			accumulated = append(accumulated, NewTextContent(RoleUser,
				fmt.Sprintf("[%s output]:\n%s", runnerName(r, fmt.Sprintf("step-%d", i)), out)))
		}
	}
	if lastResult == nil {
		return &RunResult{History: accumulated}, nil
	}
	return lastResult, nil
}

// ---------------------------------------------------------------------------
// Parallel
// ---------------------------------------------------------------------------

// ParallelRunner runs runners concurrently, optionally aggregating results.
type ParallelRunner struct {
	runners    []Runner
	aggregator Runner
}

// Parallel returns a Runner that runs all runners concurrently with the same input.
// Call .Aggregate(agg) to route combined results through an aggregator runner.
func Parallel(runners ...Runner) *ParallelRunner {
	return &ParallelRunner{runners: runners}
}

// Aggregate sets an aggregator runner that receives all parallel results combined.
// Returns the receiver for chaining.
func (p *ParallelRunner) Aggregate(agg Runner) *ParallelRunner {
	p.aggregator = agg
	return p
}

func (p *ParallelRunner) Run(ctx context.Context, contents []*Content) (*RunResult, error) {
	type indexedResult struct {
		idx    int
		result *RunResult
		err    error
	}

	ch := make(chan indexedResult, len(p.runners))
	var wg sync.WaitGroup
	for i, r := range p.runners {
		wg.Add(1)
		go func(idx int, runner Runner) {
			defer wg.Done()
			result, err := runner.Run(ctx, contents)
			ch <- indexedResult{idx: idx, result: result, err: err}
		}(i, r)
	}
	go func() {
		wg.Wait()
		close(ch)
	}()

	results := make([]*RunResult, len(p.runners))
	for ir := range ch {
		if ir.err != nil {
			return nil, fmt.Errorf("ago: parallel[%s]: %w",
				runnerName(p.runners[ir.idx], fmt.Sprintf("worker-%d", ir.idx)), ir.err)
		}
		results[ir.idx] = ir.result
	}

	if p.aggregator != nil {
		var parts []string
		for i, r := range results {
			parts = append(parts, fmt.Sprintf("[%s]:\n%s",
				runnerName(p.runners[i], fmt.Sprintf("worker-%d", i)), ExtractText(r)))
		}
		combined := strings.Join(parts, "\n\n")
		aggContents := append(append([]*Content(nil), contents...), NewTextContent(RoleUser, combined))
		return p.aggregator.Run(ctx, aggContents)
	}

	// No aggregator: return JSON map of runner name → output.
	out := make(map[string]string, len(results))
	for i, r := range results {
		out[runnerName(p.runners[i], fmt.Sprintf("worker-%d", i))] = ExtractText(r)
	}
	data, _ := json.Marshal(out)
	return &RunResult{
		Response: &Response{
			Candidates: []*Candidate{{
				Content:      NewTextContent(RoleModel, string(data)),
				FinishReason: FinishReasonStop,
			}},
		},
	}, nil
}

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------

// LoopRunner repeats its steps until a stop condition or max iterations.
type LoopRunner struct {
	steps      []Runner
	max        int
	shouldStop func(output string) bool
}

// Loop returns a Runner that runs steps sequentially in a loop.
// Use .Max(n) to cap iterations and .Until(fn) to stop early.
func Loop(steps ...Runner) *LoopRunner {
	return &LoopRunner{steps: steps}
}

// Max sets the maximum number of loop iterations. Returns the receiver for chaining.
func (l *LoopRunner) Max(n int) *LoopRunner {
	l.max = n
	return l
}

// Until sets a stop condition checked after each step. Returns the receiver for chaining.
func (l *LoopRunner) Until(fn func(output string) bool) *LoopRunner {
	l.shouldStop = fn
	return l
}

func (l *LoopRunner) Run(ctx context.Context, contents []*Content) (*RunResult, error) {
	accumulated := make([]*Content, len(contents))
	copy(accumulated, contents)

	maxIter := l.max
	if maxIter <= 0 {
		maxIter = DefaultMaxIterations
	}

	var lastResult *RunResult
	for i := 0; i < maxIter; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		stepAccum := accumulated
		var shouldBreak bool
		for j, step := range l.steps {
			r, err := step.Run(ctx, stepAccum)
			if err != nil {
				return nil, fmt.Errorf("ago: loop[%s]: %w", runnerName(step, fmt.Sprintf("step-%d", j)), err)
			}
			lastResult = r
			if out := ExtractText(r); out != "" {
				stepAccum = append(stepAccum, NewTextContent(RoleUser,
					fmt.Sprintf("[%s output]:\n%s", runnerName(step, fmt.Sprintf("step-%d", j)), out)))
			}
			if l.shouldStop != nil && l.shouldStop(ExtractText(r)) {
				shouldBreak = true
				break
			}
		}

		if lastResult != nil {
			if out := ExtractText(lastResult); out != "" {
				accumulated = append(accumulated, NewTextContent(RoleUser,
					fmt.Sprintf("[iteration %d output]:\n%s", i+1, out)))
			}
		}

		if shouldBreak {
			break
		}
	}

	if lastResult == nil {
		return &RunResult{History: accumulated}, nil
	}
	return lastResult, nil
}

// ---------------------------------------------------------------------------
// Orchestrate
// ---------------------------------------------------------------------------

// Orchestrate returns a Runner where a coordinator LLM decides which workers to call.
// Workers are registered as tools on the coordinator; the LLM dispatches them via tool calls.
// coordinator must implement AgentConfig (e.g. *agent.Agent).
func Orchestrate(coordinator AgentConfig, workers ...Runner) Runner {
	existing := coordinator.GetTools()
	allTools := make([]Tool, 0, len(existing)+len(workers))
	allTools = append(allTools, existing...)
	for i, w := range workers {
		allTools = append(allTools, &workerTool{
			name:   runnerName(w, fmt.Sprintf("worker-%d", i)),
			desc:   fmt.Sprintf("Delegate task to the %s agent.", runnerName(w, fmt.Sprintf("worker-%d", i))),
			runner: w,
		})
	}
	wrapped := &agentWithTools{AgentConfig: coordinator, tools: allTools}
	return &orchestrateRunner{agent: wrapped}
}

type orchestrateRunner struct {
	agent AgentConfig
}

func (o *orchestrateRunner) Run(ctx context.Context, contents []*Content) (*RunResult, error) {
	return AgentLoop(ctx, o.agent, contents)
}

// ---------------------------------------------------------------------------
// workerTool — wraps a Runner as a Tool for use in Orchestrate
// ---------------------------------------------------------------------------

type workerTool struct {
	name   string
	desc   string
	runner Runner
}

func (t *workerTool) Name() string { return t.name }

func (t *workerTool) Declaration() *FunctionDeclaration {
	return &FunctionDeclaration{
		Name:        t.name,
		Description: t.desc,
		Parameters: &Schema{
			Type: TypeObject,
			Properties: map[string]*Schema{
				"input": {Type: TypeString, Description: "Input for the agent"},
			},
			Required: []string{"input"},
		},
	}
}

func (t *workerTool) Execute(ctx context.Context, args map[string]any) (*ToolResult, error) {
	inputText, _ := args["input"].(string)
	result, err := t.runner.Run(ctx, []*Content{
		NewTextContent(RoleUser, inputText),
	})
	if err != nil {
		return &ToolResult{Error: err}, nil
	}
	return &ToolResult{
		Response: map[string]any{"result": ExtractText(result)},
	}, nil
}

func (t *workerTool) Options() ToolOptions { return ToolOptions{} }

var _ Tool = (*workerTool)(nil)

// ---------------------------------------------------------------------------
// agentWithTools — wraps an AgentConfig with a custom tool list
// ---------------------------------------------------------------------------

// agentWithTools wraps an AgentConfig, overriding its tool list without mutation.
type agentWithTools struct {
	AgentConfig
	tools []Tool
}

func (a *agentWithTools) GetTools() []Tool { return a.tools }

var _ AgentConfig = (*agentWithTools)(nil)
