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

// sequentialStrategy runs agents in order, threading accumulated output forward.
type sequentialStrategy struct {
	agents []Strategy
}

// Sequential returns a Strategy that runs agents in order.
// Each agent receives the original input plus all prior agents' outputs as context.
// The final agent's result is returned.
func Sequential(agents ...Strategy) Strategy {
	return &sequentialStrategy{agents: agents}
}

func (s *sequentialStrategy) GetName() string { return "sequential" }

func (s *sequentialStrategy) Execute(ctx context.Context, app *App, contents []*Content, opts *RunOptions) (*RunResult, error) {
	var err error
	opts, err = ensureStrategySession(ctx, app, s.GetName(), opts)
	if err != nil {
		return nil, err
	}
	bufferStrategyInput(ctx, app, opts, contents)

	accumulated := make([]*Content, len(contents))
	copy(accumulated, contents)

	sub := subApp(app)
	var lastResult *RunResult
	for _, agent := range s.agents {
		result, err := agent.Execute(ctx, sub, accumulated, opts)
		if err != nil {
			return nil, fmt.Errorf("ago: sequential[%s]: %w", agent.GetName(), err)
		}
		lastResult = result
		// Append prior output as a labeled user turn so the next agent treats it
		// as new input rather than its own prior model response.
		if out := extractRunResultText(result); out != "" {
			accumulated = append(accumulated, NewTextContent(RoleUser,
				fmt.Sprintf("[%s output]:\n%s", agent.GetName(), out)))
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

// parallelStrategy runs agents concurrently, optionally aggregating results.
type parallelStrategy struct {
	agents     []Strategy
	aggregator Strategy
}

// Parallel returns a Strategy that runs all agents concurrently with the same input.
// Call .Aggregate(agg) to route combined results through an aggregator agent.
func Parallel(agents ...Strategy) *parallelStrategy {
	return &parallelStrategy{agents: agents}
}

// Aggregate sets an aggregator strategy that receives all parallel results combined.
// Returns the receiver for chaining.
func (p *parallelStrategy) Aggregate(agg Strategy) *parallelStrategy {
	p.aggregator = agg
	return p
}

func (p *parallelStrategy) GetName() string { return "parallel" }

func (p *parallelStrategy) Execute(ctx context.Context, app *App, contents []*Content, opts *RunOptions) (*RunResult, error) {
	var err error
	opts, err = ensureStrategySession(ctx, app, p.GetName(), opts)
	if err != nil {
		return nil, err
	}
	bufferStrategyInput(ctx, app, opts, contents)

	type indexedResult struct {
		idx    int
		result *RunResult
		err    error
	}

	sub := subApp(app)
	ch := make(chan indexedResult, len(p.agents))
	var wg sync.WaitGroup
	for i, agent := range p.agents {
		wg.Add(1)
		go func(idx int, a Strategy) {
			defer wg.Done()
			result, err := a.Execute(ctx, sub, contents, opts)
			ch <- indexedResult{idx: idx, result: result, err: err}
		}(i, agent)
	}
	go func() {
		wg.Wait()
		close(ch)
	}()

	results := make([]*RunResult, len(p.agents))
	for ir := range ch {
		if ir.err != nil {
			return nil, fmt.Errorf("ago: parallel[%s]: %w", p.agents[ir.idx].GetName(), ir.err)
		}
		results[ir.idx] = ir.result
	}

	if p.aggregator != nil {
		var parts []string
		for i, r := range results {
			parts = append(parts, fmt.Sprintf("[%s]:\n%s", p.agents[i].GetName(), extractRunResultText(r)))
		}
		combined := strings.Join(parts, "\n\n")
		aggContents := append(append([]*Content(nil), contents...), NewTextContent(RoleUser, combined))
		return p.aggregator.Execute(ctx, sub, aggContents, opts)
	}

	// No aggregator: return JSON map of agent name → output.
	out := make(map[string]string, len(results))
	for i, r := range results {
		out[p.agents[i].GetName()] = extractRunResultText(r)
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

// loopStrategy repeats its steps until a stop condition or max iterations.
type loopStrategy struct {
	steps      []Strategy
	max        int
	shouldStop func(output string) bool
}

// Loop returns a Strategy that runs steps sequentially in a loop.
// Use .Max(n) to cap iterations and .Until(fn) to stop early.
func Loop(steps ...Strategy) *loopStrategy {
	return &loopStrategy{steps: steps}
}

// Max sets the maximum number of loop iterations. Returns the receiver for chaining.
func (l *loopStrategy) Max(n int) *loopStrategy {
	l.max = n
	return l
}

// Until sets a stop condition checked after each step. Returns the receiver for chaining.
func (l *loopStrategy) Until(fn func(output string) bool) *loopStrategy {
	l.shouldStop = fn
	return l
}

func (l *loopStrategy) GetName() string { return "loop" }

func (l *loopStrategy) Execute(ctx context.Context, app *App, contents []*Content, opts *RunOptions) (*RunResult, error) {
	var err error
	opts, err = ensureStrategySession(ctx, app, l.GetName(), opts)
	if err != nil {
		return nil, err
	}
	bufferStrategyInput(ctx, app, opts, contents)

	accumulated := make([]*Content, len(contents))
	copy(accumulated, contents)

	maxIter := l.max
	if maxIter <= 0 {
		maxIter = DefaultMaxIterations
	}

	sub := subApp(app)
	var lastResult *RunResult
	for i := 0; i < maxIter; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// Run steps sequentially; check stop condition after each step.
		stepAccum := accumulated
		var shouldBreak bool
		for _, step := range l.steps {
			r, err := step.Execute(ctx, sub, stepAccum, opts)
			if err != nil {
				return nil, fmt.Errorf("ago: loop[%s]: %w", step.GetName(), err)
			}
			lastResult = r
			if out := extractRunResultText(r); out != "" {
				stepAccum = append(stepAccum, NewTextContent(RoleUser,
					fmt.Sprintf("[%s output]:\n%s", step.GetName(), out)))
			}
			if l.shouldStop != nil && l.shouldStop(extractRunResultText(r)) {
				shouldBreak = true
				break
			}
		}

		// Carry the last step's output into the next iteration's context.
		if lastResult != nil {
			if out := extractRunResultText(lastResult); out != "" {
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

// orchestrateStrategy uses a coordinator agent (via LLM tool calls) to dispatch workers.
type orchestrateStrategy struct {
	coordinator AgentConfig
	workers     []Strategy
}

// Orchestrate returns a Strategy where a coordinator LLM decides which workers to call.
// Workers are registered as tools on the coordinator; the LLM dispatches them via tool calls.
// coordinator must implement AgentConfig (e.g. *agent.Agent).
func Orchestrate(coordinator AgentConfig, workers ...Strategy) Strategy {
	return &orchestrateStrategy{coordinator: coordinator, workers: workers}
}

func (o *orchestrateStrategy) GetName() string { return o.coordinator.GetName() }

func (o *orchestrateStrategy) Execute(ctx context.Context, app *App, contents []*Content, opts *RunOptions) (*RunResult, error) {
	// Create the session before RunStrategy so workers can share the same session ID.
	var err error
	opts, err = ensureStrategySession(ctx, app, o.GetName(), opts)
	if err != nil {
		return nil, err
	}

	// Build augmented tool list: coordinator's existing tools + strategy tools for each worker.
	existing := o.coordinator.GetTools()
	allTools := make([]Tool, 0, len(existing)+len(o.workers))
	allTools = append(allTools, existing...)
	for _, w := range o.workers {
		allTools = append(allTools, newStrategyTool(w, subApp(app), opts))
	}
	wrapped := &agentWithTools{AgentConfig: o.coordinator, tools: allTools}
	return RunStrategy(ctx, app, wrapped, contents, opts)
}

// ---------------------------------------------------------------------------
// strategyTool — wraps a Strategy as a Tool for use in Orchestrate
// ---------------------------------------------------------------------------

type strategyTool struct {
	worker Strategy
	app    *App        // sub-agent app (storage + hooks, no history load)
	opts   *RunOptions // shared session for event storage
}

func newStrategyTool(worker Strategy, app *App, opts *RunOptions) *strategyTool {
	return &strategyTool{worker: worker, app: app, opts: opts}
}

func (t *strategyTool) Name() string { return t.worker.GetName() }

func (t *strategyTool) Declaration() *FunctionDeclaration {
	return &FunctionDeclaration{
		Name:        t.worker.GetName(),
		Description: fmt.Sprintf("Delegate task to the %s agent.", t.worker.GetName()),
		Parameters: &Schema{
			Type: TypeObject,
			Properties: map[string]*Schema{
				"input": {Type: TypeString, Description: "Input for the agent"},
			},
			Required: []string{"input"},
		},
	}
}

func (t *strategyTool) Execute(ctx context.Context, args map[string]any) (*ToolResult, error) {
	inputText, _ := args["input"].(string)
	result, err := t.worker.Execute(ctx, t.app, []*Content{
		NewTextContent(RoleUser, inputText),
	}, t.opts)
	if err != nil {
		return &ToolResult{Error: err}, nil
	}
	return &ToolResult{
		Response: map[string]any{"result": extractRunResultText(result)},
	}, nil
}

func (t *strategyTool) Options() ToolOptions { return ToolOptions{IsAgentCall: true} }

var _ Tool = (*strategyTool)(nil)

// ---------------------------------------------------------------------------
// agentWithTools — wraps an AgentConfig with a custom tool list
// ---------------------------------------------------------------------------

// agentWithTools wraps an AgentConfig, overriding its tool list without mutation.
// Used by Orchestrate to augment a coordinator with worker strategy tools.
type agentWithTools struct {
	AgentConfig
	tools []Tool
}

func (a *agentWithTools) GetTools() []Tool { return a.tools }

var _ AgentConfig = (*agentWithTools)(nil)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// subApp returns an App suitable for sub-agent calls within a strategy.
// It propagates storage and hooks but disables history loading so strategies
// manage their own context accumulation rather than reloading the shared session.
// isSubAgent=true prevents sub-agents from re-buffering input that the strategy
// already stored once via bufferStrategyInput.
func subApp(app *App) *App {
	if app == nil {
		return nil
	}
	return &App{
		Storage:        app.Storage,
		HistoryLimit:   app.HistoryLimit,
		IncludeHistory: false,
		Hooks:          app.Hooks,
		isSubAgent:     true,
	}
}

// extractRunResultText returns the text from the first model candidate in a RunResult.
func extractRunResultText(r *RunResult) string {
	if r == nil || r.Response == nil || len(r.Response.Candidates) == 0 {
		return ""
	}
	c := r.Response.Candidates[0].Content
	if c == nil {
		return ""
	}
	var parts []string
	for _, p := range c.Parts {
		if p.Text != "" {
			parts = append(parts, p.Text)
		}
	}
	return strings.Join(parts, "\n")
}
