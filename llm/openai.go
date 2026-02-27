package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"os"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/Harshal1000/ago"
	"github.com/Harshal1000/ago/agent"
)

// ---------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------

// OpenAI is the OpenAI implementation of ago.LLM.
type OpenAI struct {
	client openaisdk.Client
}

// Compile-time check.
var _ ago.LLM = (*OpenAI)(nil)

// init registers the OpenAI backend factory.
func init() {
	agent.RegisterBackend(agent.BackendOpenAI, func() (ago.LLM, error) {
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("ago: OPENAI_API_KEY environment variable not set")
		}
		client := openaisdk.NewClient(option.WithAPIKey(apiKey))
		return &OpenAI{client: client}, nil
	})
}

// Name returns the backend identifier, "openai", for the ago.LLM interface.
func (o *OpenAI) Name() string {
	return "openai"
}

// Close closes the client connection. For OpenAI, this is a no-op.
func (o *OpenAI) Close() error {
	return nil
}

// Generate performs a non-streaming chat completion.
func (o *OpenAI) Generate(ctx context.Context, model string, contents []*ago.Content, config *ago.GenerateConfig) (*ago.Response, error) {
	messages, err := openaiToMessages(contents)
	if err != nil {
		return nil, fmt.Errorf("ago: openai: convert contents: %w", err)
	}

	params := openaisdk.ChatCompletionNewParams{
		Model:    model,
		Messages: messages,
	}

	if config != nil {
		if config.MaxOutputTokens > 0 {
			params.MaxCompletionTokens = openaisdk.Int(int64(config.MaxOutputTokens))
		}
		if config.Temperature != nil {
			params.Temperature = openaisdk.Float(*config.Temperature)
		}
		if config.TopP != nil {
			params.TopP = openaisdk.Float(*config.TopP)
		}
		if len(config.StopSequences) > 0 {
			params.Stop = openaisdk.ChatCompletionNewParamsStopUnion{
				OfStringArray: config.StopSequences,
			}
		}
		if config.Seed != nil {
			params.Seed = openaisdk.Int(int64(*config.Seed))
		}
		if config.PresencePenalty != nil {
			params.PresencePenalty = openaisdk.Float(*config.PresencePenalty)
		}
		if config.FrequencyPenalty != nil {
			params.FrequencyPenalty = openaisdk.Float(*config.FrequencyPenalty)
		}
		if config.SystemInstruction != nil {
			sysMsg := openaiToSystemMessage(config.SystemInstruction)
			// Prepend system message
			params.Messages = append([]openaisdk.ChatCompletionMessageParamUnion{sysMsg}, params.Messages...)
		}
		if len(config.Tools) > 0 {
			params.Tools = openaiToTools(config.Tools)
		}
	}

	resp, err := o.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("ago: openai: generate: %w", err)
	}

	return openaiFromResponse(resp), nil
}

// GenerateStream performs a streaming chat completion.
func (o *OpenAI) GenerateStream(ctx context.Context, model string, contents []*ago.Content, config *ago.GenerateConfig) iter.Seq2[*ago.StreamChunk, error] {
	return func(yield func(*ago.StreamChunk, error) bool) {
		messages, err := openaiToMessages(contents)
		if err != nil {
			yield(nil, fmt.Errorf("ago: openai: convert contents: %w", err))
			return
		}

		params := openaisdk.ChatCompletionNewParams{
			Model:    model,
			Messages: messages,
		}

		if config != nil {
			if config.MaxOutputTokens > 0 {
				params.MaxCompletionTokens = openaisdk.Int(int64(config.MaxOutputTokens))
			}
			if config.Temperature != nil {
				params.Temperature = openaisdk.Float(*config.Temperature)
			}
			if config.TopP != nil {
				params.TopP = openaisdk.Float(*config.TopP)
			}
			if len(config.StopSequences) > 0 {
				params.Stop = openaisdk.ChatCompletionNewParamsStopUnion{
					OfStringArray: config.StopSequences,
				}
			}
			if config.Seed != nil {
				params.Seed = openaisdk.Int(int64(*config.Seed))
			}
			if config.PresencePenalty != nil {
				params.PresencePenalty = openaisdk.Float(*config.PresencePenalty)
			}
			if config.FrequencyPenalty != nil {
				params.FrequencyPenalty = openaisdk.Float(*config.FrequencyPenalty)
			}
			if config.SystemInstruction != nil {
				sysMsg := openaiToSystemMessage(config.SystemInstruction)
				params.Messages = append([]openaisdk.ChatCompletionMessageParamUnion{sysMsg}, params.Messages...)
			}
			if len(config.Tools) > 0 {
				params.Tools = openaiToTools(config.Tools)
			}
		}

		stream := o.client.Chat.Completions.NewStreaming(ctx, params)
		acc := openaisdk.ChatCompletionAccumulator{}

		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			delta := choice.Delta

			// Build candidates from delta
			var parts []*ago.Part

			if delta.Content != "" {
				parts = append(parts, &ago.Part{Text: delta.Content})
			}

			// Handle tool calls in delta
			for _, tc := range delta.ToolCalls {
				if tc.Function.Name != "" || tc.Function.Arguments != "" {
					var args map[string]any
					if tc.Function.Arguments != "" {
						_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
					}
					parts = append(parts, &ago.Part{
						FunctionCall: &ago.FunctionCall{
							ID:   tc.ID,
							Name: tc.Function.Name,
							Args: args,
						},
					})
				}
			}

			if len(parts) > 0 {
				candidate := &ago.Candidate{
					Content: &ago.Content{
						Role:  ago.RoleModel,
						Parts: parts,
					},
					Index: int(choice.Index),
				}

				if choice.FinishReason != "" {
					candidate.FinishReason = openaiFromFinishReason(choice.FinishReason)
				}

				streamChunk := &ago.StreamChunk{
					Candidates: []*ago.Candidate{candidate},
				}
				if !yield(streamChunk, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("ago: openai: stream: %w", err))
			return
		}

		// Final chunk with complete response
		finalChunk := &ago.StreamChunk{
			Complete:   true,
			Candidates: openaiFromAccumulator(&acc),
		}

		if acc.Usage.TotalTokens > 0 {
			finalChunk.Usage = &ago.TokenUsage{
				PromptTokenCount:     int32(acc.Usage.PromptTokens),
				CandidatesTokenCount: int32(acc.Usage.CompletionTokens),
				TotalTokenCount:      int32(acc.Usage.TotalTokens),
			}
		}

		yield(finalChunk, nil)
	}
}

// ---------------------------------------------------------------------------
// Converters: ago -> openai
// ---------------------------------------------------------------------------

// openaiToMessages converts ago.Content slices to OpenAI chat messages.
func openaiToMessages(contents []*ago.Content) ([]openaisdk.ChatCompletionMessageParamUnion, error) {
	var messages []openaisdk.ChatCompletionMessageParamUnion

	for _, c := range contents {
		msgs, err := openaiToContentMessages(c)
		if err != nil {
			return nil, err
		}
		messages = append(messages, msgs...)
	}

	return messages, nil
}

// openaiToContentMessages converts a single ago.Content to one or more OpenAI messages.
// Tool responses may expand to multiple messages (one per tool call).
func openaiToContentMessages(c *ago.Content) ([]openaisdk.ChatCompletionMessageParamUnion, error) {
	switch c.Role {
	case ago.RoleUser:
		msg, err := openaiToUserMessage(c)
		if err != nil {
			return nil, err
		}
		return []openaisdk.ChatCompletionMessageParamUnion{msg}, nil
	case ago.RoleModel:
		msg, err := openaiToAssistantMessage(c)
		if err != nil {
			return nil, err
		}
		return []openaisdk.ChatCompletionMessageParamUnion{msg}, nil
	case ago.RoleTool:
		return openaiToToolMessages(c)
	default:
		return nil, fmt.Errorf("ago: openai: unsupported role: %s", c.Role)
	}
}

// openaiToSystemMessage converts a system content to an OpenAI system message.
func openaiToSystemMessage(c *ago.Content) openaisdk.ChatCompletionMessageParamUnion {
	var text string
	for _, p := range c.Parts {
		if p.Text != "" {
			text += p.Text
		}
	}
	return openaisdk.SystemMessage(text)
}

// openaiToUserMessage converts user content to OpenAI user message.
func openaiToUserMessage(c *ago.Content) (openaisdk.ChatCompletionMessageParamUnion, error) {
	var text string
	for _, p := range c.Parts {
		if p.Text != "" {
			text += p.Text
		}
		// TODO: Handle images, files, etc.
	}
	return openaisdk.UserMessage(text), nil
}

// openaiToAssistantMessage converts assistant/model content to OpenAI assistant message.
func openaiToAssistantMessage(c *ago.Content) (openaisdk.ChatCompletionMessageParamUnion, error) {
	var text string
	var toolCalls []openaisdk.ChatCompletionMessageToolCallUnionParam

	for _, p := range c.Parts {
		if p.Text != "" {
			text += p.Text
		}
		if p.FunctionCall != nil {
			argsJSON, _ := json.Marshal(p.FunctionCall.Args)
			toolCall := openaisdk.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openaisdk.ChatCompletionMessageFunctionToolCallParam{
					ID:   p.FunctionCall.ID,
					Type: "function",
					Function: openaisdk.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      p.FunctionCall.Name,
						Arguments: string(argsJSON),
					},
				},
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}

	if len(toolCalls) > 0 {
		// Create assistant message with tool calls
		assistantMsg := openaisdk.ChatCompletionAssistantMessageParam{
			Role:      "assistant",
			Content:   openaisdk.ChatCompletionAssistantMessageParamContentUnion{OfString: openaisdk.String(text)},
			ToolCalls: toolCalls,
		}
		return openaisdk.ChatCompletionMessageParamUnion{OfAssistant: &assistantMsg}, nil
	}
	return openaisdk.AssistantMessage(text), nil
}

// openaiToToolMessages converts tool content to OpenAI tool messages.
// Returns one message per function response part.
func openaiToToolMessages(c *ago.Content) ([]openaisdk.ChatCompletionMessageParamUnion, error) {
	var messages []openaisdk.ChatCompletionMessageParamUnion

	for _, p := range c.Parts {
		if p.FunctionResponse != nil {
			respJSON, _ := json.Marshal(p.FunctionResponse.Response)
			msg := openaisdk.ToolMessage(string(respJSON), p.FunctionResponse.ID)
			messages = append(messages, msg)
		}
	}

	return messages, nil
}

// openaiToTools converts ago function declarations to OpenAI tool parameters.
func openaiToTools(decls []*ago.FunctionDeclaration) []openaisdk.ChatCompletionToolUnionParam {
	var tools []openaisdk.ChatCompletionToolUnionParam
	for _, d := range decls {
		schema := openaiToSchema(d.Parameters)
		tool := openaisdk.ChatCompletionToolUnionParam{
			OfFunction: &openaisdk.ChatCompletionFunctionToolParam{
				Function: openaisdk.FunctionDefinitionParam{
					Name:        d.Name,
					Description: openaisdk.String(d.Description),
					Parameters:  schema,
				},
			},
		}
		tools = append(tools, tool)
	}
	return tools
}

// openaiToSchema converts an ago.Schema to a JSON schema map.
func openaiToSchema(s *ago.Schema) map[string]any {
	if s == nil {
		return map[string]any{"type": "object"}
	}

	schema := map[string]any{}

	// OpenAI requires object schemas to have properties.
	// If this is an empty object schema, return empty map to let OpenAI infer.
	if s.Type == ago.TypeObject && len(s.Properties) == 0 {
		return schema
	}

	schema["type"] = string(s.Type)

	if s.Description != "" {
		schema["description"] = s.Description
	}
	if s.Format != "" {
		schema["format"] = s.Format
	}
	if s.Nullable {
		schema["nullable"] = true
	}
	if len(s.Enum) > 0 {
		schema["enum"] = s.Enum
	}
	if len(s.Required) > 0 {
		schema["required"] = s.Required
	}
	if s.Items != nil {
		schema["items"] = openaiToSchema(s.Items)
	}
	if len(s.Properties) > 0 {
		props := make(map[string]any)
		for k, v := range s.Properties {
			props[k] = openaiToSchema(v)
		}
		schema["properties"] = props
	}

	return schema
}

// ---------------------------------------------------------------------------
// Converters: openai -> ago
// ---------------------------------------------------------------------------

// openaiFromResponse converts an OpenAI chat completion to an ago.Response.
func openaiFromResponse(resp *openaisdk.ChatCompletion) *ago.Response {
	candidates := make([]*ago.Candidate, 0, len(resp.Choices))
	for _, choice := range resp.Choices {
		candidates = append(candidates, openaiFromChoice(&choice))
	}

	result := &ago.Response{
		Candidates:   candidates,
		ModelVersion: resp.Model,
	}

	if resp.Usage.TotalTokens > 0 {
		result.Usage = ago.TokenUsage{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
	}

	return result
}

// openaiFromChoice converts an OpenAI choice to an ago.Candidate.
func openaiFromChoice(choice *openaisdk.ChatCompletionChoice) *ago.Candidate {
	msg := choice.Message
	parts := make([]*ago.Part, 0)

	// Add text content
	if msg.Content != "" {
		parts = append(parts, &ago.Part{Text: msg.Content})
	}

	// Add tool calls
	for _, tc := range msg.ToolCalls {
		if tc.Function.Name != "" {
			var args map[string]any
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			parts = append(parts, &ago.Part{
				FunctionCall: &ago.FunctionCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					Args: args,
				},
			})
		}
	}

	return &ago.Candidate{
		Content: &ago.Content{
			Role:  ago.RoleModel,
			Parts: parts,
		},
		FinishReason: openaiFromFinishReason(choice.FinishReason),
		Index:        int(choice.Index),
	}
}

// openaiFromFinishReason maps OpenAI finish reasons to ago.FinishReason.
func openaiFromFinishReason(reason string) ago.FinishReason {
	switch reason {
	case "stop":
		return ago.FinishReasonStop
	case "length":
		return ago.FinishReasonMaxTokens
	case "tool_calls":
		return ago.FinishReasonToolCall
	case "content_filter":
		return ago.FinishReasonSafety
	default:
		return ago.FinishReasonError
	}
}

// openaiFromAccumulator converts the streaming accumulator to ago candidates.
func openaiFromAccumulator(acc *openaisdk.ChatCompletionAccumulator) []*ago.Candidate {
	if len(acc.Choices) == 0 {
		return nil
	}

	candidates := make([]*ago.Candidate, 0, len(acc.Choices))
	for _, choice := range acc.Choices {
		msg := choice.Message
		parts := make([]*ago.Part, 0)

		if msg.Content != "" {
			parts = append(parts, &ago.Part{Text: msg.Content})
		}

		for _, tc := range msg.ToolCalls {
			if tc.Function.Name != "" {
				var args map[string]any
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				parts = append(parts, &ago.Part{
					FunctionCall: &ago.FunctionCall{
						ID:   tc.ID,
						Name: tc.Function.Name,
						Args: args,
					},
				})
			}
		}

		candidates = append(candidates, &ago.Candidate{
			Content: &ago.Content{
				Role:  ago.RoleModel,
				Parts: parts,
			},
			FinishReason: openaiFromFinishReason(choice.FinishReason),
			Index:        int(choice.Index),
		})
	}

	return candidates
}
