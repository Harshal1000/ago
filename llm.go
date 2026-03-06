package ago

import (
	"context"
	"iter"
)

// ---------------------------------------------------------------------------
// Token Usage
// ---------------------------------------------------------------------------

// ModalityTokenCount reports the token count for a specific modality (e.g. TEXT, IMAGE, AUDIO).
type ModalityTokenCount struct {
	Modality   string `json:"modality,omitempty"`
	TokenCount int32  `json:"tokenCount,omitempty"`
}

// TokenUsage reports detailed token consumption for an LLM request.
type TokenUsage struct {
	PromptTokenCount           int32                 `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount       int32                 `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount            int32                 `json:"totalTokenCount,omitempty"`
	CachedContentTokenCount    int32                 `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount         int32                 `json:"thoughtsTokenCount,omitempty"`
	ToolUsePromptTokenCount    int32                 `json:"toolUsePromptTokenCount,omitempty"`
	TrafficType                string                `json:"trafficType,omitempty"`
	CacheTokensDetails         []*ModalityTokenCount `json:"cacheTokensDetails,omitempty"`
	CandidatesTokensDetails    []*ModalityTokenCount `json:"candidatesTokensDetails,omitempty"`
	PromptTokensDetails        []*ModalityTokenCount `json:"promptTokensDetails,omitempty"`
	ToolUsePromptTokensDetails []*ModalityTokenCount `json:"toolUsePromptTokensDetails,omitempty"`
}

// ---------------------------------------------------------------------------
// Finish Reason
// ---------------------------------------------------------------------------

// FinishReason explains why the model stopped generating.
type FinishReason string

const (
	FinishReasonStop      FinishReason = "stop"
	FinishReasonMaxTokens FinishReason = "max_tokens"
	FinishReasonToolCall  FinishReason = "tool_call"
	FinishReasonSafety    FinishReason = "safety"
	FinishReasonError     FinishReason = "error"
)

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

// Response is the result of a non-streaming LLM.Generate call.
type Response struct {
	Candidates   []*Candidate
	Usage        TokenUsage
	ModelVersion string
}

// Candidate is one possible completion from the model.
type Candidate struct {
	Content       *Content     `json:"content,omitempty"`
	FinishReason  FinishReason `json:"finishReason,omitempty"`
	Index         int          `json:"index,omitempty"`
	TokenCount    int32        `json:"tokenCount,omitempty"`
	FinishMessage string       `json:"finishMessage,omitempty"`
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// StreamChunk is a single piece of a streaming LLM response.
// The final chunk has Complete=true and is the source of truth for candidates and usage.
type StreamChunk struct {
	Candidates   []*Candidate `json:"candidates,omitempty"`
	Usage        *TokenUsage  `json:"usageMetadata,omitempty"`
	Complete     bool         `json:"complete,omitempty"`
	ErrorMessage string       `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// GenerateConfig
// ---------------------------------------------------------------------------

// GenerateConfig holds optional sampling and output parameters for an LLM request.
type GenerateConfig struct {
	MaxOutputTokens  int
	Temperature      *float64
	TopP             *float64
	TopK             *float64
	StopSequences    []string
	ResponseMIMEType string
	ResponseSchema   *Schema
	ThinkingConfig   *ThinkingConfig
	Seed             *int
	PresencePenalty  *float64
	FrequencyPenalty *float64
}

// ThinkingConfig enables and configures extended reasoning tokens.
type ThinkingConfig struct {
	Enabled bool
	Budget  int
}

// ---------------------------------------------------------------------------
// GenerateParams
// ---------------------------------------------------------------------------

// GenerateParams bundles everything the LLM needs for a single generation request.
type GenerateParams struct {
	Contents          []*Content
	Config            *GenerateConfig
	SystemInstruction *Content
	Tools             []*FunctionDeclaration
}

// ---------------------------------------------------------------------------
// LLM Interface
// ---------------------------------------------------------------------------

// LLM is the interface that all LLM backend implementations must satisfy.
type LLM interface {
	Name() string
	Generate(ctx context.Context, model string, params *GenerateParams) (*Response, error)
	GenerateStream(ctx context.Context, model string, params *GenerateParams) iter.Seq2[*StreamChunk, error]
	Close() error
}
