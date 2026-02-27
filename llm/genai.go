package llm

import (
	"context"
	"fmt"
	"iter"

	genaisdk "google.golang.org/genai"

	"github.com/Harshal1000/ago"
	"github.com/Harshal1000/ago/agent"
)

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

// GenAIOption configures a GenAI client at construction time (e.g. API key or custom client).
type GenAIOption func(*genaiOptions)

type genaiOptions struct {
	apiKey string
	client *genaisdk.Client
}

// WithAPIKey sets the Google API key explicitly instead of using GOOGLE_API_KEY or ADC.
func WithAPIKey(key string) GenAIOption {
	return func(o *genaiOptions) { o.apiKey = key }
}

// WithGenAIClient uses the given genai.Client instead of creating one; useful for tests or custom config.
func WithGenAIClient(c *genaisdk.Client) GenAIOption {
	return func(o *genaiOptions) { o.client = c }
}

// ---------------------------------------------------------------------------
// GenAI
// ---------------------------------------------------------------------------

// GenAI is the Google Generative AI implementation of ago.LLM (Gemini and compatible models).
type GenAI struct {
	client *genaisdk.Client
}

// NewGenAI builds a GenAI client. With no options, credentials come from GOOGLE_API_KEY or
// Application Default Credentials. Options can set an explicit API key or inject a custom client.
func NewGenAI(opts ...GenAIOption) (*GenAI, error) {
	o := &genaiOptions{}
	for _, opt := range opts {
		opt(o)
	}

	if o.client != nil {
		return &GenAI{client: o.client}, nil
	}

	var cfg *genaisdk.ClientConfig
	if o.apiKey != "" {
		cfg = &genaisdk.ClientConfig{APIKey: o.apiKey}
	}
	client, err := genaisdk.NewClient(context.Background(), cfg)
	if err != nil {
		return nil, fmt.Errorf("genai: %w", err)
	}
	return &GenAI{client: client}, nil
}

// Name returns the backend identifier, "genai", for the ago.LLM interface.
func (g *GenAI) Name() string { return "genai" }

// Close releases resources; the underlying genai client does not require cleanup, so this is a no-op.
func (g *GenAI) Close() error { return nil }

// Generate performs a single non-streaming completion request and returns the full response and usage.
func (g *GenAI) Generate(ctx context.Context, model string, params *ago.GenerateParams) (*ago.Response, error) {
	genaiContents := genaiToContents(params.Contents)
	genaiConfig := genaiToConfig(params)

	result, err := g.client.Models.GenerateContent(ctx, model, genaiContents, genaiConfig)
	if err != nil {
		return nil, fmt.Errorf("genai generate: %w", err)
	}

	return genaiFromResponse(result), nil
}

// GenerateStream performs a streaming completion and yields ago.StreamChunk values (content deltas, usage, then Done).
func (g *GenAI) GenerateStream(ctx context.Context, model string, params *ago.GenerateParams) iter.Seq2[*ago.StreamChunk, error] {
	genaiContents := genaiToContents(params.Contents)
	genaiConfig := genaiToConfig(params)

	return func(yield func(*ago.StreamChunk, error) bool) {
		for chunk, err := range g.client.Models.GenerateContentStream(ctx, model, genaiContents, genaiConfig) {
			if err != nil {
				yield(&ago.StreamChunk{ErrorMessage: err.Error(), Complete: true}, err)
				return
			}
			sc := genaiFromStreamChunk(chunk)
			if !yield(sc, nil) {
				return
			}
			if sc.Complete {
				return
			}
		}
	}
}

// Ensure GenAI implements ago.LLM at compile time.
var _ ago.LLM = (*GenAI)(nil)

func init() {
	agent.RegisterBackend(agent.BackendGenAI, func() (ago.LLM, error) {
		return NewGenAI()
	})
}

// ---------------------------------------------------------------------------
// Converters: ago -> genai
// ---------------------------------------------------------------------------

// genaiToContents maps ago conversation contents to genai Content (system role skipped; handled in config).
func genaiToContents(contents []*ago.Content) []*genaisdk.Content {
	out := make([]*genaisdk.Content, 0, len(contents))
	for _, c := range contents {
		if c.Role == ago.RoleSystem {
			continue
		}
		out = append(out, &genaisdk.Content{
			Role:  genaiToRole(c.Role),
			Parts: genaiToParts(c.Parts),
		})
	}
	return out
}

// genaiToRole maps ago.Role to genai role string (tool responses use "user" in genai).
func genaiToRole(r ago.Role) string {
	switch r {
	case ago.RoleModel:
		return string(genaisdk.RoleModel)
	case ago.RoleTool:
		return string(genaisdk.RoleUser)
	default:
		return string(genaisdk.RoleUser)
	}
}

// genaiToParts converts ago parts to genai Part slice (text, function call/response, inline/file data, code).
func genaiToParts(parts []*ago.Part) []*genaisdk.Part {
	out := make([]*genaisdk.Part, 0, len(parts))
	for _, p := range parts {
		switch {
		case p.FunctionCall != nil:
			out = append(out, genaisdk.NewPartFromFunctionCall(p.FunctionCall.Name, p.FunctionCall.Args))
		case p.FunctionResponse != nil:
			out = append(out, genaisdk.NewPartFromFunctionResponse(p.FunctionResponse.Name, p.FunctionResponse.Response))
		case p.InlineData != nil:
			gp := &genaisdk.Part{
				InlineData: &genaisdk.Blob{
					MIMEType: p.InlineData.MIMEType,
					Data:     p.InlineData.Data,
				},
			}
			if p.VideoMetadata != nil {
				gp.VideoMetadata = genaiToVideoMetadata(p.VideoMetadata)
			}
			out = append(out, gp)
		case p.FileData != nil:
			gp := &genaisdk.Part{
				FileData: &genaisdk.FileData{
					MIMEType:    p.FileData.MIMEType,
					FileURI:     p.FileData.FileURI,
					DisplayName: p.FileData.DisplayName,
				},
			}
			if p.VideoMetadata != nil {
				gp.VideoMetadata = genaiToVideoMetadata(p.VideoMetadata)
			}
			out = append(out, gp)
		case p.ExecutableCode != nil:
			out = append(out, &genaisdk.Part{
				ExecutableCode: &genaisdk.ExecutableCode{
					Code:     p.ExecutableCode.Code,
					Language: genaisdk.Language(p.ExecutableCode.Language),
				},
			})
		case p.CodeExecutionResult != nil:
			out = append(out, &genaisdk.Part{
				CodeExecutionResult: &genaisdk.CodeExecutionResult{
					Outcome: genaisdk.Outcome(p.CodeExecutionResult.Outcome),
					Output:  p.CodeExecutionResult.Output,
				},
			})
		case p.Text != "":
			out = append(out, genaisdk.NewPartFromText(p.Text))
		}
	}
	return out
}

// genaiToConfig maps ago.GenerateParams to genai.GenerateContentConfig.
func genaiToConfig(params *ago.GenerateParams) *genaisdk.GenerateContentConfig {
	cfg := &genaisdk.GenerateContentConfig{}

	if config := params.Config; config != nil {
		if config.MaxOutputTokens > 0 {
			cfg.MaxOutputTokens = int32(config.MaxOutputTokens)
		}
		if config.Temperature != nil {
			t := float32(*config.Temperature)
			cfg.Temperature = &t
		}
		if config.TopP != nil {
			t := float32(*config.TopP)
			cfg.TopP = &t
		}
		if config.TopK != nil {
			t := float32(*config.TopK)
			cfg.TopK = &t
		}
		if config.Seed != nil {
			s := int32(*config.Seed)
			cfg.Seed = &s
		}
		if config.PresencePenalty != nil {
			p := float32(*config.PresencePenalty)
			cfg.PresencePenalty = &p
		}
		if config.FrequencyPenalty != nil {
			p := float32(*config.FrequencyPenalty)
			cfg.FrequencyPenalty = &p
		}
		if len(config.StopSequences) > 0 {
			cfg.StopSequences = config.StopSequences
		}
		if config.ResponseMIMEType != "" {
			cfg.ResponseMIMEType = config.ResponseMIMEType
		}
		if config.ResponseSchema != nil {
			cfg.ResponseSchema = genaiToSchema(config.ResponseSchema)
		}
		if config.ThinkingConfig != nil {
			budget := int32(config.ThinkingConfig.Budget)
			cfg.ThinkingConfig = &genaisdk.ThinkingConfig{
				ThinkingBudget: &budget,
			}
		}
	}

	if params.SystemInstruction != nil {
		cfg.SystemInstruction = &genaisdk.Content{
			Role:  string(genaisdk.RoleUser),
			Parts: genaiToParts(params.SystemInstruction.Parts),
		}
	}
	if params.Tools != nil {
		cfg.Tools = genaiToTools(params.Tools)
	}

	return cfg
}

// genaiToTools wraps ago function declarations into genai Tool slice.
func genaiToTools(decls []*ago.FunctionDeclaration) []*genaisdk.Tool {
	if len(decls) == 0 {
		return nil
	}
	gDecls := make([]*genaisdk.FunctionDeclaration, 0, len(decls))
	for _, d := range decls {
		gDecls = append(gDecls, &genaisdk.FunctionDeclaration{
			Name:        d.Name,
			Description: d.Description,
			Parameters:  genaiToSchema(d.Parameters),
		})
	}
	return []*genaisdk.Tool{{FunctionDeclarations: gDecls}}
}

// genaiToSchema recursively maps ago.Schema to genai.Schema.
func genaiToSchema(s *ago.Schema) *genaisdk.Schema {
	if s == nil {
		return nil
	}
	gs := &genaisdk.Schema{
		Type:        genaiToSchemaType(s.Type),
		Description: s.Description,
		Format:      s.Format,
		Enum:        s.Enum,
		Required:    s.Required,
		Items:       genaiToSchema(s.Items),
	}
	if s.Nullable {
		gs.Nullable = &s.Nullable
	}
	if len(s.Properties) > 0 {
		gs.Properties = make(map[string]*genaisdk.Schema, len(s.Properties))
		for k, v := range s.Properties {
			gs.Properties[k] = genaiToSchema(v)
		}
	}
	return gs
}

// genaiToSchemaType maps ago.SchemaType to genai.Type.
func genaiToSchemaType(t ago.SchemaType) genaisdk.Type {
	switch t {
	case ago.TypeString:
		return genaisdk.TypeString
	case ago.TypeNumber:
		return genaisdk.TypeNumber
	case ago.TypeInteger:
		return genaisdk.TypeInteger
	case ago.TypeBoolean:
		return genaisdk.TypeBoolean
	case ago.TypeArray:
		return genaisdk.TypeArray
	case ago.TypeObject:
		return genaisdk.TypeObject
	default:
		return genaisdk.TypeString
	}
}

// ---------------------------------------------------------------------------
// Converters: genai -> ago
// ---------------------------------------------------------------------------

// genaiFromResponse maps genai GenerateContentResponse to ago.Response.
func genaiFromResponse(r *genaisdk.GenerateContentResponse) *ago.Response {
	resp := &ago.Response{}
	if u := genaiFromUsage(r.UsageMetadata); u != nil {
		resp.Usage = *u
	}
	if r.ModelVersion != "" {
		resp.ModelVersion = r.ModelVersion
	}
	for i, c := range r.Candidates {
		resp.Candidates = append(resp.Candidates, genaiFromCandidate(c, i))
	}
	return resp
}

// genaiFromCandidate maps one genai candidate to ago.Candidate.
// When the response contains function calls, FinishReason is set to FinishReasonToolCall
// (Gemini reports STOP for tool calls, but ago distinguishes them).
func genaiFromCandidate(c *genaisdk.Candidate, index int) *ago.Candidate {
	fr := genaiFromFinishReason(c.FinishReason)
	// Gemini uses STOP for tool calls — detect and remap.
	if fr == ago.FinishReasonStop && c.Content != nil && genaiHasFunctionCalls(c.Content.Parts) {
		fr = ago.FinishReasonToolCall
	}
	cand := &ago.Candidate{
		FinishReason:  fr,
		Index:         index,
		TokenCount:    c.TokenCount,
		FinishMessage: c.FinishMessage,
	}
	if c.Content != nil {
		role := ago.RoleModel
		if c.Content.Role != "" {
			role = ago.Role(c.Content.Role)
		}
		cand.Content = &ago.Content{
			Role:  role,
			Parts: genaiFromParts(c.Content.Parts),
		}
	}
	return cand
}

// genaiFromParts converts genai parts to ago Part slice.
func genaiFromParts(parts []*genaisdk.Part) []*ago.Part {
	out := make([]*ago.Part, 0, len(parts))
	for _, p := range parts {
		switch {
		case p.FunctionCall != nil:
			out = append(out, &ago.Part{
				FunctionCall: &ago.FunctionCall{
					Name: p.FunctionCall.Name,
					Args: p.FunctionCall.Args,
				},
			})
		case p.FunctionResponse != nil:
			out = append(out, &ago.Part{
				FunctionResponse: &ago.FunctionResponse{
					Name:     p.FunctionResponse.Name,
					Response: p.FunctionResponse.Response,
				},
			})
		case p.InlineData != nil:
			part := &ago.Part{
				InlineData: &ago.Blob{
					MIMEType: p.InlineData.MIMEType,
					Data:     p.InlineData.Data,
				},
			}
			if p.VideoMetadata != nil {
				part.VideoMetadata = genaiFromVideoMetadata(p.VideoMetadata)
			}
			out = append(out, part)
		case p.FileData != nil:
			part := &ago.Part{
				FileData: &ago.FileData{
					MIMEType:    p.FileData.MIMEType,
					FileURI:     p.FileData.FileURI,
					DisplayName: p.FileData.DisplayName,
				},
			}
			if p.VideoMetadata != nil {
				part.VideoMetadata = genaiFromVideoMetadata(p.VideoMetadata)
			}
			out = append(out, part)
		case p.ExecutableCode != nil:
			out = append(out, &ago.Part{
				ExecutableCode: &ago.ExecutableCode{
					Code:     p.ExecutableCode.Code,
					Language: ago.CodeLanguage(p.ExecutableCode.Language),
				},
			})
		case p.CodeExecutionResult != nil:
			out = append(out, &ago.Part{
				CodeExecutionResult: &ago.CodeExecutionResult{
					Outcome: ago.CodeOutcome(p.CodeExecutionResult.Outcome),
					Output:  p.CodeExecutionResult.Output,
				},
			})
		case p.Thought:
			out = append(out, &ago.Part{
				Text:             p.Text,
				Thought:          true,
				ThoughtSignature: p.ThoughtSignature,
			})
		case p.Text != "":
			out = append(out, &ago.Part{Text: p.Text})
		}
	}
	return out
}

// genaiFromUsage maps genai usage metadata to ago.TokenUsage.
func genaiFromUsage(u *genaisdk.GenerateContentResponseUsageMetadata) *ago.TokenUsage {
	if u == nil {
		return nil
	}
	return &ago.TokenUsage{
		PromptTokenCount:           u.PromptTokenCount,
		CandidatesTokenCount:       u.CandidatesTokenCount,
		TotalTokenCount:            u.TotalTokenCount,
		CachedContentTokenCount:    u.CachedContentTokenCount,
		ThoughtsTokenCount:         u.ThoughtsTokenCount,
		ToolUsePromptTokenCount:    u.ToolUsePromptTokenCount,
		TrafficType:                string(u.TrafficType),
		CacheTokensDetails:         genaiFromModalityTokenCounts(u.CacheTokensDetails),
		CandidatesTokensDetails:    genaiFromModalityTokenCounts(u.CandidatesTokensDetails),
		PromptTokensDetails:        genaiFromModalityTokenCounts(u.PromptTokensDetails),
		ToolUsePromptTokensDetails: genaiFromModalityTokenCounts(u.ToolUsePromptTokensDetails),
	}
}

func genaiFromModalityTokenCounts(s []*genaisdk.ModalityTokenCount) []*ago.ModalityTokenCount {
	if len(s) == 0 {
		return nil
	}
	out := make([]*ago.ModalityTokenCount, 0, len(s))
	for _, m := range s {
		if m == nil {
			continue
		}
		out = append(out, &ago.ModalityTokenCount{
			Modality:   string(m.Modality),
			TokenCount: m.TokenCount,
		})
	}
	return out
}

// genaiFromFinishReason maps genai finish reason to ago.FinishReason.
func genaiFromFinishReason(r genaisdk.FinishReason) ago.FinishReason {
	switch r {
	case genaisdk.FinishReasonStop:
		return ago.FinishReasonStop
	case genaisdk.FinishReasonMaxTokens:
		return ago.FinishReasonMaxTokens
	case genaisdk.FinishReasonSafety, genaisdk.FinishReasonRecitation,
		genaisdk.FinishReasonBlocklist, genaisdk.FinishReasonProhibitedContent,
		genaisdk.FinishReasonSPII:
		return ago.FinishReasonSafety
	case genaisdk.FinishReasonMalformedFunctionCall:
		return ago.FinishReasonError
	default:
		return ago.FinishReasonStop
	}
}

// genaiHasFunctionCalls checks if content contains any function call parts.
func genaiHasFunctionCalls(parts []*genaisdk.Part) bool {
	for _, p := range parts {
		if p.FunctionCall != nil {
			return true
		}
	}
	return false
}

// genaiToVideoMetadata maps ago video metadata to genai.
func genaiToVideoMetadata(v *ago.VideoMetadata) *genaisdk.VideoMetadata {
	gv := &genaisdk.VideoMetadata{
		StartOffset: v.StartOffset,
		EndOffset:   v.EndOffset,
	}
	if v.FPS != nil {
		gv.FPS = v.FPS
	}
	return gv
}

// genaiFromVideoMetadata maps genai video metadata to ago.
func genaiFromVideoMetadata(v *genaisdk.VideoMetadata) *ago.VideoMetadata {
	av := &ago.VideoMetadata{
		StartOffset: v.StartOffset,
		EndOffset:   v.EndOffset,
	}
	if v.FPS != nil {
		av.FPS = v.FPS
	}
	return av
}

// genaiFromStreamChunk converts a genai streaming chunk to ago.StreamChunk.
func genaiFromStreamChunk(chunk *genaisdk.GenerateContentResponse) *ago.StreamChunk {
	sc := &ago.StreamChunk{}
	sc.Usage = genaiFromUsage(chunk.UsageMetadata)
	for i, c := range chunk.Candidates {
		sc.Candidates = append(sc.Candidates, genaiFromCandidate(c, i))
		if c.FinishReason != "" {
			sc.Complete = true
		}
	}
	return sc
}
