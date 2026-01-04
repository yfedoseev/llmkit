# Specialized APIs

Specialized APIs for text analysis, ranking, reranking, moderation, and classification beyond standard LLM chat completions.

## Quick Start

### Python

```python
from llmkit import LLMKitClient, RankingRequest, ModerationRequest

client = LLMKitClient.from_env()

# Rank documents
req = RankingRequest("python programming", ["doc1", "doc2", "doc3"])
response = client.rank_documents(req)
print(f"Top result: {response.first().document}")

# Check content moderation
req = ModerationRequest("Is this content appropriate?")
response = client.moderate_text(req)
print(f"Flagged: {response.flagged}")
```

### TypeScript

```typescript
import { LLMKitClient, RankingRequest, ModerationRequest } from 'llmkit'

const client = LLMKitClient.fromEnv()

// Rank documents
const rankReq = new RankingRequest('python', ['doc1', 'doc2'])
const rankResp = await client.rankDocuments(rankReq)
console.log(`Score: ${rankResp.first()?.score}`)

// Moderate content
const modReq = new ModerationRequest('Is this appropriate?')
const modResp = await client.moderateText(modReq)
console.log(`Flagged: ${modResp.flagged}`)
```

## API Overview

### Ranking API

Rank documents by relevance to a query.

**Providers:** Cohere, Jina

#### RankingRequest

```python
request = RankingRequest(
    query="python programming",
    documents=["doc1", "doc2", "doc3"]
)
request = request.with_top_k(2)  # Return top 2 results
```

#### RankedDocument

```python
doc = response.first()
print(f"Index: {doc.index}")
print(f"Score: {doc.score}")  # 0.0 to 1.0
print(f"Text: {doc.document}")
```

#### Example

```python
client = LLMKitClient.from_env()

documents = [
    "Python is a programming language",
    "Django is a web framework",
    "NumPy is a numerical computing library"
]

req = RankingRequest("Python data science", documents)
response = client.rank_documents(req)

for result in response.results:
    print(f"{result.document}: {result.score:.3f}")
```

### Reranking API

Rerank search results for semantic relevance (useful after BM25/keyword search).

**Providers:** Cohere Rerank, Jina Reranker

#### RerankingRequest

```python
request = RerankingRequest(
    query="best python books",
    documents=search_results
).with_top_n(5)  # Return top 5
```

#### RerankedResult

```python
result = response.first()
print(f"Relevance: {result.relevance_score}")  # Normalized relevance score
```

#### Example

```python
# After BM25 search returns results
search_results = ["result1", "result2", "result3"]

req = RerankingRequest("machine learning tutorial", search_results)
response = client.rerank_results(req)

# Top result is most relevant to the query
top_result = response.first()
```

### Moderation API

Check content for policy violations and safety issues.

**Providers:** OpenAI Moderation, Perspective API

#### ModerationRequest

```python
req = ModerationRequest("Your content here")
response = client.moderate_text(req)
```

#### ModerationScores

**Categories:**
- `hate`: Hateful speech
- `hate_threatening`: Threatening hateful speech
- `harassment`: Harassment content
- `harassment_threatening`: Threatening harassment
- `self_harm`: Self-harm content
- `self_harm_intent`: Intent to harm self
- `self_harm_instructions`: Instructions for self-harm
- `sexual`: Sexual content
- `sexual_minors`: Sexual content involving minors
- `violence`: Violent content
- `violence_graphic`: Graphic violence

```python
response = client.moderate_text(req)
print(f"Flagged: {response.flagged}")
print(f"Max score: {response.scores.max_score()}")
print(f"Violence score: {response.scores.violence}")
```

#### Example

```python
# Content moderation workflow
comments = [
    "This is a great product!",
    "I hate this so much",
    "Check this out...",
]

for comment in comments:
    req = ModerationRequest(comment)
    response = client.moderate_text(req)

    if response.flagged:
        print(f"FLAGGED: {comment}")
    else:
        print(f"OK: {comment}")
```

### Classification API

Classify text into provided categories with confidence scores.

**Providers:** OpenAI, Anthropic, Hugging Face

#### ClassificationRequest

```python
request = ClassificationRequest(
    text="This movie was amazing!",
    labels=["positive", "negative", "neutral"]
)
```

#### ClassificationResult

```python
result = response.top()
print(f"Label: {result.label}")  # Most likely label
print(f"Confidence: {result.confidence}")  # 0.0 to 1.0
```

#### Example

```python
# Sentiment classification
reviews = [
    "Excellent product, highly recommend!",
    "Terrible quality, waste of money",
    "It's okay, nothing special",
]

labels = ["positive", "negative", "neutral"]

for review in reviews:
    req = ClassificationRequest(review, labels)
    response = client.classify_text(req)

    top = response.top()
    print(f"{review}: {top.label} ({top.confidence:.2%})")
```

## Complete Examples

### Python: Multi-API Workflow

```python
from llmkit import LLMKitClient

client = LLMKitClient.from_env()

def process_user_comment(comment):
    """Process a user comment through all specialized APIs."""

    # 1. Check moderation
    mod_req = ModerationRequest(comment)
    mod_response = client.moderate_text(mod_req)

    if mod_response.flagged:
        return {"status": "rejected", "reason": "violates policy"}

    # 2. Classify sentiment
    sentiment_req = ClassificationRequest(
        comment,
        ["positive", "negative", "neutral"]
    )
    sentiment_response = client.classify_text(sentiment_req)

    return {
        "status": "approved",
        "sentiment": sentiment_response.top().label,
        "confidence": sentiment_response.top().confidence
    }

# Usage
result = process_user_comment("Great product!")
print(result)
# Output: {'status': 'approved', 'sentiment': 'positive', 'confidence': 0.98}
```

### TypeScript: Semantic Search with Reranking

```typescript
async function semanticSearch(query: string, bm25Results: string[]) {
  const client = LLMKitClient.fromEnv()

  // 1. Quick BM25 search returns many results
  // 2. Rerank for semantic relevance
  const rerankReq = new RerankingRequest(query, bm25Results).with_top_n(10)
  const rerankResp = await client.rerankResults(rerankReq)

  // 3. Return reranked results
  return rerankResp.results.map(r => ({
    document: r.document,
    relevance: r.relevance_score
  }))
}

// Usage
const query = "best practices for nodejs"
const bm25Results = [/* initial search results */]
const finalResults = await semanticSearch(query, bm25Results)
```

## Error Handling

### Python

```python
try:
    req = RankingRequest("query", ["d1", "d2"])
    response = client.rank_documents(req)
except Exception as e:
    print(f"Ranking failed: {e}")
    # Handle error appropriately
```

### TypeScript

```typescript
try {
  const req = new RankingRequest('query', ['d1', 'd2'])
  const response = await client.rankDocuments(req)
} catch (error) {
  console.error(`Ranking failed: ${(error as Error).message}`)
}
```

## Use Cases

### Document Search
**Ranking + Reranking Pipeline**
1. Use BM25 for initial retrieval (fast)
2. Use ranking/reranking for semantic relevance (accurate)

### Content Moderation
**Safety Checks**
1. Moderate user-generated content
2. Flag harmful material automatically
3. Review high-confidence violations

### Sentiment Analysis
**Classification**
1. Analyze customer reviews
2. Track sentiment trends
3. Identify negative feedback

### Multi-label Classification
**Text Categorization**
1. Route support tickets to teams
2. Tag documents by category
3. Auto-categorize products

## Best Practices

### Ranking & Reranking
- Use ranking for scoring/sorting
- Use reranking after keyword search
- Combine for best of both speed and accuracy

### Moderation
- Always check user-generated content
- Set appropriate score thresholds (0.5 is typical)
- Log flagged content for review

### Classification
- Use consistent label sets
- Test with representative data
- Monitor confidence scores

### Rate Limiting
- Add delays between requests
- Implement exponential backoff
- Cache results when appropriate

## API Reference Summary

| API | Input | Output | Use Case |
|-----|-------|--------|----------|
| Ranking | Query + Docs | Ranked docs | Score documents |
| Reranking | Query + Results | Reranked results | Semantic search |
| Moderation | Text | Flagged + scores | Safety checking |
| Classification | Text + Labels | Classifications | Text categorization |

## See Also

- [Chat/LLM API](../README.md)
- [Audio API](./audio-api.md)
- [Video API](./video-api.md)
- [Image API](./image-api.md)
