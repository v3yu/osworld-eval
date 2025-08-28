# Analysis Tools for GUI Agent

This document describes the new analysis tools that have been added to the GUI Agent framework to enhance its capabilities for web page analysis and content extraction.

## Overview

The analysis tools provide the following capabilities:
- **Comprehensive Content Analysis**: Complete page analysis including text and images
- **Map Search**: Navigate to Google Maps for geographical searches

## Tools

### 1. Content Analyzer Tool (`content_analyzer`)

Comprehensive page analysis that combines text parsing and image analysis in a single tool.

**Parameters:**
- `query` (string): Query or context for analysis (e.g., "What products are shown?")
- `reasoning` (string): Reasoning for why this content analysis is necessary

**Returns:** JSON object with comprehensive analysis including:
- Page content (parsed using SimpleDocParser)
- Image analysis with CLIP-based relevance ranking
- Summary combining text and image insights
- Query-specific analysis and insights

**Features:**
- **Unified analysis**: Combines page parsing and image analysis in one tool
- **CLIP-based image ranking**: Uses CLIP to find most relevant images to the query
- **Smart fallback**: Falls back gracefully if components fail
- **Comprehensive summary**: Generates insights combining text and image analysis
- **Efficient processing**: CLIP model initialized once in constructor

**Example Usage:**
```json
{
  "query": "What products are shown on this page?",
  "reasoning": "Need to understand the page content and images"
}
```

### 2. Map Search Tool (`map_search`)

Navigates to Google Maps for geographical searches, allowing the agent to interact directly with the map interface.

**Parameters:**
- `query` (string): Search query for map information (e.g., "Nanning China location")
- `reasoning` (string): Reasoning for why this map search is necessary

**Returns:** Success message indicating navigation to Google Maps with the search query.

**Features:**
- **Direct navigation**: Takes the agent directly to Google Maps with the search query
- **Interactive search**: Agent can then search and interact with the map interface
- **URL encoding**: Properly encodes search queries for Google Maps URLs
- **Simple interface**: No complex analysis, just navigation to the right place

**Example Usage:**
```json
{
  "query": "Nanning China location",
  "reasoning": "Need to search for Nanning's location on Google Maps"
}
```

### 4. Content Analyzer Tool (`content_analyzer`)

Analyzes page content and extracts key information, facts, and insights.

**Parameters:**
- `analysis_focus` (string): Focus area for analysis
  - `"facts"`: Extract factual information
  - `"entities"`: Extract named entities (locations, organizations, dates, numbers)
  - `"sentiment"`: Analyze sentiment
  - `"structure"`: Analyze page structure
  - `"all"`: All analysis types
- `reasoning` (string): Reasoning for why this content analysis is necessary

**Returns:** JSON object with:
- Page URL and title
- Extracted facts from content
- Named entities (locations, dates, numbers)
- Page structure analysis (headings, paragraphs, lists, tables, forms)
- Sentiment analysis (positive/negative/neutral)

**Example Usage:**
```json
{
  "analysis_focus": "all",
  "reasoning": "Need to understand the key information on this page"
}
```

## Integration with GUI Agent

The analysis tools are integrated into the GUI Agent framework and can be used alongside the existing GUI interaction tools (click, type, scroll, etc.). When an analysis tool is called:

1. The tool receives the current page context from the browser environment
2. It analyzes the page content according to the specified parameters
3. The results are logged and the agent can use the information for decision-making
4. A wait action is returned to allow the agent to process the analysis results

## Dependencies

The analysis tools require the following Python packages:
- `beautifulsoup4`: For HTML parsing
- `requests`: For HTTP requests (if needed)
- `pillow`: For image processing (if needed)

## Testing

A test script `test_analysis_tools.py` is provided to verify the functionality of all analysis tools. The script includes mock page contexts and demonstrates how each tool works.

## Usage Examples

### Example 1: Comprehensive Page Analysis
```json
{
  "function": "content_analyzer",
  "arguments": {
    "query": "What products are shown on this page?",
    "reasoning": "Need to understand the page content and images"
  }
}
```

### Example 2: Searching for Geographical Information
```json
{
  "function": "map_search",
  "arguments": {
    "query": "Beijing location in China",
    "reasoning": "Need to search for Beijing's location on Google Maps"
  }
}
```

## Error Handling

The tools include comprehensive error handling:
- Graceful handling of missing page context
- Fallback mechanisms for page access
- JSON parsing error recovery
- Detailed error logging

## Future Enhancements

Potential future enhancements include:
- Integration with external APIs for enhanced analysis
- Support for more sophisticated image analysis using computer vision
- Real-time map data integration
- Advanced natural language processing for content analysis
- Support for dynamic content and JavaScript-rendered pages 