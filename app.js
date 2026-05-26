import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

let indexData = [];
let embedder;

async function init() {
    const statusDiv = document.getElementById('llm-response');
    statusDiv.innerText = "⏳ Loading static text vector database & browser embedding model...";
    
    try {
        const response = await fetch('search_index.json');
        indexData = await response.json();
        
        // Load miniature 23MB vector engine locally into user browser cache
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        statusDiv.innerText = "✅ System ready! Ask a question about your knowledge base.";
    } catch (e) {
        statusDiv.innerText = "⚠️ Missing index database. Drop documents into your repository's 'knowledgebase/' folder to trigger generation.";
    }
}

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function handleSearch() {
    const query = document.getElementById('query').value;
    const apiKey = document.getElementById('apiKey').value;
    const llmDiv = document.getElementById('llm-response');
    if (!query || indexData.length === 0) return;

    llmDiv.innerText = "Converting question to embedding vector...";

    // Generate query embedding vector inside user's browser browser
    const output = await embedder(query, { pooling: 'mean', normalize: true });
    const queryVector = Array.from(output.data);
    
    llmDiv.innerText = "Comparing query against pre-computed knowledge base vectors...";

    // Calculate score against precompiled data vectors
    let matches = indexData.map(chunk => ({
        ...chunk,
        score: cosineSimilarity(queryVector, chunk.vector)
    }));

    // Sort and isolate top 3 highly relevant snippets
    matches.sort((a, b) => b.score - a.score);
    const topMatches = matches.slice(0, 3);

    // Show source context snippets
    document.getElementById('results').innerHTML = topMatches.map(m => `
        <div class="result-box">
            <span class="source-tag">Source: ${m.source}</span> (Relevance Match: ${(m.score * 100).toFixed(1)}%)
            <p>${m.text}</p>
        </div>
    `).join('');

    // If API token is provided, run the free Serverless Inference RAG step
    if (apiKey) {
        llmDiv.innerText = "Streaming response back from LLM Inference API...";
        const contextText = topMatches.map(m => m.text).join("\n---\n");
        
        try {
            const llmRes = await fetch("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2", {
                headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
                method: "POST",
                body: JSON.stringify({
                    inputs: `<s>[INST] Context:\n${contextText}\n\nQuestion: ${query}\n\nAnswer the question concisely using only the context provided above. If it's not in the context, say you don't know. [/INST]`,
                }),
            });
            const data = await llmRes.json();
            llmDiv.innerText = data[0]?.generated_text?.split("[/INST]")?.pop() || "Could not format API output.";
        } catch (e) {
            llmDiv.innerText = "❌ Connection failed to Hugging Face API. Ensure your token is correct.";
        }
    } else {
        llmDiv.innerText = "💡 Matches found! Supply an API key if you want a complete AI written answer.";
    }
}

document.getElementById('searchBtn').addEventListener('click', handleSearch);
window.addEventListener('DOMContentLoaded', init);
