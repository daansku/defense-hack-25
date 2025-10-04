# server/llm.py
from openai import OpenAI
from server.rag import MilitaryRAG

class ReportGenerator:
    def __init__(self):
        self.client = OpenAI()
        self.rag = MilitaryRAG()
    
    def generate_spotrep(self, alert: dict):
        """Generate SPOTREP from camera alert"""
        
        # 1. Search for relevant doctrine
        rag_results = self.rag.search(
            f"SPOTREP format for {alert['type']} detection"
        )
        
        # 2. Build context from RAG
        context = "\n\n".join([
            f"[{r['source']}, p.{r['page']}]: {r['text']}" 
            for r in rag_results
        ])
        
        # 3. Create prompt
        prompt = f"""
You are a military operations assistant. Generate a SPOTREP (Spot Report) based on this detection:

DETECTION DATA:
- Type: {alert['type']}
- Count: {alert['count']}
- Location: {alert['location']}
- Timestamp: {alert['timestamp']}
- Confidence: {alert['confidence']}
- Movement: {alert.get('movement', 'stationary')}

REFERENCE DOCUMENTS:
{context}

Generate a standard military SPOTREP following the format found in the reference documents.
Include:
- All required SPOTREP lines (A through I)
- Confidence levels for each observation
- Citations to source documents
- Assessment of threat level

Format: Use clear military report structure with line designations.
"""
        
        # 4. Call LLM
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a military report writer. Always cite sources and include confidence levels."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Low temp = more consistent/factual
        )
        
        return response.choices[0].message.content