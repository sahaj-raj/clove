def build_rag_prompt(query: str, retrived_chunks: list[dict], chat_history: list[dict] = None) -> str:

    context_blocks = []

    for chunk in retrived_chunks:
        block = (f"<chunk id='{chunk['chunk_id']}' page='{chunk.get('page', 'unknown')}'>\n"                                                                            
                f"{chunk['text']}\n"                                                                                                                                   
                f"</chunk>"
                 )

        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    history_str =" "
    if chat_history:
        history_lines = []
        for msg in chat_history:
            role = "User" if msg['role'] == "user" else "assistant"
            history_lines.append(f"{role} : {msg['content']}")
        history_str = "<conversation_history>\n" + "\n".join(history_lines) + "\n</conversation_history>\n\n" 


    prompt = f"""You are a precise, evidence-based research assistant.                                                                                             
        Your task is to answer the user's question using ONLY the facts provided in the <context> below.                                                                   
                                                                                                                                                                        
        STRICT RULES:                                                                                                                                                      
        1. Every claim you make MUST cite its source chunk using [Chunk ID] (e.g., [Chunk 3]).                                                                             
        2. If the answer cannot be found directly in the <context>, reply with:                                                                                            
        "I cannot answer this question based on the provided document."                                                                                                 
        3. Do NOT make assumptions or use external knowledge outside the <context>.
        {history_str}
        <context> {context_str} </context>

        Question : {query}

        Answer:"""
    return prompt

if __name__ == "__main__":                                                                                                                                         
        sample_chunks = [                                                                                                                                              
            {"chunk_id": 3, "page": 1, "text": "In 1925, light intensity affected Drosophila lifespan."}                                                               
        ]                                                                                                                                                              
        sample_history = [                                                                                                                                             
            {"role": "user", "content": "Tell me about aging in 1925."},                                                                                               
            {"role": "assistant", "content": "Research in 1925 studied Drosophila [Chunk 3]."}                                                                         
        ]                                                                                                                                                              
                                                                                                                                                                       
        print("--- TESTING PROMPT WITH HISTORY ---")                                                                                                                   
        print(build_rag_prompt("What animal was studied?", sample_chunks, sample_history))                                                                             
                                                                                    