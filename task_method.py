from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

def chunking(documents_df_, chunk_size_, chunk_overlap_):
    """
    chunk_size_, chunk_overlap_에 따라, documents_df_의 텍스트를 Chunking
    """
    chunk_results = []
    chunk_idx = 0
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_,
        chunk_overlap=chunk_overlap_,
        length_function=len
    )

    for idx, row in documents_df_.iterrows():
        origin_text = row['summary']

        chunks = text_splitter.create_documents([origin_text])

        for chunk in chunks:
            chunk_results.append({
                'chunk_id':chunk_idx,
                'text':chunk.page_content,
                'metadata':{
                    'category':row['category'],
                    'press':row['press'],
                    'title':row['title'],
                    'chunk_size':len(chunk.page_content)
                }
            })
            chunk_idx += 1

    return chunk_results

def retrieval(embedding_model_name: str, collection_name: str, query: str, top_k: int, use_instruct_prefix=False):
    """
    Query를 임베딩하고, Qdrant 컬렉션에서 관련 문서를 검색.
    """

    # 임베딩 모델 로드
    embedding_model = SentenceTransformer(embedding_model_name)

    # 쿼리 임베딩
    if use_instruct_prefix: # Instruct 모델을 사용하는 경우 'query: ' 접두사 추가
        query = f"query: {query}"

    query_vector = embedding_model.encode(query).tolist()

    # Qdrant Client 로드
    client = QdrantClient(host='localhost', port=6333)

    # Qdrant 검색 수행
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True, # 검색된 포인트의 메타 데이터
        with_vectors=False # 검색된 포인트의 임베딩 벡터 (필요 X)
    )

    # 결과 출력
    if not search_results:
        print(f"ERROR: 검색 결과가 없습니다.")
        
        return
    
    for rank, result in enumerate(search_results):
        payload = result.payload

        chunk_text = payload.get('text', '텍스트 없음')
        source_title = payload.get('title', '제목 없음')
        source_press = payload.get('press', '출처 없음')

        print(f"[{rank}]: \nchunk_text: {chunk_text}\nsource_title: {source_title}\nsource_press: {source_press}")

def generation(client, llm_model_params, query:str, search_results: List[Dict]):
    """
    Retrieval 결과와 Query를 바탕으로 답변을 생성

    Args:
        query (str): 사용자 질문
        search_results (List[Dict]): Qdrant 검색 결과 (payload) 포함
    """

    # Context 조합
    context_list = []

    for rank, result in enumerate(search_results):
        context_text = result.payload.get('text', 'no text')
        context_list.append(f"[{rank+1} {context_text}]")

    context = "\n--\n".join(context_list)

    # 프롬프팅 전략
    # System Instruction: Role Prompting
    system_prompt = (
        "당신은 주어진 Context에 기반하여 사용자의 질문에 정확하고 간결하게 답변하는 전문가입니다."
        "답변은 오직 Context에 있는 정보만 사용해야 합니다. 만약 Context에서 질문에 대한 답을 찾을 수 없다면, '관련 정보를 찾을 수 없음'이라고만 답변하세요."
    )

    # User Prompt: CoT & 출처 명시 형식 요구
    user_prompt = (
        "주어진 Context를 분석하여 다음 Query에 답변하는 단계를 '추론' 섹션에 작성하고, 최종 답변을 '답변' 섹션에 작성하세요. 최종 답변에는 참조한 문맥의 번호([N])를 반드시 명시해야 합니다."
        f"문맥(Context):\n{context}\n"
        f"질문(Query):\n{query}\n"
        f"출력 형식: \n"
        f"추론: [답변 도출 과정]\n"
        f"답변: [최종 답변 내용(반드시 출처 번호 명시)]"
    )

    try:
        response = client.models.generate_content(
            model=llm_model_params['model'],
            contents=[
                {'role': 'user', 'parts': [{'text': user_prompt}]}
            ],
            config={
                'system_instruction': system_prompt,
                'temperature': llm_model_params['temperature'],
                'max_output_tokens': llm_model_params['max_tokens']
            }
        )
        return response.text
    
    except Exception as e:
        return f"Gemini API 답변 생성 중 오류 발생: {e}"
