# Industrial-Text-Analytics_Term-Project
25-2학기 산업텍스트애널릭틱스 텀 프로젝트 : Vector DB와 LLM을 결합한 RAG 챗봇 개발

**목적 및 과업**
- Vector DB: 검색 시스템 구축을 위해 널리 활용되는 벡터 디비 이용
    - 기존 vector db 구현 프레임워크(faiss, qdrant, weaviate 등)의 기본 활용 방법론 확인
- LLM: llm 서빙을 위해 널리 활용되는 가속화 프레임워크 이용
    - 기존 llm 서빙을 위해 널리 활용되는 vllm 등의 기본 활용 방법론 확인
- RAG: 검색 및 답변 생성 전반에 걸친 디자인 요소 고려
    - vector db와 llm 간의 소통 및 최종 답변 생성을 위한 다양한 검색/chunking/프롬프팅/개선 전략 적용
- 최종 구현체: GRADIO
    - 사용자 단일 질의에 대한 Top-k 개 검색 결과와 이를 활용한 답변 제공

**과업 요약**
- Open-source LLM을 사용한 한국어 챗봇 기능 구현
- Open-source Retriever를 사용한 vector db 및 검색 시스템 구축

**과제 구성**
- (필수)Task 1: Vector DB를 이용한 검색 시스템 구축
    - 입력: 키워드 및 질문
    - 출력: 관련된 문서 K개
    - 파이프라인:
        1. **수집**: huggingface datasets에 공개된 한국어 뉴스 요약 데이터셋 ([https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko/viewer?views[]=train](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko/viewer?views%5B%5D=train))
        2. 검색 파이프라인: 널리 활용되는 아래 vector db 중 선택하여 사용
            1. Qdrant: https://qdrant.tech/
            2. Weaviate: https://docs.weaviate.io/weaviate
    - 주요 사항
        - 문서 전처리 방식: Chunkning 전략, 문서 최소/최대 길이 및 기타 데이터 전처리
        - Retriever 모델 선택: 검색 성능이 높은 Retriever 모델을 선택하는 기준 명시

- (필수)Task 2: LLM을 이용한 답변 생성 시스템 구축
    - 입력: 키워드 및 질문, 검색 결과 K개 문서
    - 출력: 질문에 대한 답변
    - 파이프라인:
        1. LLM을 이용한 답변 생성 시스템 구축
            1. 다양한 한국어 제공 LLM을 통한 답변 생성 시스템 구축
    - 주요 사항
        - 프롬프팅 방식: 개선된 답변 생성을 위해 적용가능한 프롬프팅 전략은 무엇이 있는가?
            - CoT, Reflection, Self-Evaluation
        - 생성 방식: 가정한 서비스 상황에서 적절한 생성 파라미터는 무엇인가?
            - Temperature, Max length, model size
            
- (선택)Task 3: Gradio 서비스화
    - 단순 기능 구현이 아닌, 서비스 배포를 위한 PoC 구현
    - 한 화면 구성(권장)
        - 검색 결과 문서 K개
        - 최종 답변
    - 위에 포함된 기능 이외에, 편의성을 위해 추가적으로 기능을 더 구현하여도 무방

**Deliverables**
- 프로젝트  결과  보고서 (PPT)
    - ① 문제정의 & 사용자 시나리오
    - ② 시스템 동작 구조
        - Ex. Task 1: 수집→정제→요약→표시
    - ③ **가설 카드 (3개)**
        - 아래 설명 참조
    - ⑤ 실제 작동 예시
    - ⑥ 결론
- 사용한 소스코드 프로젝트
- 프로젝트 최종 발표 영상 (15 min)

**가설 카드**
- 프로젝트 중 수행한 문제 정의 및 해결 과정을 정리한 카드
- 한 카드 당 하나의 가설이 기록됨
- 결과 보고서에는 가장 중요하게 작용했던 가설 3개를 골라 그에 대한 가설 카드를 작성
- 구성
    - As-is (H0)
        - 현재 상황
        - 문제점
    - To-be (H1)
        - 개선 방안 및 그 구현 방법
        - 개선될 수 있는 이유
    - Validation Protocol
        - 개선 여부 측정 방법 및 정당성
    - Result
        - V.P의 수행 결과