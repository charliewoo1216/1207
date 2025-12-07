전체 코드를 작성해드리겠습니다.

## 1. 환경 설정 파일 (.env)

```plaintext
# .env 파일 - 프로젝트 루트에 생성
OPENAI_API_KEY=your-api-key-here
```

## 2. 메인 애플리케이션 (app.py)

```python
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="간단한 RAG 시스템",
    page_icon="📚",
    layout="wide"
)

# 상수 정의
DOCUMENTS_DIR = "documents"  # 문서 폴더
VECTOR_STORE_PATH = "faiss_index"  # FAISS 인덱스 저장 경로


def load_documents(directory):
    """txt, md 파일을 로드하는 함수"""
    documents = []
    
    # txt 파일 로드
    txt_loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents.extend(txt_loader.load())
    
    # md 파일 로드
    md_loader = DirectoryLoader(
        directory,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents.extend(md_loader.load())
    
    return documents


def split_documents(documents):
    """문서를 청크로 분할하는 함수"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 청크 크기
        chunk_overlap=200,  # 청크 간 겹치는 부분
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    return splits


def create_vector_store(splits):
    """FAISS 벡터 스토어를 생성하는 함수"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def load_vector_store():
    """저장된 FAISS 인덱스를 로드하는 함수"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def create_qa_chain(vectorstore):
    """질의응답 체인을 생성하는 함수"""
    # LLM 설정
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    
    # 프롬프트 템플릿
    prompt_template = """다음 컨텍스트를 사용하여 질문에 답변하세요. 
    답변을 모르면 모른다고 말하고, 억지로 답변을 만들지 마세요.
    
    컨텍스트: {context}
    
    질문: {question}
    
    답변:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 상위 3개 문서 검색
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def initialize_system():
    """시스템을 초기화하는 함수"""
    # documents 폴더 생성
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        st.info(f"📁 '{DOCUMENTS_DIR}' 폴더가 생성되었습니다. 여기에 txt, md 파일을 넣어주세요.")
        return None
    
    # 문서 확인
    doc_files = list(Path(DOCUMENTS_DIR).glob("**/*.txt")) + \
                list(Path(DOCUMENTS_DIR).glob("**/*.md"))
    
    if len(doc_files) == 0:
        st.warning(f"⚠️ '{DOCUMENTS_DIR}' 폴더에 txt 또는 md 파일이 없습니다.")
        return None
    
    # FAISS 인덱스가 이미 존재하는지 확인
    if os.path.exists(VECTOR_STORE_PATH):
        with st.spinner("💾 저장된 벡터 스토어를 로드중..."):
            vectorstore = load_vector_store()
            st.success("✅ 벡터 스토어 로드 완료!")
            return vectorstore
    else:
        # 새로 생성
        with st.spinner("📚 문서를 로드중..."):
            documents = load_documents(DOCUMENTS_DIR)
            st.info(f"📄 {len(documents)}개의 문서를 로드했습니다.")
        
        with st.spinner("✂️ 문서를 분할중..."):
            splits = split_documents(documents)
            st.info(f"📝 {len(splits)}개의 청크로 분할했습니다.")
        
        with st.spinner("🔢 임베딩 생성 및 벡터 스토어 구축중..."):
            vectorstore = create_vector_store(splits)
            # FAISS 인덱스 저장
            vectorstore.save_local(VECTOR_STORE_PATH)
            st.success("✅ 벡터 스토어 생성 및 저장 완료!")
        
        return vectorstore


def main():
    """메인 함수"""
    st.title("📚 간단한 RAG 시스템")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 벡터 스토어 재생성 버튼
        if st.button("🔄 벡터 스토어 재생성"):
            if os.path.exists(VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(VECTOR_STORE_PATH)
                st.success("벡터 스토어가 삭제되었습니다. 페이지를 새로고침하세요.")
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 📖 사용 방법
        1. `documents` 폴더에 txt, md 파일 추가
        2. 앱 실행 (자동으로 벡터화)
        3. 질문 입력
        4. 답변 확인
        
        ### 💡 팁
        - 문서 추가/변경시 '벡터 스토어 재생성' 클릭
        - API 키는 .env 파일에 저장
        """)
    
    # 시스템 초기화
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = initialize_system()
    
    if st.session_state.vectorstore is None:
        st.error("❌ 벡터 스토어를 초기화할 수 없습니다.")
        return
    
    # QA 체인 생성
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)
    
    # 질문 입력
    st.subheader("❓ 질문하기")
    question = st.text_input(
        "질문을 입력하세요:",
        placeholder="예: 문서의 주요 내용은 무엇인가요?"
    )
    
    # 답변 생성
    if st.button("🔍 답변 받기", type="primary"):
        if question:
            with st.spinner("🤔 답변 생성중..."):
                result = st.session_state.qa_chain({"query": question})
                
                # 답변 표시
                st.markdown("### 💬 답변")
                st.write(result['result'])
                
                # 참조 문서 표시
                st.markdown("### 📄 참조 문서")
                for i, doc in enumerate(result['source_documents']):
                    with st.expander(f"문서 {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                        st.write(doc.page_content)
        else:
            st.warning("⚠️ 질문을 입력해주세요.")


if __name__ == "__main__":
    main()
```

## 3. 실행 방법

```bash
# 1. 프로젝트 폴더 구조 생성
mkdir my_rag_project
cd my_rag_project

# 2. .env 파일 생성 (위 내용 복사)
# 3. app.py 파일 생성 (위 코드 복사)

# 4. documents 폴더에 txt, md 파일 추가
mkdir documents
# 여기에 텍스트 파일들을 넣으세요

# 5. Streamlit 앱 실행
streamlit run app.py
```

## 4. 주요 기능

✅ **자동 문서 로드**: documents 폴더의 txt, md 파일 자동 인식
✅ **FAISS-CPU 사용**: 빠른 로컬 벡터 검색
✅ **영구 저장**: 벡터 인덱스를 파일로 저장하여 재사용
✅ **참조 문서 표시**: 답변의 근거가 된 문서 확인 가능
✅ **간단한 UI**: Streamlit으로 직관적인 인터페이스 제공

## 5. 테스트용 샘플 문서

`documents/sample.txt` 파일을 만들어 테스트해보세요:

```text
인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하는 기술입니다.
머신러닝은 AI의 하위 분야로, 데이터로부터 학습하는 알고리즘을 다룹니다.
딥러닝은 인공신경망을 사용하는 머신러닝의 한 종류입니다.
```

이제 앱을 실행하고 "인공지능이란 무엇인가요?"와 같은 질문을 해보세요!
