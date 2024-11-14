from flask import Flask, render_template, request, jsonify, session
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')

# FAQ 시나리오 구조 정의
disaster_scenarios = {
    '지진': {
        '고층 건물 안에 있는 상황': {
            'questions': [
                {
                    'text': '창문 근처로 대피한다',
                    'is_correct': False,
                    'feedback': '창문은 깨질 수 있어 매우 위험합니다. 책상 아래나 내부 기둥 근처로 대피하세요.'
                },
                {
                    'text': '엘리베이터를 타고 대피한다',
                    'is_correct': False,
                    'feedback': '지진 시 엘리베이터는 매우 위험합니다. 계단을 이용해 대피하세요.'
                },
                {
                    'text': '책상 아래로 들어가 몸을 보호한다',
                    'is_correct': True,
                    'feedback': '올바른 선택입니다. 책상 아래로 들어가 몸을 보호하고, 흔들림이 멈출 때까지 기다리세요.'
                },
                {
                    'text': '질문하기',
                    'is_correct': 'question',
                    'feedback': '챗봇에게 궁금한 점을 물어보세요.'
                }
            ],
            'next_situation': '대피로 확인하기'
        },
        '대피로 확인하기': {
            'questions': [
                {
                    'text': '엘리베이터로 이동한다',
                    'is_correct': False,
                    'feedback': '지진 시 엘리베이터는 사용하면 안됩니다.'
                },
                {
                    'text': '계단으로 이동한다',
                    'is_correct': True,
                    'feedback': '계단을 이용해 안전하게 대피하는 것이 올바른 방법입니다.'
                },
                {
                    'text': '제자리에서 기다린다',
                    'is_correct': False,
                    'feedback': '2차 피해를 방지하기 위해 안전한 장소로 대피해야 합니다.'
                },
                {
                    'text': '질문하기',
                    'is_correct': 'question',
                    'feedback': '챗봇에게 궁금한 점을 물어보세요.'
                }
            ],
            'next_situation': None
        }
    }
    # 다른 재난 상황들도 같은 형식으로 추가
}

# PDF 로더 및 텍스트 처리
def initialize_qa_system(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4"
        ),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# 시스템 초기화
qa_chain = initialize_qa_system('./pdf/부산광역시교육청_학교현장재난안전119.pdf')

@app.route('/')
def chatbot():
    return render_template('home.html')

@app.route('/main')
def index():
    return render_template('index.html', disasters=list(disaster_scenarios.keys()))

@app.route('/situation')
def chatbot2():
    return render_template('situation.html')

@app.route('/chatbotPage')
def chatbotPage():
    return render_template('chatbotPage.html')


@app.route('/scenario/<disaster>/<situation>', methods=['GET'])
def get_situation(disaster, situation):
    if disaster in disaster_scenarios and situation in disaster_scenarios[disaster]:
        return jsonify(disaster_scenarios[disaster][situation])
    return jsonify({'error': '상황을 찾을 수 없습니다.'}), 404

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    user_stage = data.get('stage')
    chat_history = session.get('chat_history', [])
    
    # 훈련 프롬프트 설정
    initial_prompt = (
        "당신은 재난 안전 대피에 관한 안내를 제공하는 전문가입니다. 사용자에게 도움을 줄 수 있도록 "
        "재난 상황에서 안전하게 행동하는 방법을 알려주세요. 다음은 사용자의 질문입니다:\n"
    )
    full_prompt = initial_prompt + user_message

    # OpenAI 모델에 훈련 프롬프트 포함하여 응답 생성
    result = qa_chain({
        "question": full_prompt,
        "chat_history": chat_history
    })
    
    chat_history.append((user_message, result['answer']))
    session['chat_history'] = chat_history

    user_stage = 2
    
    return jsonify({
        'is_correct' : 'False',
        'stage' : user_stage,
        'message': result['answer']
    })

@app.route('/check_answer', methods=['POST'])
def check_answer():
    data = request.json
    disaster = data.get('disaster')
    situation = data.get('situation')
    answer_index = data.get('answer_index')

    print(data)

    if disaster not in disaster_scenarios or situation not in disaster_scenarios[disaster]:
        return jsonify({'error': '상황을 찾을 수 없습니다.'}), 404
    
    scenario = disaster_scenarios[disaster][situation]
    question = scenario['questions'][answer_index]
    
    response = {
        'is_correct': question['is_correct'],
        'feedback': question['feedback'],
        'next_situation': scenario.get('next_situation')
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
