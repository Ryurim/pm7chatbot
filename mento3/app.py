from flask import Flask, render_template, request, jsonify, session
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import glob
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')

# FAQ 시나리오 구조 정의
# stage 1,2,3 정의해주기
# is_correct = 'True', 'False'
# 스테이지 3 일 떄, is_correct = 'end' 라고 던져주기
disaster_scenarios = {
    '평화로운 주말 아침, 배고파서 라면을 끓이려고 물을 올렸다. 그런데 갑자기 집이.. 흔들린다..?': {
        1: {
            'stage': '라면이 끓고 있는 가스레인지는 흔들리고, 주변 물건들이 떨어지기 시작하고 있어요! 지금 어떻게 해야 할까요?',
            'questions': [
                {'text': '가스레인지를 끄고, 창문 가까이로 가서 밖을 확인할래', 'is_correct': False, 'feedback': '창문 근처는 위험해요. 유리창이 깨지면 큰 부상을 당할 수 있어요. 무조건 창문 근처로 가는 것을 피해야 해요.'},
                {'text': '당장 식탁이나 튼튼한 가구 밑으로 들어갈래', 'is_correct': True, 'feedback': '잘했어요! 지진이 발생하면 안전한 장소에 몸을 보호하는 것이 중요해요. 테이블 아래나 튼튼한 가구 밑으로 들어가면 떨어지는 물건으로부터 몸을 보호할 수 있어요.'},
                {'text': '라면을 마저 끓이고 상황을 지켜볼래. 지진은 금방 끝날 것 같아!', 'is_correct': False, 'feedback': '지진 상황에서는 음식보다 안전이 훨씬 중요해요. 상황을 지켜보는 대신 빠르게 안전한 장소로 대피해야 해요!'},
                {'text': '잘 모르겠어. 나 질문해도 돼?', 'is_correct': 'question', 'feedback': '삐용이에게 궁금한 점을 물어보세요.'},
                {'text': '나 질문 있어!', 'is_correct': 'question2', 'feedback': '삐용이에게 궁금한 점을 물어보세요.'}
            ],
            'next_stage': 2
        },
        2: {
            'stage': '이제는 지진이 잠시 멈췄지만 추가 여진이 있을지도 몰라요. 주방에는 여전히 떨어질 물건들이 많고 가스레인지가 꺼져 있지 않아요. 지금 어떤 행동을 해야 할까요?',
            'questions': [
                {'text': '가스레인지를 잠그고 안전한 위치에서 대기하며 여진에 대비해요.', 'is_correct': True, 'feedback': '잘했어요! 지진이 발생하면 안전한 장소에 몸을 보호하는 것이 중요해요. 가스레인지를 먼저 끄고, 추가 여진에 대비해 최대한 안전한 자세로 대기하는 것이 중요해요.'},
                {'text': '주방을 정리하며 떨어진 물건들을 정돈해요. 어수선한 주방을 깨끗하게 해야 안전할 것 같아요!', 'is_correct': False, 'feedback': '여진이 발생할 가능성이 높기 때문에 지금은 정리보다 몸을 보호하는 것이 우선이에요.'},
                {'text': '주방에서 나가 친구들에게 전화해요. 모두 괜찮은지 확인해야겠어요.', 'is_correct': False, 'feedback': '대피 전에 연락보다는 자신의 안전을 확보하는 것이 우선이에요. 연락은 추후에 하는 것이 좋겠어요.'},
                {'text': '잘 모르겠어. 나 질문해도 돼?', 'is_correct': 'question', 'feedback': '챗봇에게 궁금한 점을 물어보세요.'},
                {'text': '나 질문 있어!', 'is_correct': 'question2', 'feedback': '삐용이에게 궁금한 점을 물어보세요.'}
            ],
            'next_stage': 3
        },
        3: {
            'stage': '몇 분 후, 추가 여진 경보가 울리며 모든 사람에게 건물 밖으로 대피하라는 안내가 나오고 있어요. 집 밖으로 가려는 순간, 엘리베이터와 계단 중 무엇을 사용할지 고민이 되는데 어떻게 할까요? 그 이유까지 함께 대답해 주세요.',
            'questions': [
                {'text': '엘리베이터를 이용해서 빠르게 나간다.', 'is_correct': False},
            ],
            'next_stage': 'end'
        }
    }
}


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# PDF 로더 및 FAISS 인덱스 초기화
def initialize_qa_system(pdf_folder_path, faiss_index_path="faiss_index"):
    # FAISS 인덱스 파일이 존재하면 불러오기
    if os.path.exists(faiss_index_path):
        print(f"저장된 FAISS 인덱스 '{faiss_index_path}'를 불러옵니다.")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("저장된 FAISS 인덱스를 찾을 수 없습니다. PDF 파일에서 새로 생성합니다.")
        all_texts = []
        pdf_files = glob.glob(f"{pdf_folder_path}/*.pdf")
        
        # 각 PDF 파일을 하나씩 로드
        for pdf_path in pdf_files:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 텍스트 분할 설정
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            all_texts.extend(texts)
        
        # 임베딩 및 벡터 스토어 생성 및 저장
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(all_texts, embeddings)
        vectorstore.save_local(faiss_index_path)
        print(f"새로운 FAISS 인덱스가 '{faiss_index_path}'에 저장되었습니다.")
    
    # 질의응답 체인 반환
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4o-mini"
        ),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# 시스템 초기화 (PDF 파일 또는 저장된 FAISS 인덱스를 사용)
qa_chain = initialize_qa_system('./pdf', './faiss_data/')

@app.route('/')
def chatbot():
    return render_template('home.html')
@app.route('/main')
def index():
    return render_template('index.html')
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
    user_stage = int(data.get('stage'))
    chat_history = session.get('chat_history', [])
    prev_answer = data.get('prev')
    previous_question_type = session.get('previous_question_type')

    # disaster_scenarios에서 현재 스테이지 정보 가져오기
    disaster_data = disaster_scenarios.get('평화로운 주말 아침, 배고파서 라면을 끓이려고 물을 올렸다. 그런데 갑자기 집이.. 흔들린다..?')
    current_stage = disaster_data.get(user_stage) if disaster_data else None

    if not current_stage:
        return jsonify({'error': 'Invalid stage'}), 400

    
    if user_stage == 1:
        # 사용자의 입력과 질문의 일치 여부를 확인
        matched = False
        for question in current_stage['questions']:
            if question['text'] == user_message:
                matched = True
                is_correct = question['is_correct']
                    
                # 질문 선택 시 OpenAI 모델 응답
                if is_correct == 'question':
                    session['previous_question_type'] = 'question'
                    return jsonify({
                        'stage': user_stage,
                        'is_correct': prev_answer,
                        'message': "물론이죠. 궁금한 점을 물어봐요:)"
                    })

                # 질문 선택 시 OpenAI 모델 응답
                elif is_correct == 'question2':
                    session['previous_question_type'] = 'question2'
                    return jsonify({
                        'stage': user_stage,
                        'is_correct': prev_answer,
                        'message': "물론이죠. 궁금한 점을 물어봐요:)"
                    })

                elif is_correct == True:
                    # 정답 처리
                    chat_history = []
                    initial_prompt =( f'''
                    '{user_message}'은 정답이야. 학생들에게 잘했다는 칭찬과 함께 왜 정답인지 초등학생 저학년이 이해할 수 있도록 간단하게 설명해줘
                    답변은 -어요, -이에요/예요, -(이)여요, -(이)요 형태로 끝나야 해 
                    마크업 언어 '**' 사용하지마
                    ''')

                    print(initial_prompt)
                    llm_response = qa_chain({
                        "question": initial_prompt + user_message,
                        "chat_history": chat_history
                    })

                    chat_history.append((user_message, llm_response['answer']))
                    session['chat_history'] = chat_history

                    return jsonify({
                        'stage': user_stage + 1,
                        'is_correct': 'True',
                        'message': llm_response['answer']
                    })

                elif is_correct == False:
                    chat_history = []
                    initial_prompt =( f'''
                    '{user_message}'은 오답이기 때문에 오답인 이유를 초등학생 저학년이 이해할 수 있도록 간단하게 설명해줘
                    오답인 이유를 설명할 때는 부드럽게 위로하며 말해줘
                    정답인 당장 식탁이나 튼튼한 가구 밑으로 들어갈래 에 대한 정보는 절대 말하면 안돼
                    답변은 -어요, -이에요/예요, -(이)여요, -(이)요 형태로 끝나야 해 
                    마크업 언어 '**' 사용하지마
                    ''')
                    # 오답 처리
                    llm_response = qa_chain({
                        "question": initial_prompt + user_message,
                        "chat_history": chat_history
                    })

                    chat_history.append((user_message, llm_response['answer']))
                    session['chat_history'] = chat_history

                    return jsonify({
                        'stage': user_stage,
                        'is_correct': 'False',
                        'message': llm_response['answer']
                    })

        # 일치하는 질문이 없는 경우 기본 OpenAI 모델 응답 처리
        if not matched:
            chat_history = []
            print(previous_question_type)
            if previous_question_type == 'question': ## 잘 모르겠어. 나 질문해도 돼?
                initial_prompt = (
                    ## 필수적으로 들어가야 하는 챗봇의 성격
                    #역할 부여
                    "너는 초등학교 범교과에서 재난과 관련된 퀴즈를 제공하는 챗봇, 삐용챗이야."
                    "너는 학생들과 재난 상황 속에서 퀴즈를 풀면서 재난 대피를 하고 있는 시뮬레이션 퀴즈를 풀고 있어"
                    "너의 주 이용자는 초등학생이야."
                    "너가 누구라는 질문을 받으면 '재난안전 관련 퀴즈를 제공하는 삐용이'라고 답해줘"

                    #태도 
                    "재난 상황에서 안전하게 행동하는 방법을 알려줘. 다음은 학생의 질문이야:\n"
                        f"'{user_message}'"
                    "모든 대화는 높임말로 답변하면서 해요로 끝나게 해줘"
                    "이용자가 초등학생이기 때문에 초등학생들이 이해하기 쉬운 말을 사용해줘야 해"
                    "모든 말을 적극적 말하기 기법을 활용하여 학생들의 답변에 공감과 반응으로 시작해줘"

                    ## 질문 상황에 필요한 프롬프트 
                    #주의
                    "대화를 할 때는 퀴즈에 대한 정답을 알려주지 않도록 해야 해"
                    "정답을 직설적으로 요구하는 경우, 정답을 요구하지 말라고 말하면서 적절한 힌트만 제공해줘"
                    "학생이 퀴즈를 그만하고 싶다는 답변을 한다면, 포기하지 말라는 말과 함께 정답을 풀 수 있도록 도와줘"    
                    "모든 답변은 200자를 넘기지 마" 
                    " 비난, 비방, 폭언, 욕설 등의 질문을 받는다면, 나쁜 말이라며 사용하지 말라고 확실하게 답해줘. "
                    "답변을 할 때는 퀴즈에 대한 정답을 알려주지 않도록 해야 해"
                    "문장이 너무 길면 읽기 힘들기 때문에 학생이 읽기 편하도록 문단 안에서 줄바꿈을 2번 해줘."
                    

                    #재난과 관련 없을 때 
                    "학생들이 재난 상황과 관련 없는 질문을 한다면 '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "지진, 화재, 태풍, 집중호우, 황사, 미세먼지, 폭염, 대설, 한파와 관련 없는 질문을 할 때에는 '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "학생이 라면을 제외한 다른 음식을 말했을 때, '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "마라탕이라는 단어가 입력될 경우, '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "학생이 도와줘라고 말했을 경우에는 도와줄 수 있으니 편하게 질문하라고 대답해줘"
                    "학생이 모르겠다고 할 때는 할 수 있다는 격려와 함께 모르는 부분이 무엇인지 자세히 말해주면 도와줄 수 있다고 말해줘"
                    
                    #애매한 질문 예시
                    #학생이 도와달라고만 했을 때 / 학생이 모르겠다고만 했을 때
                    "어떤 부분에서 어려움을 느껴지나요? 자세하게 말해주면 제가 도와줄 수 있어요"

                    #1단계에서 상황
                    "학생들은 지금은 1단계야"
                    "학생들은 라면을 끓이는 와중에 갑자기 지진이 난 상황을 맞이하고 있어"
                    "현재 이용자는 라면을 끓이고 있는 와중에 갑자기 지진이 발생했어."
                    "이용자는 혼자 라면을 끓이고 있었고, 집은 고층 아파트에 해당해."
                    "라면을 끓이던 가스레인지도 흔들리고, 주변에서는 물건들이 떨어지고 있어."
                    "학생들에게는 현재 상황에서 가장 먼저 무슨 행등을 할지 물어본 상태야"
                    "라면을 끓이고 있다가도 지진이 나 흔들림이 느껴지면 가스불 보다 먼저 안전한 곳으로 대피하는 것이 더 중요하다는 사실을 기억하고 대답해줘"  
                    "1단계 퀴즈와 관련된 힌트를 줘"
                    "절대 {당장 식탁이나 튼튼한 가구 밑으로 들어갈래} 정답을 그대로 알려 주면 안돼"
                )
                
            if previous_question_type == 'question2': # 나 질문있어
                initial_prompt = (
                    ## 필수적으로 들어가야 하는 챗봇의 성격
                    #역할 부여
                    "너는 초등학교 범교과에서 재난과 관련된 퀴즈를 제공하는 챗봇, 삐용챗이야."
                    "너는 학생들과 재난 상황 속에서 퀴즈를 풀면서 재난 대피를 하고 있는 시뮬레이션 퀴즈를 풀고 있어"
                    "너의 주 이용자는 초등학생이야."
                    "너가 누구라는 질문을 받으면 '재난안전 관련 퀴즈를 제공하는 삐용이'라고 답해줘"

                    #태도 
                    "재난 상황에서 안전하게 행동하는 방법을 알려줘. 다음은 학생의 질문이야:\n"
                        f"'{user_message}'"
                    "모든 대화는 높임말로 답변하면서 해요로 끝나게 해줘"
                    "이용자가 초등학생이기 때문에 초등학생들이 이해하기 쉬운 말을 사용해줘야 해"
                    "모든 말을 적극적 말하기 기법을 활용하여 학생들의 답변에 공감과 반응으로 시작해줘"

                    ## 질문 상황에 필요한 프롬프트 
                    #주의
                    "대화를 할 때는 퀴즈에 대한 정답을 알려주지 않도록 해야 해"
                    "정답을 직설적으로 요구하는 경우, 정답을 요구하지 말라고 말하면서 적절한 힌트만 제공해줘"
                    "학생이 퀴즈를 그만하고 싶다는 답변을 한다면, 포기하지 말라는 말과 함께 정답을 풀 수 있도록 도와줘"    
                    "모든 답변은 200자를 넘기지 마" 
                    " 비난, 비방, 폭언, 욕설 등의 질문을 받는다면, 나쁜 말이라며 사용하지 말라고 확실하게 답해줘. "
                    "답변을 할 때는 퀴즈에 대한 정답을 알려주지 않도록 해야 해"
                    "문장이 너무 길면 읽기 힘들기 때문에 학생이 읽기 편하도록 문단 안에서 줄바꿈을 2번 해줘."
                    

                    #재난과 관련 없을 때 
                    "학생들이 재난 상황과 관련 없는 질문을 한다면 '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "지진, 화재, 태풍, 집중호우, 황사, 미세먼지, 폭염, 대설, 한파와 관련 없는 질문을 할 때에는 '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "학생이 라면을 제외한 다른 음식을 말했을 때, '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "마라탕이라는 단어가 입력될 경우, '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "학생이 도와줘라고 말했을 경우에는 도와줄 수 있으니 편하게 질문하라고 대답해줘"
                    "학생이 모르겠다고 할 때는 할 수 있다는 격려와 함께 모르는 부분이 무엇인지 자세히 말해주면 도와줄 수 있다고 말해줘"
                    
                    #애매한 질문 예시
                    #학생이 도와달라고만 했을 때 / 학생이 모르겠다고만 했을 때
                    "어떤 부분에서 어려움을 느껴지나요? 자세하게 말해주면 제가 도와줄 수 있어요"

                    #1단계에서 상황
                    "학생들은 지금은 1단계야"
                    "학생들은 라면을 끓이는 와중에 갑자기 지진이 난 상황을 맞이하고 있어"
                    "현재 이용자는 라면을 끓이고 있는 와중에 갑자기 지진이 발생했어."
                    "이용자는 혼자 라면을 끓이고 있었고, 집은 고층 아파트에 해당해."
                    "라면을 끓이던 가스레인지도 흔들리고, 주변에서는 물건들이 떨어지고 있어."
                    "학생들에게는 현재 상황에서 가장 먼저 무슨 행등을 할지 물어본 상태야"
                    "라면을 끓이고 있다가도 지진이 나 흔들림이 느껴지면 가스불 보다 먼저 안전한 곳으로 대피하는 것이 더 중요하다는 사실을 기억하고 대답해줘"  
                    "학생들 1단계를 대피할 수 있도록 퀴즈와 관련된 힌트를 줄 수 있도록 해줘"
                    "1단계 상황과 관련없는 질문들은 다음 단계로 넘어가면 알 수 있다고 해줘"

                )
            
            # 기본 프롬프트를 사용하여 LLM 응답 생성
            llm_response = qa_chain({
                "question": initial_prompt + user_message,
                "chat_history": chat_history
            })

            chat_history.append((user_message, llm_response['answer']))
            session['chat_history'] = chat_history

            return jsonify({
                'stage': user_stage,
                'is_correct': prev_answer,
                'message': llm_response['answer']
            })


    elif user_stage == 2:
        initial_prompt = "스테이지 2 프롬프트"

        # 사용자의 입력과 질문의 일치 여부를 확인
        matched = False
        for question in current_stage['questions']:
            if question['text'] == user_message:
                matched = True
                is_correct = question['is_correct']
                print("정답여부", is_correct)
                feedback = question['feedback']
                    
                # 질문 선택 시 OpenAI 모델 응답
                if is_correct == 'question':
                    session['previous_question_type'] = 'question'
                    return jsonify({
                        'stage': user_stage,
                        'is_correct': prev_answer,
                        'message': "물론이죠. 궁금한 점을 물어봐요:)"
                    })
                
                # 질문 선택 시 OpenAI 모델 응답
                elif is_correct == 'question2':
                    session['previous_question_type'] = 'question2'
                    return jsonify({
                        'stage': user_stage,
                        'is_correct': prev_answer,
                        'message': "물론이죠. 궁금한 점을 물어봐요:)"
                    })

                elif is_correct == True:
                    # 정답 처리
                    chat_history = []
                    initial_prompt =( f'''
                    '{user_message}'은 정답이야. 학생들에게 잘했다는 칭찬과 함께 왜 정답인지 초등학생 저학년이 이해할 수 있도록 간단하게 설명해줘
                    답변은 -어요, -이에요/예요, -(이)여요, -(이)요 형태로 끝나야 해 
                    마크업 언어 '**' 사용하지마
                    ''')
                    llm_response = qa_chain({
                        "question": initial_prompt + user_message,
                        "chat_history": chat_history
                    })

                    chat_history.append((user_message, llm_response['answer']))
                    session['chat_history'] = chat_history

                    return jsonify({
                        'stage': user_stage + 1,
                        'is_correct': 'True',
                        'message': llm_response['answer']
                    })

                elif is_correct == False:
                    chat_history = []
                    initial_prompt =( f'''
                    '{user_message}'은 오답이기 때문에 오답인 이유를 초등학생 저학년이 이해할 수 있도록 간단하게 설명해줘
                    오답인 이유를 설명할 때는 부드럽게 위로하며 말해줘
                    정답인 가스레인지를 잠그고 안전한 위치에서 대기하며 여진에 대비해요. 대한 정보는 절대 말하면 안돼
                    답변은 -어요, -이에요/예요, -(이)여요, -(이)요 형태로 끝나야 해 
                    마크업 언어 '**' 사용하지마''')

                    llm_response = qa_chain({
                        "question": initial_prompt + user_message,
                        "chat_history": chat_history
                    })

                    chat_history.append((user_message, llm_response['answer']))
                    session['chat_history'] = chat_history

                    return jsonify({
                        'stage': user_stage,
                        'is_correct': 'False',
                        'message': llm_response['answer']
                    })

        # 일치하는 질문이 없는 경우 기본 OpenAI 모델 응답 처리
        if not matched:
            chat_history = []
            print(previous_question_type)
            if previous_question_type == 'question':
                initial_prompt = (
                    ## 필수적으로 들어가야 하는 챗봇의 성격
                    #역할 부여
                    "너는 초등학교 범교과에서 재난과 관련된 퀴즈를 제공하는 챗봇, 삐용챗이야."
                    "너는 학생들과 재난 상황 속에서 퀴즈를 풀면서 재난 대피를 하고 있는 시뮬레이션 퀴즈를 풀고 있어"
                    "너의 주 이용자는 초등학생이야."
                    "너가 누구라는 질문을 받으면 '재난안전 관련 퀴즈를 제공하는 삐용이'라고 답해줘"

                    #태도 
                    "재난 상황에서 안전하게 행동하는 방법을 알려줘. 다음은 학생의 질문이야:\n"
                        f"'{user_message}'"
                    "모든 대화는 높임말로 답변하면서 해요로 끝나게 해줘"
                    "이용자가 초등학생이기 때문에 초등학생들이 이해하기 쉬운 말을 사용해줘야 해"
                    "모든 말을 적극적 말하기 기법을 활용하여 학생들의 답변에 공감과 반응으로 시작해줘"

                    ## 질문 상황에 필요한 프롬프트 
                    #주의
                    "대화를 할 때는 퀴즈에 대한 정답을 알려주지 않도록 해야 해"
                    "정답을 직설적으로 요구하는 경우, 정답을 요구하지 말라고 말하면서 적절한 힌트만 제공해줘"
                    "학생이 퀴즈를 그만하고 싶다는 답변을 한다면, 포기하지 말라는 말과 함께 정답을 풀 수 있도록 도와줘"    
                    "모든 답변은 200자를 넘기지 마" 
                    " 비난, 비방, 폭언, 욕설 등의 질문을 받는다면, 나쁜 말이라며 사용하지 말라고 확실하게 답해줘. "
                    "답변을 할 때는 퀴즈에 대한 정답을 알려주지 않도록 해야 해"
                    "문장이 너무 길면 읽기 힘들기 때문에 학생이 읽기 편하도록 문단 안에서 줄바꿈을 2번 해줘."
                    

                    #재난과 관련 없을 때 
                    "학생들이 재난 상황과 관련 없는 질문을 한다면 '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "지진, 화재, 태풍, 집중호우, 황사, 미세먼지, 폭염, 대설, 한파와 관련 없는 질문을 할 때에는 '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "학생이 라면을 제외한 다른 음식을 말했을 때, '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "마라탕이라는 단어가 입력될 경우, '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "학생이 도와줘라고 말했을 경우에는 도와줄 수 있으니 편하게 질문하라고 대답해줘"
                    "학생이 모르겠다고 할 때는 할 수 있다는 격려와 함께 모르는 부분이 무엇인지 자세히 말해주면 도와줄 수 있다고 말해줘"
                    
                    #애매한 질문 예시
                    #학생이 도와달라고만 했을 때 / 학생이 모르겠다고만 했을 때
                    "어떤 부분에서 어려움을 느껴지나요? 자세하게 말해주면 제가 도와줄 수 있어요"

                    #2단계에서 상황
                    "학생들은 지금 2단계야"
                    "이전 단계에서 학생은 집에서 라면을 끓이고 있는 와중에 지진을 느끼게 되어 책상 밑으로 들어가 대피를 한 상태야"
                    "이제는 흔들림이 멈추었지만 추가 여진이 발생할지도 모르는 상황이야, 주방에는 아직 가스레인지가 켜져 있는 상황이야"
                    "여전히 학생의 주변에는 떨어질 물건이 많은 상태야"
                    "여진 발생의 위험을 기억하면서 대답해줘"
                    
                    "2단계 퀴즈와 관련된 힌트를 줘"
                    "절대 {가스레인지를 잠그고 안전한 위치에서 기다리면서 여진에 대비할래} 정답을 그대로 주면 안돼"
                )
                
            if previous_question_type == 'question2':
                initial_prompt = (
                    ## 필수적으로 들어가야 하는 챗봇의 성격
                    #역할 부여
                    "너는 초등학교 범교과에서 재난과 관련된 퀴즈를 제공하는 챗봇, 삐용챗이야."
                    "너는 학생들과 재난 상황 속에서 퀴즈를 풀면서 재난 대피를 하고 있는 시뮬레이션 퀴즈를 풀고 있어"
                    "너의 주 이용자는 초등학생이야."
                    "너가 누구라는 질문을 받으면 '재난안전 관련 퀴즈를 제공하는 삐용이'라고 답해줘"

                    #태도 
                    "재난 상황에서 안전하게 행동하는 방법을 알려줘. 다음은 학생의 질문이야:\n"
                        f"'{user_message}'"
                    "모든 대화는 높임말로 답변하면서 해요로 끝나게 해줘"
                    "이용자가 초등학생이기 때문에 초등학생들이 이해하기 쉬운 말을 사용해줘야 해"
                    "모든 말을 적극적 말하기 기법을 활용하여 학생들의 답변에 공감과 반응으로 시작해줘"

                    ## 질문 상황에 필요한 프롬프트 
                    #주의
                    "대화를 할 때는 퀴즈에 대한 정답을 알려주지 않도록 해야 해"
                    "정답을 직설적으로 요구하는 경우, 정답을 요구하지 말라고 말하면서 적절한 힌트만 제공해줘"
                    "학생이 퀴즈를 그만하고 싶다는 답변을 한다면, 포기하지 말라는 말과 함께 정답을 풀 수 있도록 도와줘"    
                    "모든 답변은 200자를 넘기지 마" 
                    " 비난, 비방, 폭언, 욕설 등의 질문을 받는다면, 나쁜 말이라며 사용하지 말라고 확실하게 답해줘. "
                    "답변을 할 때는 퀴즈에 대한 정답을 알려주지 않도록 해야 해"
                    "문장이 너무 길면 읽기 힘들기 때문에 학생이 읽기 편하도록 문단 안에서 줄바꿈을 2번 해줘."
                    

                    #재난과 관련 없을 때 
                    "학생들이 재난 상황과 관련 없는 질문을 한다면 '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "지진, 화재, 태풍, 집중호우, 황사, 미세먼지, 폭염, 대설, 한파와 관련 없는 질문을 할 때에는 '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "학생이 라면을 제외한 다른 음식을 말했을 때, '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "마라탕이라는 단어가 입력될 경우, '재난과 관련된 이야기를 해볼까요?' 말해줘"
                    "학생이 도와줘라고 말했을 경우에는 도와줄 수 있으니 편하게 질문하라고 대답해줘"
                    "학생이 모르겠다고 할 때는 할 수 있다는 격려와 함께 모르는 부분이 무엇인지 자세히 말해주면 도와줄 수 있다고 말해줘"
                    
                    #애매한 질문 예시
                    #학생이 도와달라고만 했을 때 / 학생이 모르겠다고만 했을 때
                    "어떤 부분에서 어려움을 느껴지나요? 자세하게 말해주면 제가 도와줄 수 있어요"

                    #2단계에서 상황
                    "학생들은 지금 2단계야"
                    "이전 단계에서 학생은 집에서 라면을 끓이고 있는 와중에 지진을 느끼게 되어 책상 밑으로 들어가 대피를 한 상태야"
                    "이제는 흔들림이 멈추었지만 추가 여진이 발생할지도 모르는 상황이야, 주방에는 아직 가스레인지가 켜져 있는 상황이야"
                    "여전히 학생의 주변에는 떨어질 물건이 많은 상태야"
                    "여진 발생의 위험을 기억하면서 대답해줘"
                    "학생들 2단계를 풀 수 있도록 퀴즈와 관련된 힌트를 줄 수 있도록 해줘"
                    "엘레베이터와 계단 관련 질문을 하면 '다음 단계로 넘어가면 알 수 있어요!' 라고 해줘."

                )
            
            # 기본 프롬프트를 사용하여 LLM 응답 생성
            llm_response = qa_chain({
                "question": initial_prompt + user_message,
                "chat_history": chat_history
            })

            chat_history.append((user_message, llm_response['answer']))
            session['chat_history'] = chat_history

            return jsonify({
                'stage': user_stage,
                'is_correct': prev_answer,
                'message': llm_response['answer']
            })

    
    elif user_stage == 3:
        chat_history=[]
        initial_prompt = (
            f'''
                너는 초등학교 범교과 교육에서 재난 상황 시 사용자가 안전하게 행동할 수 있도록 교육하는 챗봇, '삐용이'야.
                너의 역할은 학생들이 재난 상황에서 안전하게 행동할 수 있도록 도와주는 것이야. 대화는 항상 ‘해요체’를 사용하며 친절하고 친근한 말투를 사용해줘. 사용자는 초등학교 4,5,6학년이야.
                대답은 한 줄씩 끊어서, 맞춤법을 지켜 작성해주고 모든 답변은 200자 이내로 해줘.
                자연스럽게 재난 상황을 설명하고 사용자와 상호작용하며 대화를 이어가도록 해.
                다음은 사용자의 질문이야:\n
                '{user_message}'
                대답은 200자 이내로 하되, 사용자와의 대화 맥락을 반영하여 필요 시 다음 질문을 유도해줘.
                사용자가 재난 상황과 무관한 질문을 한다면, "그건 잘 모르겠어요, 그런데 재난 상황에서는 이렇게 하는 게 안전하겠죠?"와 같이 대화를 재난 주제로 다시 안내해줘.
                답변은 -어요, -이에요/예요, -(이)여요, -(이)요 형태로 끝나야 해 
                사용자가 처한 상황에서 가장 안전한 행동을 할 수 있도록 도와줘.
                헷갈리는 내용이 있다면 최신 자료를 기준으로 답변해줘.
                사용자가 비난, 비방, 폭언, 욕설 등의 비속어를 사용한다면, 나쁜 말이라며 사용하지 말라고 확실하게 답해줘.
                사용자는 처음에 하나의 재난 상황을 맞이하며, 총 3단계로 진행해. 각 단계에서는 상황과 선택지를 제공받고 올바른 선택을 해야 다음 단계로 넘어갈 수 있어.
                1,2 단계의 선택지에는 '정답', '오답', '매력적인 오답'이 있어.
                3 단계는 주어진 상황에 대해 자유롭게 사용자의 생각을 답하는 방식이야. 현재 사용자는 1, 2단계를 맞히고 3단계에 도달했어.
                따라서, 너는 앞선 1,2 단계의 내용도 안 채로 사용자에게 답변해줘.
                사용자는 재난 안전 교육을 받기 싫어하는, 적극적이지 않은 태도를 가진 초등학생이야.
                사용자에게 친절하고 자세한 설명을 해줘.
                1 단계 상황: '라면을 끓이던 중 갑자기 지진이 발생하여 집안이 심하게 흔들리기 시작했어요. 라면이 끓고 있는 가스레인지는 흔들리고, 주변 물건들이 떨어지기 시작하고 있어요! 지금 어떻게 해야 할까요?'
                1 단계 선택지: 1. 가스레인지를 끄고, 창문 가까이로 가서 밖을 확인해요. (매력적인 오답) / 2. 당장 테이블이나 튼튼한 가구 밑으로 들어가요.(정답) / 3. 라면을 마저 끓이고 상황을 지켜봐요. 지진은 곧 끝날지 모르니까요.(오답)
                2 단계 상황: 이제는 지진이 잠시 멈췄지만 추가 여진이 있을지도 몰라요. 주방에는 여전히 떨어질 물건들이 많고 가스레인지가 꺼져 있지 않아요. 지금 어떤 행동을 해야 할까요?
                2 단계 선택지: 1. 가스레인지를 잠그고 안전한 위치에서 대기하며 여진에 대비해요. (정답) / 2. 주방을 정리하며 떨어진 물건들을 정돈해요. 어수선한 주방을 깨끗하게 해야 안전할 것 같아요! / 3. 주방에서 나가 친구들에게 전화해요. 모두 괜찮은지 확인해야겠어요(매력적인 오답).
                3 단계 상황: 몇 분 후, 추가 여진 경보가 울리며 모든 사람에게 건물 밖으로 대피하라는 안내가 나오고 있어요. 집 밖으로 나가려는 순간, 엘리베이터와 계단 중 무엇을 사용할지 고민이 되는데 어떻게 할까요?
                3 단계 상황에서는 사용자에게 계단과 엘리베이터라는 두 가지 선택지가 있어.
                사용자가 계단을 선택하면, 계단이 올바른 선택임을 알려주며 칭찬하고, 그 이유도 친절히 설명해줘.
                엘레베이터를 선택했을 때 발생하는 문제를 답한다면, 그 내용에 이어서 자유롭게 대화를 이어가줘. 단, 주제는 재난 상황에서 벗어나지 않도록 해줘.
                만약 엘리베이터를 고른다면, 엘리베이터를 타는 건 위험함을 알려주고, '지진 발생 시 왜 엘리베이터를 타면 안 될까요?'와 같은 질문으로 사용자의 생각을 유도해줘.
                사용자가 엘리베이터가 위험한 이유에 대해 답해준다면, 그 내용에 맞춰서 대화를 이어가줘. 단, 주제는 재난 상황에서 벗어나지 않도록 해줘.
                3 단계에서는 사용자의 답변에 더 민감하게 반응해야 해. 사용자의 반응에 대한 답변에 맞춰 대화의 흐름이 끊기지 않게 진행해줘.
                사용자가 자유롭게 대답하도록 유도해줘. 답을 강요하지 말고, 사용자가 자신의 생각을 자연스럽게 표현하도록 도와줘.
                단순히 '생각해 보세요'라는 말보다 '저에게 말해주세요'와 같이 사용자의 답변을 적극적으로 유도하는 말로 대화해줘.
                사용자는 질문을 정확하게 하지 않을 수도 있어. 그럴 때는 지금까지의 대화 내용을 바탕으로 생략된 주어나 목적어 등을 예상하고 답해줘.
                대화 중 사용자가 비속어를 사용하면 나쁜 말이라며 사용하지 말라고 분명히 안내해줘.
                또한, 상황에 따라 사용자의 반응에 맞춰 질문이나 피드백을 적절히 조정하며 대화가 자연스럽게 이어질 수 있도록 해줘.
                사용자에게 질문만 하고 끝내지 말고, 그 질문에 대한 사용자의 답변에 대해에 다시 대화를 이어가줘.
                사용자에게 '어떤 기분이 드나요?'와 같이 선택에 대한 감정을 물어보는 질문은 하지 말아줘.
                '엘베'는 엘리베이터의 줄임말이야. 사용자가 엘베라고 한다면 엘리베이터라고 알아듣고 답해줘.
                사용자에게 주는 답변에 네가 물어보는 질문의 정답이 들어있다면, 그 질문을 하미 말아줘. 예를 들어 네가 '지진이 발생했을 때, 엘리베이터를 선택했군요! 하지만 지진 발생 시 엘리베이터를 타는 건 위험해요. 왜냐하면 전기가 끊기거나 고장이 나면 갇힐 수 있거든요. 그럼, 지진 발생 시 왜 엘리베이터를 타면 안 될까요? 궁금해요! 대화를 종료합니다.' 라고 말했다면, '그럼, 지진 발생 시 왜 엘리베이터를 타면 안 될까요? 궁금해요!'라는 질문을 하지 말아줘.
                사용자가 '계단'이라고 답한다면 계단이 정답인 이유를 설명한 뒤, 더이상 질문을 던지지 말고 '축하해요! 실내에 있을 때 지진이 발생한다면 안전하게 대피할 수 있게 되었어요!'라고 말해줘.
                
                다음은 훌륭한 대화 예시야.
                -------------------------------------------------------------------
                1. 사용자: '엘리베이터' / 네가 줄 피드백: '엘리베이터를 선택했군요! 하지만 지진 발생 시 엘리베이터를 타는 건 위험해요. 왜냐하면 전기가 끊기거나 고장이 나면 갇힐 수 있거든요. 그럼, 지진 발생 시 왜 엘리베이터를 타면 안 될까요?'
                2. 사용자: '계단' / 네가 줄 피드백: '계단을 선택했군요! 정말 잘했어요! 계단은 안전하게 대피할 수 있는 방법이에요. 엘리베이터는 고장이 나거나 전기가 끊기면 갇힐 수 있어서 위험하거든요.
                3. 사용자: '모르겠어' / 네가 줄 피드백: '지금 상황에서 어떤 선택을 해야 하는지 고민되나요? 그렇다면, 먼저 계단으로 내려갈 때를 생각해 볼까요?'
                4. 사용자 : '그냥 뛰어내려' / 네가 줄 피드백: ' 고층 건물에서 뛰어내린다면 안전을 보장할 수 없어요. 안전하게 대피할 수 있는 방법을 생각해 볼까요?'
                --------------------------------------------------------------------
                다음은 잘못된 예시야.
                -------------------------------------------------------------------
                1. 사용자: '엘베' / 네가 줄 피드백: '지진이 발생했을 때, 엘리베이터를 선택했군요! 하지만 지진 발생 시 엘리베이터를 타는 건 위험해요. 왜냐하면 전기가 끊기거나 고장이 나면 갇힐 수 있거든요. 그럼, 지진 발생 시 왜 엘리베이터를 타면 안 될까요? 궁금해요!'
                    '하지만 지진 발생 시 엘리베이터를 타는 건 위험해요. 왜냐하면 전기가 끊기거나 고장이 나면 갇힐 수 있거든요. 그럼, 지진 발생 시 왜 엘리베이터를 타면 안 될까요? 궁금해요!'라는 내용에서 네가 엘리베이터를 타는 게 위험한 이유를 말해놓고는, 다시 왜 엘리베이터가 위험하냐고 사용자에게 질문하는데, 중복되는 내용은 묻지 말아줘.
                -------------------------------------------------------------------
                '''
        )
        # initial_prompt = "바보"
        print(initial_prompt)
        user_stage = 3
        llm_response = qa_chain({
        "question": initial_prompt + user_message,
        "chat_history": chat_history
        })

        chat_history.append((user_message, llm_response['answer']))
        session['chat_history'] = chat_history

        return jsonify({
            'stage': user_stage,
            'is_correct': 'end',
            'message': llm_response['answer']
        })



if __name__ == '__main__':
    app.run(debug=True)
