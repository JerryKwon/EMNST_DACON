# EMNIST DACON

<h2 id="context"> :pushpin: 대회 개요 </h2>
https://dacon.io/competitions/official/235626/overview/

기존의 MNIST Dataset인 숫자 위에 알파벳이 겹쳐진 EMNIST Dataset에서 각 데이터에 겹쳐진 숫자(Label)을 예측하라

## :clipboard: 목차
<ol>
<li><a href="#context">대회개요</a></li>
<li><a href="#schedule">진행일정</a></li>
<li><a href="#reference">참고자료</a></li>
<li><a href="#repo-composit">Repo 구성</a></li>
<li><a href="#execution">실행</a></li>
<li><a href="#review">대회후기</a></li>
</ol>


<h2 id="schedule"> :calendar: 진행일정</h2>
2020년 8월 03일(월) ~ 2020년 9월 14(일) [43일]

※ 실제 참가 시작일: 2020년 9월 5일(토)

<h2 id="reference"> :books: 참고자료 </h2>

* Emnist를 이미지로 저장하여 훈련부터 테스트까지(pytorch) <br/>
   https://dacon.io/competitions/official/235626/codeshare/1592?page=2&dtype=recent&ptype=pub
* 3분 딥러닝 파이토치 맛 <br/>
   https://github.com/keon/3-min-pytorch <br />
   https://github.com/JerryKwon/3-min-pytorch-review
* Private: 4위, Public: 0.95098, ResNet + SSL <br/>
   https://dacon.io/competitions/official/235626/codeshare/1677?page=1&dtype=recent&ptype=pub
* Kaggle BangaliAI Competetion Baseline - Youhan Lee Youtube <br/>
   https://www.youtube.com/channel/UC--LgKcZVgffjsxudoXg5pQ
   
<h2 id="repo-composit"> :open_file_folder: Repo 구성 </h2>

* ./idea, \__pycache\__, venv: pycharm 프로젝트 구성 파일 디렉터리
* /input/data: 학습 및 예측을 위한 데이터 적재 디렉터리 (w/o Mel-Spectogram) <br/>
   **- .json 파일들(train,test,val,song_meta,genre_gn_all)의 용량 문제로 인해 학습 및 예측 수행 전 데이터 적재 필요!**
* /output/models: 
* /output/results: 학습을 통해 예측을 수행한 결과 파일(.json)이 저장되는 디렉터리 <br/>
   **- 저장 포맷: '[valid | test]_[hybrid | icbf]_rcomm_result.json'**
* inference.py: 학습 및 예측을 실행을 위한 python 파일 <br/>
   **- 사용법은 <a href="#how-to-execute">실행-실행법</a> 참고**
* train.py: 학습 실행을 위한 python 파일 <br/>
   **- 사용법은 <a href="#how-to-execute">실행-실행법</a> 참고**
* data_loader.py: 학습 데이터 로드를 위한 python 파일 (inference.py or train.py에서 실행 시 내부적으로 사용)
* models.py: 학습에 사용될 model들을 저장하는 python 파일 <br/>
   - 클래스 및 예측 모델 구성 
       + CustomCNN - EMNIST 예측 수행을 위한 CNN 모델

<h2 id="execution"> :exclamation: 실행 </h2>

<h3><b>※주의</b></h3>
해당 프로젝트는 Windows, Linux 환경 모두에서 실행될 수 있도록 만들어졌으나, 시간 상의 이유로 Windows만 구동 테스트를 완료하였습니다.
<b>따라서, Windows 환경에서 실행해야 합니다.</b>

<h3>구현 알고리즘</h3>

1. CustomCNN

<h3>사용 패키지</h3>

* Python3 내장 패키지<br/>
argparse, warnings, json, io, platform, os, collections<br/>
* Python3 외장 패키지
    - 신경망 구축 및 학습/예측 수행 - pytorch, torchvision
    - 데이터 조작 - pandas, numpy, sklearn, scipy <br/>
    - 진행상황 모니터링 - tqdm <br/>

<h3 id="how-to-execute">실행법</h3>

* inference.py <br/>
    1. 설명 <br/>
    주어진 데이터를 활용하여 학습/예측을 수행하는 python 파일
    2. 실행 <br/>
    python inference.py --is_generated [True | False] --n_folds int --output [valid | test] --epochs int --batch_size int --use_pretrained [True | False] --predict_file str
    3. 옵션
        * --is_generated [True | False] <br/>
        : /input/data 아래의 train / test img를 1장 당 별도로 분리한 데이터를 Local에 생성하였는지 여부 
        * --n_folds int 
        : Train 데이터을 몇 개의 데이터로 나눌 것인지 정의
        * --output [valid | test] 
        : 예측을 수행할 데이터셋을 결정
            * valid: Train에서 지정될 임의의 fold인 Valid 데이터에 대해 예측 진행
            * test: Test 데이터에 대해 예측 진행 
        * --epochs int
        : CNN 모델에 대해 학습 수행시 EPOCH의 수
        * --batch_size int    
        : CNN 모델에 대해 학습 수행시 BATCH의 수 
        * --use_pretrained [True | False]
        : /output/models 아래의 이전에 학습한 state_dict를 사용하는지 여부
        * --predict_file str
        : /output/results 아래에 예측 결과를 나타낼 csv 파일의 이름
    
* train.py <br/>
  **※ 주의** <br/>
    **해당 프로젝트에서는 pytorch, tensorflow, keras등의 학습 모델 생성을 통한 예측을 진행하지 않기 때문에, inference.py 와 동일함.**
  
    1. 설명 <br/>
      주어진 데이터를 활용하여 모델을 학습하는 python 파일 <br/>
    
    2. 실행 <br/>
      python train.py --model_type hybrid --is_valid True
    3. 옵션
        * --model_type - [icbf | hybrid] <br/>
        : 예측을 수행할 모델의 타입을 결정하는 파라미터. <br/>
            * icbf: 아이템 기반 협업 필터링 방식을 통한 Recommendation 진행
            * hybrid: 아이템 기반 협업 필터링 + 컨텐츠기반 필터링 + 예외처리 방식을 통한 Recommendation 진행
        * --is_valid - boolean [True | False] <br/>
        : 예측을 수행할 데이터 타입을 결정하는 파라미터 <br/>
            * True: /data/val.json을 대상으로 하여 Recommendation 진행
            * False: /data/test.json을 대상으로 하여 Recommendation 진행
    4. 수행시간
        * --model_type = icbf +  --is_valid = True <br/>
            : 약 40분 ~ 1시간
        * --model_type = icbf +  --is_valid = False <br/>
            : 약 30분 ~ 40분
        * --model_type = hybrid +  --is_valid = True <br/>
            : 약 40분 ~ 1시간
        * --model_type = hybrid +  --is_valid = False <br/>
            : 약 30분 ~ 40분
            
<h3>결과 파일</h3>

**/result 아래, '[valid | test]_[hybrid | icbf]_rcomm_result.json' 형태로 결과값이 반환됨.** <br/>
e.g) test_hybrid_rcomm_result.json

<h3>예측 결과</h3>

* icbf <br/>
  valid LB - song - 0.159576 / tag - 0.340179 = 0.18666645 [61st in leaderboard]

* **[제출] hybrid** <br/>
  valid LB - song - 0.160008 / tag - 0.411810 = 0.197778 [55th in leaderboard]
            
<h2 id="review"> :checkered_flag: 대회후기</h2>

**추후 작성 예정**