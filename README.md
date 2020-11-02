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
argparse, warnings, io, platform, os<br/>
* Python3 외장 패키지
    - 신경망 구축 및 학습/예측 수행 - pytorch, torchvision
    - 데이터 조작 - pandas, numpy, sklearn <br/>
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
    1. 설명 <br/>
      주어진 데이터를 활용하여 모델을 학습하는 python 파일 <br/>
    2. 실행 <br/>
      python train.py --is_generated [True | False] --n_folds int --output [valid | test] --epochs int --batch_size int --get_pretrained [True | False] --use_pretrained [True | False] --model_file str
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
        * --get_pretrained [True | False]
        : /output/models 아래의 이전에 학습한 state_dict load하여 학습 없이 return 하는지 여부
        * --use_pretrained [True | False]
        : /output/models 아래의 이전에 학습한 state_dict를 사용하여 학습하는지 여부
        * --model_file str
        : /output/models 아래에 학습한 모델을 저장할 파일의 이름
        
        4. 수행시간
            
<h3>결과 파일</h3>

**/output/results 아래, 이름을 지정한 형태로 결과값이 반환됨.** <br/>
e.g) results.csv

<h3>예측 결과</h3>

<h2 id="review"> :checkered_flag: 대회후기</h2>

EMNIST 대회에 참가하게 된 계기는, 신경망을 설계하고 예측하는게 경험이 부족했기 때문이다. 그래서 대회를 통한 나의 목표는 높은 점수보다는 pytorch의 사용과 신경망에 대한 폭넓은 이해를 하는 것이다.

이를 위해, 이유한님의 Bangali-AI Competition Baseline 동영상을 통해 pytorch를 활용한 이미지 분류를 먼저 체험했다. 이후, Dacon의 pytorch baseline code를 통해 pytorch에서의 파이프라인 흐름에 대한 이해를 쉽게 할 수 있었다.

대회참가기간 1주 동안 위의 두 가지 참고자료를 통해 먼저 설정했던 목표를 달성했다. 그리고, 대회기간 이후, 이미지 분류를 위한 CNN뿐 아니라 전반적인 신경망에 대한 설명과 구현을 다룬 『3분 딥러닝 파이토치맛』으로 왜 CNN을 위와 같은 형태로 구성하였으며, 우승자 Code에 사용된 ResNet을 구현해볼 수 있었다.

앞으로, 'Private: 4위, Public: 0.95098, ResNet + SSL' 코드를 살펴보며 높은 수준의 예측을 위해 어떤 절차와 방법이 필요한지 살펴보려고 한다.