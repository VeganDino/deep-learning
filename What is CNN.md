### **CNN (Convolution Neural Network)**

컨볼루션 신경망 모델에서 주로 사용되는 컨볼루션(Convolution) 레이어, 
맥스풀링(Max Pooling) 레이어, 플래튼(Flatten) 레이어에 대해서 알아보고자 한다.    
각 레이어별로 레이어 구성 및 역할에 대해 보자.

<br>

### **Convolution Layer (컨볼루션 레이어)**

필터로 특징을 뽑아주는 특징이 있다. 
여러가지 컨볼루션 레이어 종유 중 영상 처리에 주로 사용되는
Conv2D 레이어를 보도록 한다. 다음은 Conv2D 클래스 사용 예제이다.

<br>

```python
tf.keras.layers.Conv2D(32, (3,3), padding='vaild', activation='relu', input_shape=(28, 28, 1))
```

<br>

< 주요 인자 >
- **첫번째 인자** : 컨볼루션 필터의 수 ex) 32
- **두번째 인자** : 컨볼루션 커널의 (행, 열) ex) (3,3)
- **padding** : 경계 처리 방법 정의
    - 'valid’ : 유효한 영역만 출력. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작다.
    - 'same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일
- **input_shape** : 샘플 수를 제외한 입력 형태를 정의. 모델에서 첫 레이어일 때만 정의하면 된다.
    - (행, 열, 채널 수)로 정의. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정
- **activation** : 활성화 함수 설정.
    - 'linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옴.
    - 'relu’ : rectifier 함수, 은익층에 주로 쓰임.
    - 'sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰임.
    - 'softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰임.
    
<br>

< 입력 형태 >
- image_data_format이 ‘channels_first’인 경우 (샘플 수, 채널 수, 행, 열)로 이루어진 4D 텐서
- image_data_format이 ‘channels_last’인 경우 (샘플 수, 행, 열, 채널 수)로 이루어진 4D 텐서
- image_data_format 옵션은 “keras.json” 파일 안에 있는 설정이다. 
콘솔에서 “vi ~/.keras/keras.json”으로 keras.json 파일 내용을 변경할 수 있다.

<br>

< 출력 형태 >
- image_data_format이 ‘channels_first’인 경우 (샘플 수, 필터 수, 행, 열)로 이루어진 4D 텐서
- image_data_format이 ‘channels_last’인 경우 (샘플 수, 행, 열, 필터 수)로 이루어진 4D 텐서
- 행과 열의 크기는 padding가 ‘same’인 경우에는 입력 형태의 행과 열의 크기가 동일함

<br>

< 예제 >
```python
Conv2D(1, (2, 2), padding='valid', input_shape=(3, 3, 1))
```
입력 이미지는 채널 수 1, 너비 3픽셀, 높이 3픽셀, 크기 2x2인 필터가 하나인 경우의 레이어이다.
image_data_format이 ‘channels_last’인 경우이다. 이를 도식화하면 다음과 같다.

<img src ='https://user-images.githubusercontent.com/56749776/134941328-524369ca-aa31-41d0-8a58-879cc580ce1c.png' width="85%">

필터는 가중치를 의미하며 하나의 필터가 입력 이미지를 순회하면서 적용돈 결과값을 모으면 출력 이미지가 생성된다. 
여기에 두 가지 특성이 있다. 

1) 하나의 필터로 입력 이미지를 순회하기 때문에 순회할 때 적용되는 가중치는 모두 동일하다. 이를 파라미터 공유라고 부른다. 이는 학습해야할 가중치 수를 현저하게 줄여준다. 
2) 출력에 영향을 미치는 영역이 지역적으로 제한되어 있다. 그림으로 볼 때 y0에 영향을 미치는 입력은 x0, x1, x3, x4로 한정되어 있다. 예로 코를 볼 때 코 주변만 보고, 눈을 볼 때는 눈 주변만 보면서 학습 및 인식한다.

<br>

### **가중치의 수**

Dense 레이어와 컨볼루션 레이어와 비교하면서 차이점을 알아보자.   
영상도 결국에는 픽셀의 집합이므로 입력 뉴력이 9개 (3x3)이고, 출력 뉴런이 4개(2x2)인 Dense 레이어로 표현할 수 있다. 

```python
Dense(4, input_dim=9)
```

이를 도식화하면 다음과 같다. 

<img src ='https://user-images.githubusercontent.com/56749776/134943666-84f1e6d0-773b-4fe7-9a5f-7547bc152e57.png' width="80%">

가중치(시냅스 강도)는 녹색 블럭으로 표시되어 있다.    
컨볼루션 레이어에서는 가중치 4개로 구성된 크기가 2x2인 필터를 적용하면 뉴런 상세 구조는 다음과 같다. 

<img src='https://user-images.githubusercontent.com/56749776/134943706-2d835dc8-1209-4778-87eb-0298dbb7a1b3.png' width='80%'>

필터가 지역적으로만 적용되어 출력 뉴런에 영향을 미치는 입력 뉴런이 
제한적이므로 Dense 레이어와 비교했을 때, 가중치가 많이 줄어든 것을 
볼 수 있다. 게다가 녹색 블럭 상단에 표시된 빨간색, 파란색, 분홍색, 
노란색끼리는 모두 동일한 가중치(파라미터 공유)이므로 결국 사용되는 
가중치는 4개다. 즉 Dense 레이어에서는 36개의 가중치가 사용되었지만, 
컨볼루션 레이어에서는 필터의 크기인 4개의 가중치만을 사용한다.

<br>

### **경계 처리 방법**

컨볼루션 레이어 설정 옵션에는 ```border_mode```가 있는데, 'vaild'와 'same'으로 설정 할 수 있다.

<img src='https://user-images.githubusercontent.com/56749776/134945663-d0b5d07e-1613-4003-9660-a35b589acc84.png' width='80%'>

‘valid’인 경우에는 입력 이미지 영역에 맞게 필터를 적용하기 때문에 
출력 이미지 크기가 입력 이미지 크기보다 작아진다. 반면에 ‘same’은 
출력 이미지와 입력 이미지 사이즈가 동일하도록 입력 이미지 경계에 
빈 영역을 추가하여 필터를 적용한다. ‘same’으로 설정 시, 입력 이미지에 
경계를 학습시키는 효과가 있다.

<br>

### **필터 수**

입력 이미지가 단채널의 3x3이고, 2x2인 필터가 하나 있다면 다음과 
같이 컨볼루션 레이어를 정의할 수 있다. 

```python
Conv2D(1, (2, 2), padding='same', input_shape=(3, 3, 1))
```

<img src='https://user-images.githubusercontent.com/56749776/134946120-a0eb8543-3bcb-4e9c-8464-81f8df5fe159.png' width='80%'>

만약 여기서 사이즈가  2x2 필터를 3개 사용한다면 다음과 같이 정의할 수 있다.

```python
Conv2D(3, (2, 2), padding='same', input_shape=(3, 3, 1))
```

<img src='https://user-images.githubusercontent.com/56749776/134946530-2f500ad0-e58c-465c-a8ad-8d8ccd148057.png' width='80%'>

필터가 3개라서 출력 이미지도 필터 수에 따라 3개로 늘어났다.
총 가중치의 수는 3 x 2 x 2으로 12개이다. 
필터마다 고유한 특징을 뽑아 고유한 출력 이미지로 만들기 때문에 
필터의 출력값을 더해서 하나의 이미지로 만들거나 그렇게 하지 않는다.
간단히 카메라 필터라고 생각하면 된다.
스마트폰 카메라로 사진을 찍을 떄 필터를 적용해볼 수 있는데, 
적용되는 필터 수에 따라 다른 사진이 나옴을 알 수 있다. 

<img src='https://user-images.githubusercontent.com/56749776/134947096-7ffec3fb-99be-49b2-86e3-f16129e4c6e0.png' width='80%'>

뒤에서 각 레이어를 레고처럼 쌓아올리기 위해서 약식으로 표현하면 다음과 같다. 

<img src='https://user-images.githubusercontent.com/56749776/134947264-4bbaa563-d2aa-4f50-a5cb-099ebbbb3552.png' width='80%'>

- 입력 이미지 사이즈가 3x3이다. 
- 2x2 커널을 가진 필터가 3개고 가중치는 총 12개이다.
- 출력 이미지 사이즈가 3x3이고 총 3개다. 이는 채널이 3개다라고도 표현한다.