## CNN (Convolution Neural Network)

컨볼루션 신경망 모델에서 주로 사용되는 컨볼루션(Convolution) 레이어, 
맥스풀링(Max Pooling) 레이어, 플래튼(Flatten) 레이어에 대해서 알아보고자 한다. 
각 레이어별로 레이어 구성 및 역할에 대해 보자.

## Convolution Layer (컨볼루션 레이어)

필터로 특징을 뽑아주는 특징이 있다. 
여러가지 컨볼루션 레이어 종유 중 영상 처리에 주로 사용되는
Conv2D 레이어를 보도록 한다. 다음은 Conv2D 클래스 사용 예제이다.

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
