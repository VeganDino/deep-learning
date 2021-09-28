## Convolution Neural Network Model

[What is CNN](/What is CNN.md)에서 알아본 CNN에서 주로 사용하는 레이어를 이용해서
간단한 컨볼루션 신경망 모델을 만들어보자.   
먼저 손으로 삼각형, 사각형, 원을 손으로 그린 이미지가 있고 이미지 크기가 8x8이라고
가정해보자.    
삼각형, 사각형, 원을 구분하는 3개의 클래스를 분류하는 문제이기 때문에 출력 벡터는
3개여야 한다. 필요하다고 생각하는 레이어를 구성한다.  

<img src='https://user-images.githubusercontent.com/56749776/135010431-f72b1a74-e387-4452-8d44-848366ced1ca.png' width='80%'>

<br>

- Convolution layer : 입력 이미지 크기 8x8, 입력 이미지 채널 1개, 필터 크기 3x3, 필터 수 2개, 경계 타입 ‘same’, 활성화 함수 ‘relu’

<img src='https://user-images.githubusercontent.com/56749776/135010727-15bfdba4-7144-4171-87fe-516c5b3622da.png' width='80%'>

- Max Pooling layer : 풀 크기 2x2

<img src='https://user-images.githubusercontent.com/56749776/135010922-24e3aa26-c6d3-43b0-8e9a-38f709a56a17.png' width='80%'>

- Convolution layer : 입력 이미지 크기 4x4, 입력 이미지 채널 2개, 필터 크기 2x2, 필터 수 3개, 경계 타입 ‘same’, 활성화 함수 ‘relu’

<img src='https://user-images.githubusercontent.com/56749776/135011010-dcae3d2b-ac5e-4a0b-a4be-a3f99c25df57.png' width='80%'>

- Max Pooling layer : 풀 크기 2x2

<img src='https://user-images.githubusercontent.com/56749776/135011150-c9c2b41c-0e29-46fe-a575-81c49b45aa1d.png' width='80%'>

- Flatten layer

<img src= 'https://user-images.githubusercontent.com/56749776/135011590-795fbaf4-e816-406a-bbaf-3fbb1bfb829b.png' width='80%'>

- Dense layer : 입력 뉴런 수 12개, 출력 뉴런 수 8개, 활성화 함수 ‘relu’

<img src='https://user-images.githubusercontent.com/56749776/135011671-2abf1559-645e-44e0-bc59-17c60799251c.png' width='80%'>

- Dense layer : 입력 뉴런 수 8개, 출력 뉴런 수 3개, 활성화 함수 ‘softmax’

<img src='https://user-images.githubusercontent.com/56749776/135011751-6e2e1409-ec07-4f5e-807e-d1b2451821ba.png' width='80%'>

위의 모든 레이어를 조합하면 다음과 같다. 입출력 크기만 맞으면 레고 끼우듯이 합치면 된다. 
참고로 케라스 코드에서는 가장 첫번째 레이어를 제회하고는 입력 형태를 자동으로 계싼하므로 
이 부분은 신경쓰지 않아도 된다. 이렇게 레이어를 조립하면 간단한 컨볼루션 모델이 생성된다. 
이 모델에 이미지를 입력하면 삼각형, 사각형, 원을 나타내는 벡터가 출력된다. 

<img src='https://user-images.githubusercontent.com/56749776/135012833-8577979c-5995-4f4c-9da2-fcea561940bf.png' width='80%'>

그럼 케라스 코드로 어떻게 구현하는지 보자. 먼저 필요한 패키지를 추가한다. 

```python
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
```

Sequential 모델을 하나 생성한 뒤 위에서 정의한 레이어를 차례차례 
추가하면 컨볼루션 모델이 생성된다. 

```python
model = Sequential()

model.add(Conv2D(2, (3, 3), padding='same', activation='relu', input_shape=(8, 8, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(3, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
```

생성한 모델을 케라스에서 제공하는 함수를 이용하여 가시화 시킨다. 

```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```
