# 여러명이 등장하는 동영상에서 특정 인물을 찾아내고 추적하는 기술 

Tensorflow KR 페이스북에 포스트에 따르면 Clova Video AI 기술이 V LIVE에 적용되었다고 한다. 
여러명의 아이돌이 등장하는 안무 영상에서 특정 아이돌의 직캠 영상을 자동으로 만들 수 있게 하는 기술이다.
높은 정확도를 위해서 
 
  - Human Detection 
  - Pose Estimation 
  - Optical Flow
  - +) 의상과 외모가 비슷한 아이돌들을 구분하기 위해 독자적인 방법 

의 기술이 순차적으로 적용되었다고 한다. 


영상을 보면서 느낀점: 
- 여러 사람이 겹쳐 있을 때의 정확도가 뛰어나지는 않았다 (e.g. Rose 영상의 0:33 쯤 Rose 가 다른 멤버 뒤에 있으니까 제니를 인식했음) 
- Pose Estimation의 기술덕분에 빠르고 격하게 움직여도 (안무에서는 중요한) 손의 움직임 등이 짤리지 않았다! 
- 안무를 딸 때 상당히 많이 쓰일 수 있는 기술같다! 
- 빠르고 신속하게 움직이는 동작이 많은데도 비교적 부드럽게 카메라가 움직이는 것 같아 보여서 보기 편하다 
- 너무너무너무재미있겠다 공부하면서 더 발전시키고 싶다!!!!!! 

--- 


## Proposed Flow of Work 

libraries: Tensorflow/pytorch, opencv etc.... 

* 따로 따로 트레이닝을 시키고 적용...? 

Step 1: Human Detection 을 이용해서 한 프레임에서 관심있는 사람을 identify 한다 
 - 모든 인물에 대해서 각각 비디오를 생성할 때: 한 사람 한 사람에 대해서 다음 단계를 진행하면 됨 
 - 주어진 인물에 대해서 하고 싶을 때: 특정 인물을 찾아내는 단계가 추가된다 (FaceNet etc) 
 - User가 내가 원하는 인물은 여기에 있어! 라고 힌트를 줬을 때 (예를들어, 한 프레임에 대해서 관심있는 인물이 있는 박스를 만들어준다 -> Detection 해야하는 넓이를 줄여줌) -> 그 박스 안에 있는 사람에 대해서 다음 단계를 진행하면 됨 
  - 또는, 모든 인물을 일단 찾아낸 후, user에게 누구에 대해서 진행하고 싶은지 물어보기 
일단 Detect 해서 어느정도 Region of Interest를 찾아서 localize 하기 


Step 2: ROI 에서 Pose Estimation 적용시켜서 ROI 를 더욱 세밀화 시키기 
 - Pose Estimation 으로 얻으려는 것? 더욱 세밀화된 ROI, 인물이 가운데 올 수 있도록 조절하는 것 (가슴~배꼽까지가 가운데 오게 하기?) 
 - crop 하는 부분이 계속 너무 움직이면 보는 사람 눈이 너무 피곤하고 안정적이지 못함.
 - 어느정도 박스 크기의 Threshold 를 만들어서 그 안에서 인물이 움직이고 있다면 crop 되는 부분을 움직이지 않기 
 - 안무의 경우, 인물이 팔이나 다리를 크게크게 사용할 수 있으니 인물의 어깨 & 가슴 부분을 crop된 부분의 가운데에 오게 하기 

Step 3: ROI 에 대해서 현재 프레임과 바로 다음 프레임을 이용해서 optical flow 를 사용해서 인물의 위치를 알아내기/예측. 
 - 인물이 빠르게 움직이면 ROI 밖에 있을 수도 있으니까 crop 할 크기보다는 조금 더 크게 input 으로 넣기 
 - Optical flow 를 통해서 finalize region 


--- 


## 공부할 때 참고할 것, Relevent Papers

  - Human Detection: 구글에서 공개한 Object Detection 용 API - Microsoft COCO 로 트레이닝 시킨 여러가지 모델이 있다 
     - 시간 & 정확도의 tradeoff 가 있기 때문에... ssd + Inception v2 모델 이상의 mAP로 해야하지 않을까. 
     - 사람을 제외한 다른 물체를 filter 해서 Human Detection 으로 만드는 법? 
     - 시간/장비가 없으면 Transfer Learning 을 할 수밖에 없나? 
     - 특정 인물에 대해서 만드는거면, Face Recognition 을 사용해서 detect 를 한 후에 optical flow 를 사용해서 따라다니기? 
     - 영상에 있는 모든 사람을 찾아낸 후 -> 모든 사람에 대해서 직캠을 만드는 것? 
     - 한 프레임에 대해서 관심있는 인물이 어디에 있는지 localize 하도록 user에게 주문해도 좋을 듯 (그 부분에만 detection 을 실시해서, 그 박스에 있는 인물에 대한 직캠을 만들기) 
     - OR 영상에 있는 모든 사람을 찾아낸 후 -> User에게 제외할 사람을 고르게 한다 -> 그 외에 사람에 대해서 직캠영상을 만든다 
     - 결국 중요한건 Optical Flow? 


  - Pose Estimation 의 예시 
  <img src="https://github.com/Hyunjulie/Naver_Campus_Hackday_Study/blob/master/dance1.png" width="200" height="300" />
   - 포즈를 분석하면서 crop 할 화면의 크기를 더욱 세밀하게 조정 할 수 있음. 다만 여러 사람이랑 겹쳐있을 때 오류가 있음 
   - 참고할 논문들 & Githubs
    - Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields[https://arxiv.org/pdf/1611.08050.pdf]
    - PersonLab (pose estimation & Segmentation)[https://arxiv.org/pdf/1803.08225.pdf] 
    - PoseNet (https://github.com/tensorflow/tfjs-models/tree/master/posenet) - 브라우저에서도 할 수 있음 
    - https://github.com/ildoonet/tf-pose-estimation
  
  - Optical Flow: FlowNet 2.0, Spatial Pyramid Network (lighter, faster) 
  
  
  
---

## 의상과 외모가 비슷한 아이돌을 구분하기 위한 Suggestion?






--- 

** Other Thoughts: ** 

- Test할 때 블랙핑크의 불장난은 정말 좋은 선택인것 같다 -> 멤버들끼리 뭉치는 부분, 따로 추는 부분, 빨리 움직이는 부분, 꼼지락 거리는 부분, 그리고 무엇보다 팔을 크게크게 벌리는 부분이 많아서 상당히 어려운 예시같다... 
- 다음 테스트로 정말... 어려울 Exo 의 늑대와 미녀를 도전하면 재미있겠다 https://www.youtube.com/watch?v=jmyQqIELyfU 

초기 모델로는 Human/Face detection 을 통해서 특정 인물을 찾아내서 영상을 따는것 보다, 

User 에게 원하는 인물의 box 를 그리게 해서 -> 그 박스 안에서 human detection -> 그 박스 안에 있는 사람을 따라다니는 모델도 좋을 것 같다 
 - 그렇게 된다면 누군지 모르는 사람도 따라다니면서 영상을 만들 수 있는 기술로 좀 더 generalize할 수 있지 않을까? 
 - 물론 이 방법은 end-to-end 가 아니여서 disadvantage 가 있겠지만, 오히려 user에게 freedom 을 줄 수 있는 부분? 
