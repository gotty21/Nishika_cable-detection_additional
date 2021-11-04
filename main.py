import cv2
import streamlit as st
from PIL import Image

# detect.pyの249行目のprint文はエラーがでるためコメントアウト
from yolov5.detect import run

from my_utils.preprocess import Preprosessing


st.title('簡易ケーブルコネクタ種類判別')


"""
---------------------------------------------------------------------------------------------------------------------------
## 目次
- 始めに
- 使い方
- ケーブルコネクタ判別


---------------------------------------------------------------------------------------------------------------------------
## 始めに
本アプリケーションは作者がstreamlitを学習する目的で作成したものです。
モデルの精度はあまり期待しないでください。



---------------------------------------------------------------------------------------------------------------------------
## 使い方

ケーブルコネクタが映った画像をアップロードするとそのコネクタの種類を判別します。
対応している画像形式はjpgです。

判別できる種類は以下の15種類です。

<判別可能なコネクタの種類>
- DisplayPort
- Dock
- HDMI
- Lightning
- Lightning_T
- Mini_DisplayPort
- RJ_45
- USB_Micro_B
- USB_Micro_B_3.1
- USB_Micro_B_W
- USB_Mini
- USB_Type_A
- USB_Type_B
- USB_Type_C
- VGA


Lightning系の空洞がないコネクタ以外は断面方向からの画像の方が精度よく判別してくれます。

判別結果の横に表示される数値は、判別結果の確信度を表しています。
1に近いほど判別結果である確率が高いことを示しています。



---------------------------------------------------------------------------------------------------------------------------



"""

st.header('ケーブルコネクタ判別')

uploaded_file = st.file_uploader('画像をアップロードしてください：', type='jpg')



if uploaded_file is not None:
    img = Image.open(uploaded_file)
    #st.image(img, caption='Uploaded Image', use_column_width=True)
    #st.write('判別中....')

    p = Preprosessing()
    img = p(img)

    file_path = './img/tmp.jpg'
    cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    run(
        weights='./weights/best_weights.pt',
        source=file_path,
        imgsz=512,
        conf_thres=0.001,
        max_det=1,
        exist_ok=True
    )




    img = Image.open('yolov5/runs/detect/exp/tmp.jpg')
    st.image(img, caption='Uploaded Image', use_column_width=True)




    


