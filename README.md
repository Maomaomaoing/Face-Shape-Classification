# Face-Shape-Classification
希望透過臉型辨識，找出受試者之臉型，進而給出合適自己臉型的鏡框類型之建議，並將鏡框合成至臉上。  
<img src="https://github.com/Maomaomaoing/Face-Shape-Classification/blob/master/glasses.jpg" width="300" height="300">

## 爬取圖片
* 從google的圖片搜尋，爬取各種臉型的圖片
* 使用[google-images-download](https://github.com/hardikvasa/google-images-download)
下載圖片
* 用round face, oval face, square face等等作為搜尋的關鍵字

## 清理資料
* 清除不需要的臉型圖片(包含混雜臉型、臉
型不清等)
* 將組圖切割成單一圖片、刪除各類別內及跨類別之重複的圖片

## 前處理
利用AAM+Face segmentation擷取臉部  
<img src="https://github.com/Maomaomaoing/Face-Shape-Classification/blob/master/face_shape.jpg" width="250" height="150">

## 模型
嘗試VGG、dilated convolution、SPP net、Inception、coordinate convolution等模型進行分類  
最後coordinate convolution分類結果最佳
