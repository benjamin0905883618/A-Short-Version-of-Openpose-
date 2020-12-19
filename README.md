# Openpose 用法與操作
## 安裝
1. 下載github桌面版，擷取github上的openpose。
2. 安裝Visual Studio。
3. 安裝Cuda,cuDNN(2跟3順序不可調換，因為有一些初始過程會在這個期間完成)。
4. 在openpose\3rdparty\windows找到4個.bat檔，各執行一次，會自行下載所需的文件。
5. 下載並安裝CMake
6. 跑一次CMake，選擇編譯器、作業系統、cpu，選擇功能後生成library
7. 在Visual Studio上用Release模式build Openpose.sln，即完成。

注意 : 使用時只能在CMake建立的資料夾內執行，離開資料夾後會找不到pyopenpose這個library。
## 原理說明
### 目的
在一張2D圖片或影片上偵測人的軀幹及姿態，而且如果是多人在同一張圖片的情況下必須解決多人重疊的問題。
進階目的為達成即時偵測。
### Part Affinity Fields for Part Association(PAF)
輸入一張圖片後，先對圖片中的肢幹進行偵測。
![](https://i.imgur.com/cscseFR.jpg)
以上面這張圖為範例，進行偵測。
![](https://i.imgur.com/hHZd0oT.jpg)
上面這張圖為對鼻子的位置進行預測，在圖上隨機找一點p點，如果這個p點剛好在鼻子的長寬投影範圍內，就給予一個v的值;反之則給予0的值。
利用上面方法尋找"鼻子"這個物件後，再對預測出來的點製作一張Confidence Maps，此為Part Affinity Fields for Part Association(PAF)。
將圖形的PAF算出來、設定好loss後即可開始訓練網路，透過將每個點的區域用PAF算出來後，將點跟點連線，這個問題為NP-hard的問題。
附上官網提供的模型架構。
![](https://i.imgur.com/kbYbtZ5.png)
## 使用openpose
目前就我個人的使用狀況，需要的是僅有預測出的肢幹的圖來進行動作的預測，但如果有其他的需求，需要依照所求來調整參數，已得到最佳效果。
### openpose的flag
1. **logging_level** : 提醒等級，範圍為0~255，如果為0會輸出所有的提示訊息，為255則不會輸出任何提示訊息。提示中也有分重要程度，1為最輕微的，4為最重要的提示。
2. **disable_multi_thread** :透過減少frame rate來降低延遲。為boolean值，True為開。
3. **camera** : 相機選擇，選擇範圍為0~9，default值為負數，自動偵測可以使用的鏡頭作為輸入。
4. **camera_resolution** : 選擇相機解析度，如:"1920x1080"。default值為"-1x-1"，意即使用1280x720。
5. **video**、**image_dir** : 預設圖片及影片路徑，皆為\examples\media。
6. **keypoint_scale** : 最終偵測出的x,y的座標縮放，將使用**write_json**或是**write_keypoint**保存座標。選擇0將其縮放到原始source的解析度，1將只有讀取出的解析度，2為最終輸出此吋，2會將其縮放到(0,1)的範圍內。
7. **number_people_max** : 選擇最大會讀取的人數，default值為-1，意即所有在畫面中的人都讀取。
8. **maximum_positive** : 降低辨識出人的門檻，會大大提高錯誤率。
9. **fps_max** : 最大處理幀數速率，default為-1，如果openpose顯示圖像的數率過快可以調慢速度，讓使用者可以更好的分析每一幀。
10. **body** : 選擇0會停止身體keypoint檢測，default值為1，偵測keypoint。如果選擇2，會禁用內部的神經網路，但還是會進行greedy association parsing algorithm。
11. **model_pose** : 模組所使用的偵測模型。default為'BODY_25'，可選擇'COCO'(18個節點)、'MPI'(15個節點)、'MPI_4_layers'(較前面快，但準確率降低)
12. **net_resolution** : 輸入解析度，如果提升解析度可以提升精確度，降低解析度可增加辨識速度。
13. **scale_number** : 要平均的小數位數。
14. **scale_gap** : 刻度跟刻度間的間隔，預設為1。
15. **heatmaps_add_(身體部位)** : 如果為true，會使用身體部位的熱圖來填充op::Datum ::poseHeatMaps數組(上面看到的例圖)。可選擇部位，如果超過一個部位被加入，會把機器學習的模型改成body parts + background + PAFs，且由此，POSE_BODY_PART_MAPPING這份程式碼的速度會明顯變慢。
16. **heatmaps_add_bkg** : 跟前面函式類似，但是會直接將熱圖加到背景圖。
17. **heatmaps_add_PAFs** : 跟add_heatmaps_parts類似，但會把熱圖加到PAFs。
18. **part_candidates** : 還要啟用**write_json**才能儲存到變更。如果為true，
19. **face_detector** : 偵測面部，預設為0,為使用openpose偵測，1會選擇使用opencv，2會使用使用者所提供的函式，3會連手的偵測一起增強(會增加手部的點)。
20. **disable_blending** : 如果為true，原先輸出的背景會變成黑背景，而不是原圖。
21. **part_to_show** : 預設為0，會看見所有body parts(沒有熱圖),1會看見背景熱圖，2會看見熱圖疊加，3則會看到PAFs疊加(預測肢幹，會有線)。 

至此，共補充了21個flag，其餘如果使用到，可上[Openpose GitHub的檔案](https://reurl.cc/VXr6xR) 上查詢。
### 程式碼解釋
1. **op.Datum().cvInputData** : 輸入的圖片，放的為圖片。可透過陣列將所有在路徑中的圖片都放入opWrapper。
2. **op.Datum().poseKeypoint** : 輸出身體各個肢幹的點及精確率。
![](https://i.imgur.com/lH9j0bM.png)
最外層為人，第二層為軀幹，最內層為所有軀幹，
如:第一個人[[[鼻子][脖子]...]第二個人[[[鼻子][脖子]...]。
又最內層各個軀幹包含所預測點的位置及精確率。
3. **op.Datum().cvOutputData** : 輸出的圖片，依照前面flag可能有不同輸出。
一般輸出 : 
![](https://i.imgur.com/VGeWdAB.jpg)
去背後輸出 : 
![](https://i.imgur.com/3j2O0mx.jpg)
part_to_show = 3的輸出 : 
![](https://i.imgur.com/jtG0xpT.jpg)

以上的程式碼解析都可以在[Openpose提供的網站](https://reurl.cc/0O5OK6) 找到相對應的資料。

### 完整程式碼
```
from sys import platform
import argparse
import time
import numpy as np

t = 0

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    parser.add_argument("--num_gpu", default=op.get_gpu_number(), help="Number of GPUs.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["num_gpu"] = int(vars(args[0])["num_gpu"])
    #params["disable_blending"] = True
    params["part_to_show"] = 3
    numberGPUs = int(params["num_gpu"])

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);

    # Read number of GPUs in your system
    start = time.time()

    # Process and display images
    for imageBaseId in range(0, len(imagePaths), numberGPUs):

        # Create datums
        datums = []
        images = []

        # Read and push images into OpenPose wrapper
        for gpuId in range(0, numberGPUs):

            imageId = imageBaseId+gpuId
            if imageId < len(imagePaths):

                imagePath = imagePaths[imageBaseId+gpuId]
                datum = op.Datum()
                images.append(cv2.imread(imagePath))
                datum.cvInputData = images[-1]
                datums.append(datum)
                opWrapper.waitAndEmplace([datums[-1]])

        # Retrieve processed results from OpenPose wrapper
        for gpuId in range(0, numberGPUs):

            imageId = imageBaseId+gpuId
            if imageId < len(imagePaths):
                datum = datums[gpuId]
                opWrapper.waitAndPop([datum])

                print("Body keypoints: \n" + str(datum.poseKeypoints))
                if not args[0].no_display:
                    cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
                    cv2.imwrite("ForwardFlexion_incorrect"+str(t) + ".jpg",datum.cvOutputData)
                    key = cv2.waitKey(15)
                    if key == 27: break
        t += 1

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)

```



