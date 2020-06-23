def preImage(pil_image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    th = 0
    max_val = 255
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     plt.imshow(img,cmap = 'gray')
#     plt.show()
#     print(img.shape)

    ret, thresh1 = cv2.threshold(img_gray, th , max_val,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)


    kernel = np.ones((3,3),np.uint8)

    dilate = cv2.dilate(thresh1,kernel,iterations = 9)

    cnt, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for c in cnt:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2) 
        boxes.append(img[y:y+h, x:x+w])

    length = len(boxes)


    height = [32,64,96,128,160,800]
    w_32_64 = [128,160,192,224,256,320,384,480]
    w_96 = 384
    w_128 = [480,512,800]
    w_160 = [480,800]
    w_800 = 800


    for a in range(length):
        ht = boxes[a].shape[0]
        wt = boxes[a].shape[1]

        for i in range(6):
            if height[i]>=ht:
                h_ = height[i]
                h_add = height[i]-ht
                break

        if h_ == 32 or 64:
            for i in range(8):
                if w_32_64[i] > wt:
                    w_add = w_32_64[i]-wt
                    break

        if h_ == 96:
            w_add = w_96 - wt

        if h_ == 128:
            for i in range(3):
                if w_128[i] > wt:
                    w_add = w_128[i]-wt
                    break 

        if h_ == 160:
            for i in range(2):
                if w_160[i] > wt:
                    w_add = w_160[i]-wt
                    break 

        if h_ == 800:
            w_add = w_800 - wt

        # padding
        h1 = h_add - h_add//2
        h2 = h_add//2
        w1 = w_add - w_add//2
        w2 = w_add//2

        boxes[a]= cv2.copyMakeBorder(boxes[a],h1,h2,w1,w2,cv2.BORDER_CONSTANT,value = (255,255,255))

    equations = []
        
    for box in boxes:
        print(box.shape)
#         box = cv2.resize(box, (box.shape[1]//2, box.shape[0]//2))
#         print(box.size)
#         box = cv2.resize(box, (box.shape[1]*2, box.shape[0]*2))
#         print(box.size)
        equations.append(box)

    return Image.fromarray(cv2.cvtColor(equations[0], cv2.COLOR_BGR2RGB))
