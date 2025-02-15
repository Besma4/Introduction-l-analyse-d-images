import numpy as np
import cv2
import random


#Exercice 1 : Etiquetage en composantes connexes
def Voisins(image, px, py):
    v = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    new_v = []

    for n in v:
        new_p = (px + n[0], py + n[1])

        if 0 <= new_p[0] < image.shape[0] and 0 <= new_p[1] < image.shape[1]:
            new_v.append(new_p)

    return new_v

def parcoursCC(image, labels, px, py, label):
    stack = [(px, py)]
    binr = image[px, py]

    while stack:
        x, y = stack.pop()
        labels[x, y] = label

        for nx, ny in Voisins(image, x, y):
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                if image[nx, ny] == binr and labels[nx, ny] == 0:
                    labels[nx, ny] = label
                    stack.append((nx, ny))


def regions(image):
    labels = np.zeros_like(image[:,:,0], dtype=int)
    label = 1

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if np.all(image[x, y] == [255, 255, 255]) and labels[x, y] == 0:
                parcoursCC(image[:,:,0], labels, x, y, label)
                label += 1

    return labels

def couleurs(labels):
    img_clr = np.zeros((*labels.shape, 3), dtype=np.uint8)
    lbls = np.unique(labels)
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#8E44AD"]



    color_map = {}
    for i, label in enumerate(lbls):
        if label != 0:

            hex = colors[i % len(colors)]
            rgb = tuple(int(hex[j:j+2], 16) for j in (1, 3, 5))
            color_map[label] = rgb

    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels[x, y] != 0:
                img_clr[x, y] = color_map[labels[x, y]]

    return img_clr


image1= cv2.imread('binary.png')
binr = np.array(image1)

image_label = regions(binr)
image_clr = couleurs(image_label)
image_f = cv2.resize(image_clr, (600, 600))

cv2.imshow("Exercice 1",image_f)
cv2.waitKey(0)
cv2.destroyAllWindows()





#Exercice 2 : Filtre d'aire

def ccLabel(image):
    res = np.zeros(image.shape, dtype=np.int32)
    h, w = image.shape
    label = 1

    
    def dfs(x, y, current_label):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if res[cx, cy] == 0 and image[cx, cy] != 0:
                res[cx, cy] = current_label
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        stack.append((nx, ny))
    
    for i in range(h):
        for j in range(w):
            if image[i, j] != 0 and res[i, j] == 0:
                dfs(i, j, label)
                label += 1
    
    return res

def ccAreaFilter(image, size):
    assert size > 0
    lbl_image = ccLabel(image)
    unique, counts = np.unique(lbl_image, return_counts=True)
    area_map = dict(zip(unique, counts))
    
    filtered_image = np.zeros_like(image)
    for label, area in area_map.items():
        if area >= size and label != 0:
            filtered_image[lbl_image == label] = 255
    
    return filtered_image



image2 = cv2.imread("binary.png", cv2.IMREAD_GRAYSCALE)
fil_image = ccAreaFilter(image2, 400)
image_f2 = cv2.resize(fil_image, (600, 600))
cv2.imshow("Exercice 2",image_f2)
cv2.waitKey(0)
cv2.destroyAllWindows()





#Exercice 3 : Etiquetage en composantes connexes - Mieux

def Voisins(image, x, y):
    v = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    valid_v = []
    
    for dx, dy in v:
        nx, ny = x + dx, y + dy
        if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
            valid_v.append((nx, ny))
    
    return valid_v

def parcoursCC(image, labels, px, py, label):
    stack = [(px, py)]
    binr = image[px, py]

    while stack:
        x, y = stack.pop()
        labels[x, y] = label

        for nx, ny in Voisins(image, x, y):
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                if image[nx, ny] == binr and labels[nx, ny] == 0:
                    labels[nx, ny] = label
                    stack.append((nx, ny))

def cc_two_pass_label(image):
    labels = np.zeros_like(image, dtype=np.int32)
    label = 1
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 255 and labels[i, j] == 0:
                parcoursCC(image, labels, i, j, label)
                label += 1
    
    return labels

def couleurs(labels):
    clr_image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    unique_lbl = np.unique(labels)
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#8E44AD"]

    color_map = {}
    for i, label in enumerate(unique_lbl):
        if label != 0:

            hex_color = colors[i % len(colors)]
            rgb_color = tuple(int(hex_color[j:j+2], 16) for j in (1, 3, 5))
            color_map[label] = rgb_color

    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels[x, y] != 0:
                clr_image[x, y] = color_map[labels[x, y]]

    return clr_image


image3 = cv2.imread("binary.png", cv2.IMREAD_GRAYSCALE)
labeled_image = cc_two_pass_label(image3)
colored_labels = couleurs(labeled_image)
    
image_f3 = cv2.resize(colored_labels, (600, 600))
cv2.imshow("Exercice 3", image_f3)
cv2.waitKey(0)
cv2.destroyAllWindows()




#Exercice 4 : Seuillage par histogramme et méthode d’Otsu

def threshold_otsu(image):

    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    
    best_threshold = 0
    max_var = 0
    
    for t in range(256):
        background = hist[:t]
        foreground = hist[t:]
        
        w_back = np.sum(background)
        w_fore = np.sum(foreground)
        
        if w_back == 0 or w_fore == 0:
            continue
        
        mean_back = np.sum(np.arange(t) * background) / w_back
        mean_fore= np.sum(np.arange(t, 256) * foreground) / w_fore
        
        var_between = w_back * w_fore * (mean_back - mean_fore) ** 2
        
        if var_between > max_var:
            max_var = var_between
            best_threshold = t
    
    binr_image = (image > best_threshold).astype(np.uint8) * 255
    return binr_image


image4 = cv2.imread("image1.png", cv2.IMREAD_GRAYSCALE)
binr_img = threshold_otsu(image4)
image_f4 = cv2.resize(binr_img, (600, 600))
cv2.imshow("Exercice 4", image_f4)

cv2.waitKey(0)
cv2.destroyAllWindows()



#Exercice 5 : Segmentation par croissance de régions

def voisins(h, w, x, y):
    v = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid_v = []
    for dx, dy in v:
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w:
            valid_v.append((nx, ny))
    return valid_v

def region_growing(image, threshold=15):
    h, w = image.shape 
    seg = np.zeros((h, w, 3), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    color_map = {}
    label = 1
    
    for i in range(h):
        for j in range(w):
            if visited[i, j] or image[i, j] == 0:
                continue
            
            stack = [(i, j)]
            seed_value = int(image[i, j])
            
            if label not in color_map:
                color_map[label] = tuple(random.randint(50, 255) for _ in range(3))
            
            while stack:
                x, y = stack.pop()
                if visited[x, y]:
                    continue
                visited[x, y] = True
                seg[x, y] = color_map[label]
                
                for nx, ny in voisins(h, w, x, y):
                    if not visited[nx, ny] and abs(int(image[nx, ny]) - seed_value) < threshold:
                        stack.append((nx, ny))
            
            label += 1
    
    return seg


image5 = cv2.imread("corner2.png", cv2.IMREAD_GRAYSCALE)
image_f5 = region_growing(image5)
    
cv2.imshow("Exercice 5", image_f5)
cv2.waitKey(0)
cv2.destroyAllWindows()



#Exercice 6 : Segmentation par décomposition/fusion (split/merge)

def split(image, x, y, w, h, threshold):
    region = image[y:y+h, x:x+w]
    mean, stddev = cv2.meanStdDev(region)
    
    if stddev[0][0] < threshold:
        return [(x, y, w, h)]
    
    half_w, half_h = w // 2, h // 2
    segs = []
    
    if half_w > 0 and half_h > 0:
        sub_regions = [
            (x, y),
            (x + half_w, y),
            (x, y + half_h),
            (x + half_w, y + half_h)
        ]
        
        for sub_x, sub_y in sub_regions:
            segs += split(image, sub_x, sub_y, half_w, half_h, threshold)
    
    return segs

def merge(image, segments):

    merged = np.zeros_like(image)
    
    for (x, y, w, h) in segments:
        for i in range(y, y + h):
            for j in range(x, x + w):
                merged[i, j] = image[i, j]
    
    return merged


image6 = cv2.imread("macaws.png", cv2.IMREAD_COLOR)

segments = split(image6, 0, 0, image6.shape[1], image6.shape[0], threshold=4)
image_f6 = merge(image6, segments)
    
cv2.imshow("Exercice 6", image_f6)
cv2.waitKey(0)
cv2.destroyAllWindows()



