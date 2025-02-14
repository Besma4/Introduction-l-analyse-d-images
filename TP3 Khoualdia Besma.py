import numpy as np
import cv2
import random


'''
#Exercice 1 : Etiquetage en composantes connexes
def Voisins(image, px, py):
    neighbours = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    new_neighbours = []

    for n in neighbours:
        new_p = (px + n[0], py + n[1])

        if 0 <= new_p[0] < image.shape[0] and 0 <= new_p[1] < image.shape[1]:
            new_neighbours.append(new_p)

    return new_neighbours

def parcoursCC(image, labels, px, py, label):
    stack = [(px, py)]
    binary_value = image[px, py]

    while stack:
        x, y = stack.pop()
        labels[x, y] = label

        for nx, ny in Voisins(image, x, y):
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                if image[nx, ny] == binary_value and labels[nx, ny] == 0:
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
    colored_image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#8E44AD"]



    color_map = {}
    for i, label in enumerate(unique_labels):
        if label != 0:

            hex_color = colors[i % len(colors)]
            rgb_color = tuple(int(hex_color[j:j+2], 16) for j in (1, 3, 5))
            color_map[label] = rgb_color

    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels[x, y] != 0:
                colored_image[x, y] = color_map[labels[x, y]]

    return colored_image

img = cv2.imread('binary.png')

binary_image = np.array(img)

image_label = regions(binary_image)
image_clr = couleurs(image_label)
image_f = cv2.resize(image_clr, (600, 600))

cv2.imshow("image labelisee",image_f)
cv2.waitKey(0)
cv2.destroyAllWindows()






#Exercice 2 : Filtre d'aire

def ccLabel(image):
    res = np.zeros(image.shape, dtype=np.int32)
    h, w = image.shape
    label = 1
    label_map = {}
    
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
    labeled_image = ccLabel(image)
    unique, counts = np.unique(labeled_image, return_counts=True)
    area_map = dict(zip(unique, counts))
    
    filtered_image = np.zeros_like(image)
    for label, area in area_map.items():
        if area >= size and label != 0:
            filtered_image[labeled_image == label] = 255
    
    return filtered_image



image2 = cv2.imread("binary.png", cv2.IMREAD_GRAYSCALE)
filtered_image = ccAreaFilter(image2, 100)
#cv2.imwrite("output.png", filtered_image)
image_f2 = cv2.resize(filtered_image, (600, 600))
cv2.imshow("image labelisee",image_f2)
cv2.waitKey(0)
cv2.destroyAllWindows()





#Exercice 3 : Etiquetage en composantes connexes - Mieux
import cv2
import numpy as np

def get_neighbors(image, x, y):
    neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    valid_neighbors = []
    
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
            valid_neighbors.append((nx, ny))
    
    return valid_neighbors

def parcoursCC(image, labels, px, py, label):
    stack = [(px, py)]
    binary_value = image[px, py]

    while stack:
        x, y = stack.pop()
        labels[x, y] = label

        for nx, ny in Voisins(image, x, y):
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                if image[nx, ny] == binary_value and labels[nx, ny] == 0:
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
    colored_image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#8E44AD"]

    color_map = {}
    for i, label in enumerate(unique_labels):
        if label != 0:

            hex_color = colors[i % len(colors)]
            rgb_color = tuple(int(hex_color[j:j+2], 16) for j in (1, 3, 5))
            color_map[label] = rgb_color

    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels[x, y] != 0:
                colored_image[x, y] = color_map[labels[x, y]]

    return colored_image


image3 = cv2.imread("binary.png", cv2.IMREAD_GRAYSCALE)
labeled_image = cc_two_pass_label(image3)
colored_labels = couleurs(labeled_image)
    
image_f3 = cv2.resize(colored_labels, (600, 600))
cv2.imshow("Labeled Image", image_f3)
cv2.waitKey(0)
cv2.destroyAllWindows()





#Exercice 4 : Seuillage par histogramme et méthode d’Otsu

def threshold_otsu(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = image.shape[0] * image.shape[1]
    
    current_max, threshold, sum_total, sum_foreground, weight_background, weight_foreground = 0, 0, np.sum(np.arange(256) * hist), 0, 0, 0
    
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        
        sum_foreground += t * hist[t]
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance_between > current_max:
            current_max = variance_between
            threshold = t
    
    binary_image = (image > threshold).astype(np.uint8) * 255
    return binary_image


image4 = cv2.imread("image1.png", cv2.IMREAD_GRAYSCALE)
binary_img = threshold_otsu(image4)
image_f4 = cv2.resize(binary_img, (600, 600))
cv2.imshow("Otsu Thresholded Image", image_f4)

cv2.waitKey(0)
cv2.destroyAllWindows()





#Exercice 5 : Segmentation par croissance de régions



def get_neighbors(h, w, x, y):
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid_neighbors = []
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w:
            valid_neighbors.append((nx, ny))
    return valid_neighbors

def region_growing(image, threshold=15):
    h, w = image.shape  # Ensure we only take height and width
    segmented = np.zeros((h, w, 3), dtype=np.uint8)
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
                segmented[x, y] = color_map[label]
                
                for nx, ny in get_neighbors(h, w, x, y):
                    if not visited[nx, ny] and abs(int(image[nx, ny]) - seed_value) < threshold:
                        stack.append((nx, ny))
            
            label += 1
    
    return segmented


image5 = cv2.imread("corner2.png", cv2.IMREAD_GRAYSCALE)
image_f5 = region_growing(image5)
    
cv2.imshow("Segmented Image", image_f5)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
#Exercice 6 : Segmentation par décomposition/fusion (split/merge)

import cv2
import numpy as np

def split(image, x, y, w, h, threshold):
    region = image[y:y+h, x:x+w]
    mean, stddev = cv2.meanStdDev(region)
    
    if stddev[0][0] < threshold:
        return [(x, y, w, h)]
    
    half_w, half_h = w // 2, h // 2
    segments = []
    
    if half_w > 0 and half_h > 0:
        segments.extend(split(image, x, y, half_w, half_h, threshold))
        segments.extend(split(image, x + half_w, y, half_w, half_h, threshold))
        segments.extend(split(image, x, y + half_h, half_w, half_h, threshold))
        segments.extend(split(image, x + half_w, y + half_h, half_w, half_h, threshold))
    
    return segments

def merge(image, segments):
    merged = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    color_map = {}
    label = 1
    
    for (x, y, w, h) in segments:
        if label not in color_map:
            color_map[label] = tuple(np.random.randint(50, 255, 3).tolist())
        
        for i in range(y, y + h):
            for j in range(x, x + w):
                merged[i, j] = color_map[label]
        
        label += 1
    
    return merged

if __name__ == "__main__":
    img = cv2.imread("macaws.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error loading image. Make sure the file exists and is a valid image.")
    
    segments = split(img, 0, 0, img.shape[1], img.shape[0], threshold=15)
    segmented_img = merge(img, segments)
    
    cv2.imshow("Segmented Image", segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



