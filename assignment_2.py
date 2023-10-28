import cv2 as cv
import numpy as np

src = cv.imread("crosses.bmp")
img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

th, img = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

labels = np.zeros(img.shape, dtype=int)
labelcount = 1

# connected component labeling
def ccl(img, labels, labelcount):
    rows, cols = img.shape
    equivalences = [-1] * labelcount

    def find(x):
        if equivalences[x] < 0:
            return x
        equivalences[x] = find(equivalences[x])
        return equivalences[x]

    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        if x_root != y_root:
            if equivalences[x_root] < equivalences[y_root]:
                equivalences[x_root] += equivalences[y_root]
                equivalences[y_root] = x_root
            else:
                equivalences[y_root] += equivalences[x_root]
                equivalences[x_root] = y_root

    for i in range(rows):
        for j in range(cols):
            if img[i][j] == 255:
                neighbors = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)
                ]
                neighbor_labels = []
                for ni, nj in neighbors:
                    if 0 <= i + ni < rows and 0 <= j + nj < cols and labels[i + ni][j + nj] > 0:
                        neighbor_labels.append(labels[i + ni][j + nj])

                if not neighbor_labels:
                    labels[i][j] = labelcount
                    equivalences.append(-1)
                    labelcount += 1
                else:
                    labels[i][j] = min(neighbor_labels)
                    for neighbor in neighbor_labels:
                        union(labels[i][j], neighbor)

    for i in range(rows):
        for j in range(cols):
            if labels[i][j] > 0:
                labels[i][j] = find(labels[i][j])

ccl(img, labels, labelcount)

# remove any small objects
considerable_labels = {}
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        if labels[i][j] != 0:
            if labels[i][j] not in considerable_labels:
                considerable_labels.update({labels[i][j]: 1})
            else:
                considerable_labels[labels[i][j]] += 1

labels1 = np.zeros(img.shape, dtype=int)
labels2 = np.zeros(img.shape, dtype=int)
labels3 = np.zeros(img.shape, dtype=int)

# construct a new image
colour_dict = {}
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        if labels[i][j] != 0 and labels[i][j] in considerable_labels:
            if labels[i][j] not in colour_dict.keys():
                color = np.random.randint(0, 255, size=3)
                colour_dict.update({labels[i][j]: color})
            labels1[i][j] = colour_dict[labels[i][j]][0]
            labels2[i][j] = colour_dict[labels[i][j]][1]
            labels3[i][j] = colour_dict[labels[i][j]][2]
rgb_img = np.stack((labels1, labels2, labels3), axis=-1)
new_img = rgb_img.astype(np.uint8)

print("Number of objects: ", len(colour_dict))

cv.imshow('og_img', img)
cv.imshow('new_img', new_img)
cv.waitKey(0)
cv.destroyAllWindows()
