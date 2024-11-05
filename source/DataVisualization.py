import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Đọc dữ liệu từ các file CSV
data_class1 = pd.read_csv('../Data/click.csv')
data_class2 = pd.read_csv('../Data/non_click.csv')
data_class1 = data_class1.fillna(0)
data_class2 = data_class2.fillna(0)

# Gắn nhãn cho mỗi lớp
data_class1['label'] = 0
data_class2['label'] = 1

# Ghép hai tập dữ liệu lại với nhau
data = pd.concat([data_class1, data_class2])

# Tách phần dữ liệu và nhãn
X = data.drop(columns=['label'])
y = data['label']

X = X.fillna(0)

# Giảm chiều bằng PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Vẽ biểu đồ phân tán
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='red', label='Click')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='blue', label='Non Click')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('PCA Projection of the Two Classes')
plt.show()
