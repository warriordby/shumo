from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys
# 训练模型
model_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear')  # 可以选择不同的核函数，如 'rbf', 'poly' 等
log_reg = LogisticRegression()
tree = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
gbm = GradientBoostingClassifier()
# 将模型保存到列表
model_list = [model_RandomForest, svm_model, log_reg, tree, knn, gbm]
# 并使用setattr保存模型的名称
for model in model_list:
    setattr(sys.modules[__name__], model.__class__.__name__, model)