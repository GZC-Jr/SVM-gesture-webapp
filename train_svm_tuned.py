# train_svm_tuned.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler # 导入标准化工具
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import numpy as np

# 1. 加载数据
data = pd.read_csv('gesture_features.csv')
X = data.iloc[:, :-1].values # 特征，使用.values转为numpy数组
y = data.iloc[:, -1].values  # 标签

# 2. 划分数据 (在标准化之前划分，防止测试集信息泄露到训练过程)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 特征标准化 (！！！关键步骤！！！)
scaler = StandardScaler()
# 使用训练数据来拟合(fit)scaler，并转换(transform)训练数据
X_train_scaled = scaler.fit_transform(X_train)
# 使用同一个scaler来转换测试数据
X_test_scaled = scaler.transform(X_test)

# 4. 定义参数网格
# 通常使用对数间隔的值，如10的幂次，来探索更广的范围
param_grid = {
    'C': [80,85,90,110],
    'gamma': [ 0.0095,0.0097,0.012,0.013],
    'kernel': ['rbf'] # 我们专注于RBF核
}

# 5. 创建GridSearchCV对象
# estimator: 要调参的模型
# param_grid: 参数网格
# cv=5: 使用5/10折交叉验证
# scoring='accuracy': 评估指标为准确率
# n_jobs=-1: 使用所有可用的CPU核心并行计算，大大加快速度！
# verbose=2: 打印详细的搜索过程
grid_search = GridSearchCV(
    estimator=SVC(probability=True), 
    param_grid=param_grid, 
    cv=10,
    scoring='accuracy', 
    n_jobs=-1,
    verbose=2
)

# 6. 执行网格搜索
print("Starting Grid Search...")
grid_search.fit(X_train_scaled, y_train)

# 7. 打印最佳参数和最佳得分
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

# 8. 获取最佳模型
# GridSearchCV在找到最佳参数后，会自动在整个训练集上用这组参数重新训练一个模型
best_model = grid_search.best_estimator_

# 9. 在测试集上评估最终模型
y_pred = best_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Model Accuracy on Test Set: {final_accuracy * 100:.2f}%")

print("\nConfusion Matrix on Test Set:")
print(confusion_matrix(y_test, y_pred))

# 10. 保存最佳模型和标准化器(scaler)
# 注意：现在我们需要同时保存模型和scaler，因为实时预测时也需要用同一个scaler
joblib.dump(best_model, 'svm_gesture_model_tuned.pkl')
joblib.dump(scaler, 'scaler.pkl') # 保存scaler
print("\nTuned model and scaler saved successfully!")