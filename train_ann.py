import os
import json
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# TỐI ƯU SỨC MẠNH CHO CHIP KHỦNG i9-14900HX
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

# ==========================================
# PHẦN 1: QUÉT THƯ MỤC, TỰ ĐỘNG GIẢI NÉN & NẠP DATA
# ==========================================
with open('classes.json', 'r', encoding='utf-8') as f:
    classes = json.load(f)

X = []
y = []

print("=== BẮT ĐẦU QUÉT, GIẢI NÉN VÀ CHUẨN HÓA DATA ===")
max_samples_per_class = 20000

for idx, class_name in enumerate(classes):
    npy_file = f"{class_name}.npy"
    zip_file = f"{class_name}.npy.zip"
    
    # Tự động giải nén tại chỗ nếu chỉ có file .zip
    if not os.path.exists(npy_file):
        if os.path.exists(zip_file):
            print(f"[{idx+1}/{len(classes)}] 📦 Đang giải nén tự động: {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
        else:
            print(f"[CẢNH BÁO] Không tìm thấy cả file .npy và .npy.zip của class: {class_name}")
            continue

    data = np.load(npy_file)
    data = data[:max_samples_per_class]

    X.append(data)
    y.append(np.full(data.shape[0], idx))
    print(f"[{idx+1}/{len(classes)}] 💾 Đã nạp thành công mảng: {class_name}")

if len(X) == 0:
    raise ValueError("❌ LỖI KHÔNG NẠP ĐƯỢC DỮ LIỆU: Vui lòng kiểm tra lại thư mục làm việc!")

X = np.concatenate(X)
y = np.concatenate(y)

print("\nĐang chuẩn hóa dữ liệu toàn cục (chia 255)...")
X = (X / 255.0).astype(np.float32)

# Trích hẳn 20% dữ liệu gốc ra làm tập tốt nghiệp biệt lập
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# PHẦN 2: XÂY DỰNG MẠNG ANN SÁCH GIÁO KHOA CỦA NHÓM
# ==========================================
model = Sequential([
    Dense(784, activation='relu', input_shape=(784,)),
    BatchNormalization(), 
    Dropout(0.3),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- BẢNG CẤU TRÚC MODEL ---")
model.summary()

with open('model_info.txt', 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("✅ Đã ghi nhận cấu trúc vào file 'model_info.txt'")

# ==========================================
# PHẦN 3: TIẾN HÀNH TRAINING VỚI BÍ KÍP ÉP XUNG i9
# ==========================================
early_stop = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

print("\n🚀 BẮT ĐẦU TRAINING (Tối đa 50 vòng)...")
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=512, 
                    validation_split=0.2, # Lấy 20% của tập train để làm bài thi thử kiểm tra răng cưa
                    callbacks=[early_stop, reduce_lr],
                    shuffle=True)

# ==========================================
# PHẦN 4: VẼ BIỂU ĐỒ NGHIỆM THU ĐẸP (ĐÃ CHUẨN HÓA THUẬT NGỮ)
# ==========================================
print("\nĐang xuất biểu đồ phân tích...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

# Đồ thị tỷ lệ chính xác
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b-', label='Train Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, 'r--', label='Val Accuracy', linewidth=2) # Đã sửa thành Val
plt.title('Độ chính xác (Accuracy)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Tỷ lệ')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Đồ thị sai số
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b-', label='Train Loss', linewidth=2)
plt.plot(epochs_range, val_loss, 'r--', label='Val Loss', linewidth=2) # Đã sửa thành Val
plt.title('Độ sai lệch (Loss)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Đã lưu ảnh biểu đồ hoàn chỉnh vào file 'training_history.png'")

# ==========================================
# PHẦN 5: ĐÁNH GIÁ CUỐI CÙNG VÀ XUẤT MODEL (.h5)
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🎯 ĐỘ CHÍNH XÁC KHÁCH QUAN CUỐI CÙNG TRÊN TẬP TEST BIỆT LẬP: {test_acc * 100:.2f}%")

model.save('emoji_model.h5')
print("\n🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH! emoji_model.h5 đã có.")