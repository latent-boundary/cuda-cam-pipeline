# ===== コンパイラ設定 =====
NVCC = nvcc
GXX  = g++

# ===== パス設定 =====
SRC_DIR     = src
CUDA_DIR    = src/cuda
BUILD_DIR   = build
OBJ_DIR     = $(BUILD_DIR)/obj
INC_DIR     = include

# ===== OpenCV / CUDA =====
CFLAGS  = -I/usr/include/opencv4 -I/usr/local/cuda/include -I$(INC_DIR)
LDFLAGS = -L/usr/lib/aarch64-linux-gnu \
	-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs \
	-L/usr/local/cuda/lib64 -lcudart

# ===== オブジェクトファイル =====
OBJS = $(OBJ_DIR)/sobel_shared.o \
	$(OBJ_DIR)/gaussian5x5_shared.o

# ===== 最終バイナリ =====
TARGET = $(BUILD_DIR)/bridge

# ===== ルール =====
all: $(TARGET)

$(OBJ_DIR)/%.o: $(CUDA_DIR)/%.cu
	$(NVCC) -c $< -o $@

$(TARGET): $(SRC_DIR)/bridge.cpp $(OBJS)
	$(GXX) $(SRC_DIR)/bridge.cpp $(OBJS) -o $@ $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(OBJ_DIR)/*.o
	rm -f $(TARGET)
