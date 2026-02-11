# 🚀 PROJECT MASTER PLAN: ANTIGRAVITY MATH SOLVER

**Project:** Hệ thống Nhận diện & Giải toán viết tay (Handwritten Math OCR & Solver)  
**Team:** Antigravity  
**Role:** AI & Software Engineering Team  
**Timeline dự kiến:** 8 - 10 Tuần  

## 📖 Giới thiệu
Dự án nhằm xây dựng một hệ thống AI có khả năng nhận diện công thức toán học viết tay (Handwritten Math OCR) và tự động giải quyết bài toán đó. Hệ thống kết hợp giữa Vision Transformer (ViT) cho việc nhận diện hình ảnh và các thư viện toán học tượng trưng (SymPy) để đưa ra lời giải chi tiết.

---

## 📅 GIAI ĐOẠN 1: PHÂN TÍCH & ĐẶC TẢ (Weeks 1-2)

**Mục tiêu:** Xác định rõ Input/Output và chuẩn bị "nguyên liệu" cho AI.

### 1.1. Phân tích yêu cầu phần mềm (Software Requirements)
- **Chức năng cốt lõi:**
  - Người dùng chụp ảnh/upload ảnh chứa công thức toán.
  - Hệ thống crop ảnh, tiền xử lý (khử nhiễu, cân bằng sáng).
  - AI nhận diện ra chuỗi LaTeX (VD: `\int_{0}^{1} x^2 dx`).
  - Module Solver giải ra kết quả cuối cùng.
- **Yêu cầu phi chức năng (KPIs):**
  - Độ trễ (Latency): < 2 giây/request (GPU T4) hoặc < 4 giây (CPU).
  - Độ chính xác (Accuracy): > 90% trên tập test CROHME.
  - Concurrency: Chịu tải tối thiểu 10 requests/giây.

### 1.2. Đặc tả kỹ thuật & Dữ liệu (AI Specs & Data)
- **Kiến trúc AI (SOTA):** Vision Encoder-Decoder.
  - **Encoder:** ViT (Vision Transformer) hoặc ResNet-101.
  - **Decoder:** GPT-2 (small) hoặc RoBERTa (sinh token LaTeX).
- **Dữ liệu (Data Pipeline):**
  - Nguồn: Tập dữ liệu CROHME (2014/2016/2019).
  - Data Augmentation: Gaussian Noise, Elastic Transform, Random Rotation (+/- 15 độ), Brightness Contrast.

---

## 📐 GIAI ĐOẠN 2: THIẾT KẾ HỆ THỐNG THEO CHUẨN UML (Week 3)

**Mục tiêu:** Xây dựng bản vẽ kỹ thuật cho hệ thống.

### 2.1. Kiến trúc hệ thống (System Architecture)
Mô hình Microservices đơn giản hóa:
- **Frontend (Client):** Streamlit (Web) hoặc Flutter (Mobile).
- **API Gateway:** NGINX (Load Balancing).
- **Backend Core:** FastAPI (Python) - Xử lý logic nghiệp vụ.
- **AI Inference Service:** Docker Container riêng chạy PyTorch.

### 2.2. Các biểu đồ UML bắt buộc (Design Artifacts)
- **Use Case Diagram:** Actor (Student, Admin) ↔ Use Cases (Scan Math, View Solution, Export PDF).
- **Sequence Diagram:** User Upload → Backend → Preprocessing → AI Model → SymPy Solver → Response.
- **Activity Diagram:** Grayscale → Binarization → Resize.

---

## 💻 GIAI ĐOẠN 3: CÀI ĐẶT & TỐI ƯU HÓA (Weeks 4-7)

**Mục tiêu:** Coding (Giai đoạn trọng tâm).

### 3.1. Module AI (The Brain)
- **Framework:** PyTorch, HuggingFace Transformers.
- **Task 1:** Xây dựng `DatasetLoader` (CROHME + token hóa LaTeX).
- **Task 2:** Huấn luyện mô hình (Training Loop).
  - Loss: Cross-Entropy Loss + Label Smoothing.
  - Optimizer: AdamW.
- **Task 3:** Optimization (Mixed Precision fp16, ONNX Runtime).

### 3.2. Module Solver & Backend
- **Solver Engine:** Parser chuyển đổi LaTeX → SymPy.
- **Xử lý lỗi:** Levenshtein Distance để sửa lỗi OCR.
- **API:** FastAPI endpoint `/predict`.

---

## 📦 GIAI ĐOẠN 4: ĐÓNG GÓI & TRIỂN KHAI (Week 8)

**Mục tiêu:** Biến code thành sản phẩm chạy được (Deliverable).

### 4.1. Dockerization
- **Dockerfile:** Base Image `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`.
- **docker-compose.yml:** Kết nối Frontend và Backend AI.

### 4.2. UI Integration
- **Giao diện Demo:** Upload ảnh, Hiển thị LaTeX (KaTeX), Hiển thị lời giải.

---

## 📝 GIAI ĐOẠN 5: THUYẾT MINH & BÁO CÁO (Week 9-10)

**Mục tiêu:** Bảo vệ thành công.
- Viết báo cáo (Thesis/Report).
- Quay video demo.

---

## 🛠 TECH STACK

| Component | Technology | Lý do chọn |
|-----------|------------|------------|
| Language | Python 3.9+ | Hệ sinh thái AI mạnh nhất. |
| Deep Learning | PyTorch, Transformers | Support kiến trúc Encoder-Decoder tốt nhất. |
| Vision Backbone | ViT / DeiT | Hiệu suất cao hơn CNN truyền thống. |
| Backend API | FastAPI | Nhanh, support Async/Await. |
| Math Engine | SymPy | Tính toán đại số tượng trưng mạnh mẽ. |
| Deployment | Docker | "Write once, run anywhere". |

---
*Created by Antigravity AI Team*
