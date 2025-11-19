# Script chạy WebUI với CUDA
# Tự động sử dụng GPU NVIDIA nếu có.

$ErrorActionPreference = "Stop"

Write-Host "Đang khởi động WebUI trên CUDA..." -ForegroundColor Green

# Tự động tải model nếu chưa có (được xử lý bởi webui.py)
# if (-not (Test-Path "checkpoints/config.yaml")) { ... }

# Chạy WebUI
# --fp16: Bật chế độ tính toán 16-bit để nhanh hơn trên GPU
# --cuda_kernel: (Tùy chọn) Bật kernel tối ưu cho BigVGAN nếu hỗ trợ
uv run python webui.py --cuda_kernel

# Nếu muốn chia sẻ ra ngoài mạng LAN, thêm tham số: --host 0.0.0.0
