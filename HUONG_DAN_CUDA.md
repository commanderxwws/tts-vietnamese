# Hướng dẫn chạy IndexTTS2 trên CUDA (GPU)

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy training/inference sử dụng GPU NVIDIA (CUDA).

## 1. Yêu cầu hệ thống

- **GPU NVIDIA**: Cần có GPU hỗ trợ CUDA (ví dụ: RTX 3060, 4090, A100, ...).
- **Driver**: Đã cài đặt driver mới nhất cho GPU từ trang chủ NVIDIA.
- **CUDA Toolkit**: (Tùy chọn) Nên cài đặt CUDA Toolkit 12.x nếu cần biên dịch thêm thư viện, nhưng `uv` sẽ tự tải PyTorch có sẵn CUDA.

## 2. Cài đặt môi trường

Dự án này sử dụng `uv` để quản lý thư viện, rất nhanh và tiện lợi.

1.  **Mở Terminal** (PowerShell hoặc Command Prompt) tại thư mục dự án.

2.  **Tạo môi trường ảo và cài đặt thư viện**:
    Lệnh sau sẽ tự động tải PyTorch phiên bản hỗ trợ CUDA (đã được cấu hình trong `pyproject.toml`).

    ```bash
    uv sync
    ```

    *Lưu ý: Quá trình này có thể mất một lúc để tải PyTorch (khoảng 2-3GB).*

## 3. Kiểm tra CUDA

Sau khi cài đặt xong, hãy chạy script kiểm tra để đảm bảo nhận diện được GPU:

```bash
uv run python verify_cuda.py
```

Nếu kết quả hiện ra tên GPU và "Success!", bạn đã sẵn sàng.
Nếu hiện "CUDA available: False", hãy kiểm tra lại driver hoặc thử cài lại môi trường: `uv sync --reinstall`.

## 4. Chạy Training trên CUDA

Các script training đã được viết để tự động nhận diện CUDA. Bạn chỉ cần chạy lệnh như bình thường.

Ví dụ chạy file `train.bat` (đã sửa để dùng `uv`):

```bash
.\train.bat
```

Hoặc chạy lệnh trực tiếp:

```bash
uv run python trainers/train_gpt_v2.py ^
  --train-manifest runs/vi/processed/gpt_pairs_train.jsonl ^
  --val-manifest runs/vi/processed/gpt_pairs_val.jsonl ^
  --tokenizer runs/vi/tokenizer/vi_bpe.model ^
  --config checkpoints/config.yaml ^
  --base-checkpoint checkpoints/gpt_old.pth ^
  --output-dir runs/vi/finetune_ckpts ^
  --batch-size 8 ^
  --grad-accumulation 2 ^
  --amp
```

*Lưu ý:*
- `--amp`: Kích hoạt chế độ Mixed Precision (FP16) giúp chạy nhanh hơn và tốn ít VRAM hơn.
- `--batch-size`: Nếu bị lỗi "Out of Memory" (OOM), hãy giảm số này xuống (ví dụ: 4 hoặc 2) và tăng `--grad-accumulation` lên tương ứng.

## 5. Chạy Inference trên CUDA

Script `infer_vi.py` cũng tự động dùng CUDA nếu có.

```bash
uv run python infer_vi.py ^
  --text "Xin chào, đây là thử nghiệm chạy trên GPU." ^
  --output ket_qua.wav ^
  --fp16
```

- `--fp16`: Dùng tính toán FP16 để tăng tốc độ inference trên GPU.

## 6. Xử lý lỗi thường gặp

- **CUDA Out of Memory**:
    - Giảm `--batch-size`.
    - Giảm độ dài câu input.
    - Tắt các ứng dụng khác đang dùng GPU.

- **Torch not compiled with CUDA enabled**:
    - Chạy lại `uv sync` để đảm bảo đã cài đúng phiên bản PyTorch.
    - Kiểm tra file `pyproject.toml` phần `[tool.uv.sources]` có trỏ đúng đến `pytorch-cuda` không.

## 7. Chạy WebUI trên CUDA

Để chạy giao diện web (Gradio) tận dụng sức mạnh GPU:

```bash
.\run_webui.ps1
```

Hoặc lệnh thủ công:

```bash
uv run python webui.py --fp16 --cuda_kernel
```

- `--fp16`: Giảm VRAM và tăng tốc độ.
- `--cuda_kernel`: Tăng tốc độ tạo sóng âm (Vocoder) nếu GPU hỗ trợ.
