# Script training cho tiếng Việt
# Đảm bảo bạn đã chạy các bước chuẩn bị dữ liệu trước khi chạy script này.

$ErrorActionPreference = "Stop"

Write-Host "Bắt đầu training IndexTTS2 tiếng Việt..." -ForegroundColor Green

uv run python trainers/train_gpt_v2.py `
    --train-manifest runs/vi/processed/gpt_pairs_train.jsonl `
    --val-manifest runs/vi/processed/gpt_pairs_val.jsonl `
    --tokenizer runs/vi/tokenizer/vi_bpe.model `
    --config checkpoints/config.yaml `
    --base-checkpoint checkpoints/gpt_old.pth `
    --output-dir runs/vi/finetune_ckpts `
    --batch-size 8 `
    --grad-accumulation 2 `
    --epochs 10 `
    --learning-rate 1e-5 `
    --weight-decay 1e-2 `
    --warmup-steps 1000 `
    --log-interval 10 `
    --val-interval 2000 `
    --grad-clip 1.0 `
    --text-loss-weight 0.2 `
    --mel-loss-weight 0.8 `
    --amp `
    --resume auto

# Lưu ý:
# --amp: Bật Mixed Precision cho CUDA (nhanh hơn, ít VRAM hơn)
# --batch-size: Giảm nếu bị lỗi Out Of Memory
