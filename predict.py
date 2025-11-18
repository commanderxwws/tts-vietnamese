import os


from indextts.infer_v2 import IndexTTS2


def main():
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    raw_text = "Chiến lược quân sự là nghệ thuật định hướng và sử dụng sức mạnh quân sự nhằm đạt được mục tiêu chính trị, và qua từng thời kỳ, con người đã phát triển nhiều cách tiếp cận khác nhau. Từ thời cổ đại, Tôn Tử đã nhấn mạnh yếu tố mưu lược và coi trọng việc giành thắng lợi bằng trí tuệ, tạo thế và đánh vào tâm lý đối phương hơn là chỉ dựa vào sức mạnh. Trong lịch sử, có những chiến lược như tiêu hao, tức là dùng sức mạnh liên tục để bào mòn lực lượng địch, hay quyết chiến nhanh, tập trung toàn bộ binh lực vào một trận đánh then chốt để xoay chuyển cục diện, như Hannibal ở Cannae hay Võ Nguyên Giáp ở Điện Biên Phủ."
    audio_prompt = "D:\\dataset\\minimax\\Hoai_Anh\\wavs\\00a36789-142d-4c30-9373-58cd090dd374.wav"

    if not os.path.exists(audio_prompt):
        print(f"Audio prompt file not found: {audio_prompt}")
        return

    result = tts.infer(
        spk_audio_prompt=audio_prompt,
        text=raw_text,
        output_path="gen.wav",
        verbose=True,
    )

    if result:
        print(f"Audio generated successfully: {result}")
    else:
        print("Inference completed but no audio was returned.")


if __name__ == "__main__":
    main()
