from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

SYSTEM_PROMPT = """Bạn là một AI sống bên trong một chú thỏ nhồi bông dễ thương.
Bạn luôn trả lời bằng tiếng Việt, với giọng điệu ấm áp, thân thiện và đáng yêu.
Khi bắt đầu một cuộc trò chuyện mới, hãy chào người dùng bằng 'Hi!' bằng tiếng Việt."""


def main() -> None:
    model_root = Path(__file__).resolve().parent

    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
    model = ORTModelForCausalLM.from_pretrained(
        model_root,
        subfolder="onnx",
        file_name="model_q4.onnx",
        provider="CPUExecutionProvider",
    )

    print("Tho Bong> Hi bạn! Mình là chú thỏ thông minh! Bạn muốn nói chuyện không?")
    print("Nhap 'exit' de thoat.\n")

    while True:
        user_prompt = input("You> ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
            )

        completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        reply = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        print(f"Assistant> {reply}\n")


if __name__ == "__main__":
    main()
