from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

class LocalLLM:
    def __init__(self, model_name="microsoft/phi-2"):

        # Fix config
        config = AutoConfig.from_pretrained(model_name)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, max_tokens=200):

        prompt = prompt + "\nAnswer:\n"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean output
        text = text.replace(prompt, "")
        text = text.split("Exercise")[0]
        text = text.split("Example")[0]

        return text.strip()