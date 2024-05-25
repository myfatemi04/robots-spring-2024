from transformers import AutoModelForCausalLM, AutoTokenizer
import PIL.Image

moondream_model_id = "vikhyatk/moondream2"
moondream_revision = "2024-05-20"
moondream_model = AutoModelForCausalLM.from_pretrained(
    moondream_model_id,
    revision=moondream_revision,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
moondream_tokenizer = AutoTokenizer.from_pretrained(moondream_model_id)

def vqa(image: PIL.Image.Image, prompt: str):
    encoded_image = moondream_model.encode_image(image)
    return moondream_model.answer_question(encoded_image, prompt, moondream_tokenizer)
