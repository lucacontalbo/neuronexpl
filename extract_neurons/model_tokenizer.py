"""Tokenizer of a model, with the fixes some model families need. Used by the corpus, the map tools and the classifier,
so that all of them tokenize and format chats in the same way.

Mistral family (Ministral 3, Mistral Small 3):
- the tokenizer must be loaded with fix_mistral_regex=True, otherwise words are split into the wrong tokens;
- with no system message, the chat template inserts a long default system prompt (~600 tokens, with today's date).
  We add an empty system message instead, so that these models, like Qwen and Gemma, see no system prompt.
"""
from transformers import AutoConfig, AutoTokenizer

MISTRAL_TYPES = {"mistral", "mistral3", "ministral", "ministral3"}


def load_tokenizer(model_path):
    is_mistral = AutoConfig.from_pretrained(model_path).model_type in MISTRAL_TYPES
    tok = AutoTokenizer.from_pretrained(model_path, **({"fix_mistral_regex": True} if is_mistral else {}))
    tok.add_empty_system = is_mistral  # read by chat_text
    return tok


def chat_text(tok, messages, **kwargs):
    """The chat template applied to the messages (as text), with an empty system message where needed."""
    messages = list(messages)
    if tok.add_empty_system and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": ""}] + messages
    return tok.apply_chat_template(messages, tokenize=False, **kwargs)
