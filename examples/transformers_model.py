import torch
from nnprofiler import LayerProf, get_children
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", return_dict=True
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
labels = torch.tensor([1, 0]).unsqueeze(0)

# Warm-up
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

with LayerProf(model) as prof:
    for name, _ in get_children(model):
        prof.attach_backward_hook(name)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    prof.get_timings()
    print(model)
