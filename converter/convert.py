from re import A
import numpy as np
import coremltools as ct
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
# Get a pytorch model and save it as a *.pt file
model = AutoModel.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student", return_dict=False)
# tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
# text = "I am so happy to do this task"



# token_dataset = tokenizer(text)
# input = token_dataset['input_ids']

#print(token_dataset)
model.cpu()


model.eval()
example_input = torch.randint(1000,(1, 20))
traced_model = torch.jit.trace(model, example_input, strict=False)
traced_model.eval()
#traced_model.save("distilbert_emotion.pt")

# Convert the saved PyTorch model to Core ML
mlmodel = ct.convert(traced_model,
                    inputs=[ct.TensorType(name = "input", shape=example_input.shape, dtype = np.int32 )]
                    )

mlmodel.save("distilbert_emotion.mlmodel")