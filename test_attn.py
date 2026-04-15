import torch
from transformers import AutoModelForCausalLM

with open('result.txt', 'w') as f:
    try:
        model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m')
        input_ids = torch.randint(0, 1000, (1, 10))
        output = model(input_ids, output_attentions=True)
        
        f.write(f"Has attentions: {hasattr(output, 'attentions')}\n")
        if hasattr(output, 'attentions') and output.attentions is not None:
            f.write(f"Attentions length: {len(output.attentions)}\n")
            f.write(f"First attention type: {type(output.attentions[0])}\n")
            if hasattr(output.attentions[0], 'shape'):
                f.write(f"First attention shape: {output.attentions[0].shape}\n")
            elif isinstance(output.attentions[0], tuple):
                f.write(f"First attention is tuple of len {len(output.attentions[0])}\n")
                f.write(f"First attention[0] type: {type(output.attentions[0][0])}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
