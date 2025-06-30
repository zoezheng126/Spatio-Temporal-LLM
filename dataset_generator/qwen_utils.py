import re
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

# calculate relative distance
import os
import numpy as np
from tqdm import tqdm

class VLMPredictor:
    def __init__(self, model_name="Qwen/Qwen-VL", device="cuda", torch_type=torch.bfloat16, width=1280, height=720):
        """
        Initialize the detector with the specified model and tokenizer/processor.
        Supports Qwen/Qwen-VL and Qwen/Qwen2-VL-7B-Instruct.
        """
        self.model_name = model_name
        self.device = device
        self.width = width
        self.height = height

        if model_name == "Qwen/Qwen-VL":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device, trust_remote_code=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.processor = None  # Processor is not used for this model
        elif model_name == "Qwen/Qwen2-VL-7B-Instruct":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name, torch_dtype=torch_type
            )
            self.tokenizer = None  # Tokenizer is not used for this model
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def create_prompt(self, question, frame_list, type='image', downsample_factor=1):
        # Messages containing a images list as a video and a text query
        if self.model_name in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]:
            if type == 'video':
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": frame_list,
                                "fps": 2.0,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_list[i]} for i in range(len(frame_list))
                        ],
                    }
                ]
                messages[0]['content'].append({"type": "text", "text": question})

            if downsample_factor == 1:
                return messages

            dw, dh = self.width // downsample_factor, self.height // downsample_factor
            messages[0]["content"][0]["max_pixels"] = dw * dh
            return messages

        elif self.model == "Qwen/Qwen-VL":
            raise ValueError("TODO: add prompt template for Qwen/Qwen-VL")
        return messages

    def parse_output(self, output_str):
        return output_str
    
    def run_tokenizer(self, question=None, frame_list=None, prompt=None):
        if prompt == None:
            prompt = self.create_prompt(question, frame_list)

        if self.model_name in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]:
            text = self.processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(prompt)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
        elif self.model_name == "Qwen/Qwen-VL":
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        else:
            raise ValueError("Unsupported model for inference.")
        
        return inputs

    def run_inference(self, inputs):
        """
        Run inference on the specified image and return parsed results.
        Handles both models based on initialization.
        """

        if self.model_name == "Qwen/Qwen-VL":
            # Tokenize input for Qwen/Qwen-VL
            outputs = self.model.generate(**inputs)
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif self.model_name in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]:
            # Process input for Qwen/Qwen2-VL-7B-Instruct or Qwen/Qwen2-VL-7B-Instruct
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text[0]

        else:
            raise ValueError("Unsupported model for inference.")

        # Parse the output
        parsed_results = self.parse_output(output_text)
        return parsed_results