#!/usr/bin/env python3


from threading import Thread

from queue import Queue
from tokenizers.decoders import DecodeStream

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# from transformers import TextIteratorStreamer


class IncrementalTextIteratorStreamer:
    def __init__(
        self,
        tokenizer,
        skip_prompt=False,
        timeout=None,
        skip_special_tokens=True,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout

        self.queue = Queue()
        self.stop_signal = object()

        self.stream = DecodeStream(
            skip_special_tokens=skip_special_tokens
        )

        self.first_put = True

    def put(self, value):
        # First call from generate() contains the prompt
        if self.first_put:
            self.first_put = False

            if self.skip_prompt:
                return

        ids = value.detach().cpu().tolist()

        # generate() normally gives [[id]] or [[id1, id2, ...]]
        if isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("Only batch size 1 is supported")
            ids = ids[0]

        text = self.stream.step(
            self.tokenizer.backend_tokenizer,
            ids
        )

        if text:
            self.queue.put(text)

    def end(self):
        self.queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get(timeout=self.timeout)

        if item is self.stop_signal:
            raise StopIteration

        return item


class LLMEngine():
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        
        self.last_generated = ""
        self.last_prompt = ""
        self.messages = []

        self.reset_session()
        
        self.message_length_limit = 100
        


    def reset_session(self):
        self.last_generated = ""
        self.last_prompt = ""
        self.messages = []
        # self.messages = [
        #     {"role": "system", "content": "You are a helpful assistant that will speak in the same language as the user. Your responses are wrong sometimes, but you do your best to be accurate."},
        #     {"role": "user", "content": "I want to know more about the moon. Explain the moon to me, please!"},
        #     {"role": "assistant", "content": "The moon is a natural satellite of Earth. It orbits around Earth in the same direction as Earth's rotation. It looks like a disk, but it is actually a nearly spherical object. It is composed primarily of rock and dust, along with occasional small meteorites. It shines at night by reflecting sunlight. Additionally, the Moon features craters and lunar maria, which were formed by impacts and erosion on its surface."},
        
        #     {"role": "user", "content": "What is 2+2?"},
        #     {"role": "assistant", "content": "2+2 is 4."},
        # ]
        
        
    def chat(self, prompt):
        generation_args = self._setup_chat(prompt)
        return self._full_response(generation_args, stop_token="<|im_end|>")
    
    def complete(self, prompt):
        generation_args = self._setup_completion(prompt)
        return self._full_response(generation_args)
    
    def stream_chat(self, prompt):
        generation_args = self._setup_chat(prompt)
        return self._stream_response(generation_args)
    
    def stream_complete(self, prompt):
        generation_args = self._setup_completion(prompt)
        return self._stream_response(generation_args)
    
    def print_complete(self, prompt):
        response = self.stream_complete(prompt)
        for chunk in response:
            print(chunk, end="", flush=True)
        return
    
    def print_chat(self, prompt):
        response = self.stream_chat(prompt)
        for chunk in response:
            print(chunk, end="")
        return
    
    def _setup_chat(self, prompt, stop_token=None):
        self.messages.append({
            "role": "user",
            "content": prompt
        })
        full_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
        )
        generation_args = self._setup_completion(full_prompt,
                                            stop_token=stop_token)
        
        return generation_args
    
    def _setup_completion(self, prompt, stop_token=None):
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)
        
        self.last_prompt = prompt
        
        
        generation_args = self._generation_args(inputs, stop_token)
        
        return generation_args
        
    
    def _generation_args(self, inputs, stop_token=None):
        eos_token_ids = [self.tokenizer.eos_token_id]
    
        if stop_token is not None:
            eos_token_ids.append(
                self.tokenizer.convert_tokens_to_ids(stop_token)
            )
    
        generation_args = {
            **inputs,
            "max_new_tokens": 100,
            "do_sample": False,
            "eos_token_id": eos_token_ids,
        }
        return generation_args
    
    def _full_response(self, generation_args):
        
        with torch.no_grad():
            output = self.model.generate(**generation_args)
        
        input_length = generation_args["input_ids"].shape[1]

        generated = output[0][input_length:]
        
        text_response = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )
        
        self.last_response = text_response
        
        return text_response
        
    
    def _stream_response(self, generation_args):
        
        streamer = IncrementalTextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        new_generation_args = generation_args | {"streamer": streamer}

        thread = Thread(
            target=self.model.generate,
            kwargs=new_generation_args
        )

        thread.start()
        
        full_response = ""

        for text in streamer:
            full_response += text
            yield text
            
        self.last_response = full_response

        thread.join()
    
    
    def get_response(self, prompt):
        self.messages.append({
            "role": "user",
            "content": prompt
        })

        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)


        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        text_response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text_response

    def stream_response(self, prompt):
        self.messages.append({
            "role": "user",
            "content": prompt
        })

        formatted = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
        )
        print("Starting stream with: ")
        print(repr(formatted))
        
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)


        streamer = IncrementalTextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_args = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 100,
            "do_sample": False,
            "tokenizer": self.tokenizer,
            "eos_token_id": [
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                self.tokenizer.eos_token_id,
            ],
            #"stop_strings": ["</RESPONSE>"],
        }

        thread = Thread(
            target=self.model.generate,
            kwargs=generation_args
        )

        thread.start()

        full_response = ""

        for text in streamer:
            full_response += text
            yield text

        thread.join()
        self.messages.append({
            "role": "assistant",
            "content": full_response
        })



