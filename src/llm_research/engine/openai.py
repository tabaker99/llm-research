#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:37:28 2026

@author: tbaker
"""


from openai import OpenAI

from .base import CompletionEngine

import logging
logger = logging.getLogger(__name__)

    
class OpenAICompletionEngine(CompletionEngine):
    dummy_response = "Dummy response for testing"
    
    def __init__(self, model="gpt-6-luna"):
        self.model = model
        self.client = OpenAI()
        self.testing_mode = True
        
    def enable_testing_mode(self):
        self.testing_mode = True
        
    def disable_testing_mode(self):
        self.testing_mode = False
        
    def complete(prompt):
        """
        Get a GPT response using the prompt. If global testing mode is enabled,
        a dummy response will be returned without calling the API.
    
        Parameters
        ----------
        prompt : str
            Prompt to send to GPT.
        model : str, optional
            GPT model to use for generation. The default is "gpt-6-luna".
    
        Returns
        -------
        output_text : str
            Text response from the API.
        """
        
        # For testing: send a filler response without using the API
        if self.testing_mode:
            return self.dummy_response
        
        # Get API response
        response = client.responses.create(
            model=self.model,
            input=prompt,
        )
        output_text = response.output_text
        return output_text