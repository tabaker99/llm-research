#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:36:38 2026

@author: tbaker
"""


class CompletionEngine():
    def __init__(self):
        return
    
    def complete(self, prompt):
        return ""
    
    
class TestCompletionEngine(CompletionEngine):
    def __init__(self, completion_string="Test completion string."):
        self.completion_string = completion_string
        
    def complete(self, prompt):
        return self.completion_string
    