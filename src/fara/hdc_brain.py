import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

class HyperNestedBrain:
    def __init__(self, model_path, dim=10000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        
        # --- 1. THE LAYERS (Unchanged) ---
        self.delta_memory = torch.zeros(dim, device=device)
        self.theta_timeline = torch.zeros(dim, device=device)
        self.alpha_inhibitions = torch.zeros(dim, device=device)

        # --- 2. ENCODERS (Gamma Layer - Native Vision & Text) ---
        print(f"Initializing Native Neuro-Symbolic Brain from: {model_path}")
        
        # A. Vision Processor (Handles image resizing/norm)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # B. Text Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # C. The Model Backbone
        # We load 'AutoModel' to get the raw hidden states without the Language Modeling head.
        # This acts as our universal encoder.
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            device_map=device
        )
        
        # Projections to scale model dims (e.g. 4096) to HDC dim (10000)
        self.proj_vision = None
        self.proj_text = None
        
        # Fixed Random Bases for binding concepts
        self.basis_action = torch.randn(dim, device=device).sign()
        self.basis_outcome = torch.randn(dim, device=device).sign()

    def _encode_image(self, pil_image):
        """Gamma Layer (Vision): Converts visual sensation to Hypervector"""
        with torch.no_grad():
            # 1. Process image
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            # 2. Forward pass through Vision Tower
            # Note: For Qwen-VL/Fara, we access the visual module specifically if available,
            # or rely on the model to handle pixel_values automatically.
            # Most HF VLMs expose 'vision_model' or handle pixel_values in the main forward.
            # We assume the model returns 'last_hidden_state'.
            if hasattr(self.model, "visual"):
                # Direct access to vision tower (faster)
                outputs = self.model.visual(inputs.pixel_values)
            else:
                # Fallback: General forward pass
                outputs = self.model(**inputs)
            
            # 3. Pooling: Mean of all visual patches -> Single Vector
            # Shape: [Batch, Patches, Hidden] -> [Batch, Hidden]
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs # Some vision towers return the tensor directly
                
            features = hidden.mean(dim=1)
            
            # 4. Projection
            if self.proj_vision is None:
                feat_dim = features.shape[1]
                self.proj_vision = torch.randn(feat_dim, self.dim, device=self.device).sign()

            hv = torch.matmul(features, self.proj_vision).sign().squeeze()
        return hv

    def _encode_text(self, text_str):
        """Gamma Layer (Text): Converts action description to Hypervector"""
        with torch.no_grad():
            # 1. Tokenize
            inputs = self.tokenizer(text_str, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # 2. Forward pass through Transformer Backbone
            # output_hidden_states=True ensures we get the embeddings
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # 3. Extract last layer hidden states
            # Shape: [Batch, Seq_Len, Hidden]
            last_hidden_state = outputs.last_hidden_state
            
            # 4. Pooling: Mean of all token embeddings
            # We use a simple mean pooling strategy for semantic representation
            features = last_hidden_state.mean(dim=1)
            
            # 5. Projection
            if self.proj_text is None:
                feat_dim = features.shape[1]
                self.proj_text = torch.randn(feat_dim, self.dim, device=self.device).sign()
                
            hv = torch.matmul(features, self.proj_text).sign().squeeze()
        return hv

    def _bind(self, u, v):
        return u * v

    def _bundle(self, u, v):
        return torch.sign(u + v)

    def recall(self, screenshot: Image.Image, current_task: str) -> str:
        state_hv = self._encode_image(screenshot)
        query_result = self._bind(self.delta_memory, state_hv)
        similarity = torch.cosine_similarity(query_result.unsqueeze(0), self.alpha_inhibitions.unsqueeze(0))
        
        if similarity.item() > 0.3: 
            return "[MEMORY WARNING]: You have been on this screen before and failed. Try a different strategy."
        return "" 

    def learn(self, screenshot: Image.Image, action_desc: str, outcome: str):
        """
        The Learning Step:
        1. Encode State (Vision Tower)
        2. Encode Action (Text Transformer)
        3. Bind & Bundle
        """
        state_hv = self._encode_image(screenshot)
        
        # --- NEW: Use the native text encoder ---
        action_hv = self._encode_text(action_desc)

        # Create the Experience Vector
        experience = self._bind(state_hv, action_hv)
        
        if "error" in outcome.lower() or "fail" in outcome.lower():
            self.alpha_inhibitions = self._bundle(self.alpha_inhibitions, experience)
        else:
            self.delta_memory = self._bundle(self.delta_memory, experience)
            
        self.theta_timeline = self._bundle(torch.roll(self.theta_timeline, 1), experience)
