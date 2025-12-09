import torch
import clip
from PIL import Image
import numpy as np

class HyperNestedBrain:
    def __init__(self, dim=10000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        
        # --- 1. THE LAYERS ---
        # Delta Layer: Long-term Knowledge Graph (The "Soul")
        self.delta_memory = torch.zeros(dim, device=device)
        
        # Theta Layer: Episodic Sequence Buffer (The "Timeline")
        self.theta_timeline = torch.zeros(dim, device=device)
        
        # Alpha Layer: Inhibition Vectors (The "Filter")
        self.alpha_inhibitions = torch.zeros(dim, device=device)

        # --- 2. ENCODERS (Gamma Layer) ---
        # We use CLIP to turn Fara's screenshots into HDC-compatible vectors
        print(f"Initializing HyperNested Brain on {device}...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Fixed Random Bases for binding concepts
        self.basis_action = torch.randn(dim, device=device).sign()
        self.basis_outcome = torch.randn(dim, device=device).sign()

    def _encode_image(self, pil_image):
        """Gamma Layer: Instantly convert visual sensation to Hypervector"""
        with torch.no_grad():
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            # Encode and project to HDC dimension
            features = self.clip_model.encode_image(image_input)
            # Random projection to scale up to 10k dims if needed
            projection = torch.randn(features.shape[1], self.dim, device=self.device)
            hv = torch.matmul(features, projection).sign().squeeze()
        return hv

    def _bind(self, u, v):
        """HDC Binding Operation (XOR approximation via element-wise mult)"""
        return u * v

    def _bundle(self, u, v):
        """HDC Bundling Operation (Superposition)"""
        return torch.sign(u + v)

    def recall(self, screenshot: Image.Image, current_task: str) -> str:
        """
        The Recall Step:
        1. Encode current state (Gamma).
        2. Unbind from Delta Memory to find similar past situations.
        3. If a match is found, decode the "outcome" and return a text hint.
        """
        # 1. Perception
        state_hv = self._encode_image(screenshot)
        
        # 2. Query: "What happened last time I was in this state?"
        # We unbind the state from the Delta Memory
        # query_result = Delta * State
        query_result = self._bind(self.delta_memory, state_hv)
        
        # 3. Similarity Check (Cosine Similarity with known outcome bases)
        # In a real impl, we would store "Outcome Key-Value Pairs". 
        # Here we simplify: if similarity to 'basis_action' is high, we recall success.
        
        # Check against Inhibitions (Alpha Layer)
        similarity = torch.cosine_similarity(query_result.unsqueeze(0), self.alpha_inhibitions.unsqueeze(0))
        
        if similarity.item() > 0.3: # Threshold
            return "[MEMORY WARNING]: You have been on this screen before and failed. Try a different strategy."
        
        return "" # No relevant memory found

    def learn(self, screenshot: Image.Image, action_desc: str, outcome: str):
        """
        The Learning Step:
        1. Create an episodic vector: (State * Action * Outcome).
        2. Bundle into Theta (Timeline).
        3. Bundle into Delta (Long-term).
        """
        state_hv = self._encode_image(screenshot)
        
        # Bind the sequence: State + Action -> Outcome
        # Ideally we encode 'action_desc' using CLIP text encoder too
        with torch.no_grad():
            text = clip.tokenize([action_desc]).to(self.device)
            action_hv = self.clip_model.encode_text(text)
            proj = torch.randn(action_hv.shape[1], self.dim, device=self.device)
            action_hv = torch.matmul(action_hv, proj).sign().squeeze()

        # Create the Experience Vector
        experience = self._bind(state_hv, action_hv)
        
        # If outcome was bad, add to Inhibition (Alpha)
        if "error" in outcome.lower() or "fail" in outcome.lower():
            self.alpha_inhibitions = self._bundle(self.alpha_inhibitions, experience)
        else:
            # Add to Wisdom (Delta)
            self.delta_memory = self._bundle(self.delta_memory, experience)
            
        # Update Timeline (Theta) - Permute current timeline and add new experience
        self.theta_timeline = self._bundle(torch.roll(self.theta_timeline, 1), experience)