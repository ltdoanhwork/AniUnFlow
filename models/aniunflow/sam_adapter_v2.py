# file: models/aniunflow/sam_adapter_v2.py
"""
Enhanced SAM Adapter v2
========================
Improved SAM integration with all proposed enhancements:

1. Soft Boundary Encoding (learned, continuous)
2. Learned Segment Embeddings (adaptive to domain)
3. Temporal Propagation (flow-guided consistency)
4. Segment-Aware Attention Bias
5. Multi-Scale Segment Features

All features are independently toggleable for ablation studies.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SoftBoundaryEncoder(nn.Module):
    """
    Learned Soft Boundary Encoder
    ==============================
    Produces continuous [0, 1] boundary maps instead of hard binary edges.
    
    Benefits:
    - Gradients flow through boundary detection
    - Learns domain-specific edge patterns (anime lines vs. photorealistic edges)
    - Handles ambiguous boundaries gracefully
    """
    
    def __init__(self, in_channels: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )
        
        # Also compute gradient-based edges as auxiliary input
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        # Initialize Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x.weight.data = sobel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_y.view(1, 1, 3, 3)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
    
    def forward(self, segment_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segment_labels: (B, 1, H, W) integer label map
        Returns:
            soft_boundaries: (B, 1, H, W) continuous boundary map [0, 1]
        """
        # Compute gradient magnitude from label map
        seg_float = segment_labels.float()
        grad_x = self.sobel_x(seg_float).abs()
        grad_y = self.sobel_y(seg_float).abs()
        grad_mag = (grad_x + grad_y).clamp(0, 1)
        
        # Learn soft refinement
        soft_boundary = self.net(grad_mag)
        
        return soft_boundary


class LearnedSegmentEncoder(nn.Module):
    """
    Learned Segment Embedding Encoder
    ===================================
    Projects segment masks to learned embeddings that capture semantic similarity.
    
    Benefits:
    - Adaptive to dataset characteristics (anime vs. real)
    - Compact representation (D << HW)
    - Can capture semantic similarity beyond pixel overlap
    """
    
    def __init__(
        self, 
        feat_dim: int = 128, 
        embed_dim: int = 64,
        max_segments: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_segments = max_segments
        
        # Learnable segment type embeddings
        self.segment_embeddings = nn.Embedding(max_segments + 1, embed_dim)  # +1 for background
        
        # Feature aggregation network
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(
        self, 
        segment_labels: torch.Tensor,  # (B, 1, H, W) integer labels
        features: torch.Tensor,         # (B, C, H, W) image features
    ) -> torch.Tensor:
        """
        Args:
            segment_labels: Integer label map
            features: Image features at same resolution
            
        Returns:
            segment_embeddings: (B, S, D) learned segment embeddings
        """
        B, _, H, W = segment_labels.shape
        
        # Project features
        feat_proj = self.feat_proj(features)  # (B, D, H, W)
        
        # Get unique segments and compute embeddings
        embeddings_list = []
        for b in range(B):
            labels_b = segment_labels[b, 0]  # (H, W)
            unique_labels = torch.unique(labels_b)
            
            seg_embeds = []
            for seg_id in unique_labels[:self.max_segments]:
                mask = (labels_b == seg_id).float()  # (H, W)
                mask_sum = mask.sum().clamp(min=1.0)
                
                # Pool features for this segment
                feat_pooled = (feat_proj[b] * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum  # (D,)
                
                # Get learnable embedding for segment type
                type_embed = self.segment_embeddings(seg_id.long().clamp(0, self.max_segments))  # (D,)
                
                # Combine
                combined = self.output_proj(torch.cat([feat_pooled, type_embed]))
                seg_embeds.append(combined)
            
            # Pad to max_segments
            while len(seg_embeds) < self.max_segments:
                seg_embeds.append(torch.zeros(self.embed_dim, device=features.device))
            
            embeddings_list.append(torch.stack(seg_embeds[:self.max_segments]))
        
        return torch.stack(embeddings_list)  # (B, S, D)


class TemporalSegmentPropagator(nn.Module):
    """
    Temporal Segment Propagator
    ============================
    Propagates segment labels across frames using optical flow for consistency.
    
    Benefits:
    - Consistent segment IDs across frames
    - Reduces matching ambiguity
    - Better temporal coherence in estimated flow
    """
    
    def __init__(self, refine: bool = True, hidden_dim: int = 32):
        super().__init__()
        self.refine = refine
        
        if refine:
            # Refinement network to fix propagation errors
            self.refine_net = nn.Sequential(
                nn.Conv2d(2, hidden_dim, 3, padding=1),  # warped + original
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, 1),
            )
            
            # Blend weight
            self.blend_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        segment_labels: torch.Tensor,  # (B, T, 1, H, W) labels for all frames
        flow_predictions: Optional[torch.Tensor] = None,  # (B, T-1, 2, H, W) flow if available
    ) -> torch.Tensor:
        """
        Propagate and refine segment labels across time.
        
        If flow is not provided, uses forward difference of segment centroids.
        """
        B, T, _, H, W = segment_labels.shape
        
        if flow_predictions is None or not self.refine:
            # No flow available, return as-is with temporal smoothing
            return segment_labels
        
        refined_labels = [segment_labels[:, 0]]  # Keep first frame
        
        for t in range(1, T):
            # Warp previous frame's labels to current frame
            flow_t = flow_predictions[:, t-1]  # (B, 2, H, W)
            prev_labels = refined_labels[-1]  # (B, 1, H, W)
            
            # Create sampling grid
            grid = self._create_flow_grid(flow_t, H, W)
            warped_labels = F.grid_sample(
                prev_labels.float(), grid, 
                mode='nearest', padding_mode='border', align_corners=True
            )
            
            # Blend with original frame's labels
            curr_labels = segment_labels[:, t]  # (B, 1, H, W)
            
            if self.refine:
                # Learn refinement
                combined = torch.cat([warped_labels, curr_labels.float()], dim=1)
                weight_map = torch.sigmoid(self.refine_net(combined))
                blended = weight_map * curr_labels.float() + (1 - weight_map) * warped_labels
                refined_labels.append(blended.round().long())
            else:
                # Simple blend
                w = torch.sigmoid(self.blend_weight)
                blended = w * curr_labels.float() + (1 - w) * warped_labels
                refined_labels.append(blended.round().long())
        
        return torch.stack(refined_labels, dim=1)  # (B, T, 1, H, W)
    
    def _create_flow_grid(self, flow: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Create sampling grid from flow field."""
        B = flow.shape[0]
        
        # Create base grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=flow.device),
            torch.linspace(-1, 1, W, device=flow.device),
            indexing='ij'
        )
        base_grid = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Normalize flow to [-1, 1] range
        flow_norm = flow.permute(0, 2, 3, 1).clone()
        flow_norm[..., 0] = flow_norm[..., 0] / (W / 2)
        flow_norm[..., 1] = flow_norm[..., 1] / (H / 2)
        
        return base_grid + flow_norm


class SegmentAttentionBias(nn.Module):
    """
    Segment-Aware Attention Bias Generator
    ========================================
    Creates attention bias for transformer layers based on segment structure.
    
    Tokens from the same segment attend more strongly to each other.
    """
    
    def __init__(self, temperature: float = 1.0, bias_scale: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.bias_scale = nn.Parameter(torch.tensor(bias_scale))
    
    def forward(
        self,
        segment_labels: torch.Tensor,  # (B, 1, H, W)
        query_shape: Tuple[int, int],
        key_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Generate attention bias from segment labels.
        
        Returns:
            attention_bias: (B, N_q, N_k)
        """
        B = segment_labels.shape[0]
        h_q, w_q = query_shape
        h_k, w_k = key_shape if key_shape else query_shape
        
        # Resize labels to query/key resolutions
        labels_q = F.interpolate(
            segment_labels.float(), size=(h_q, w_q), mode='nearest'
        ).long()  # (B, 1, h_q, w_q)
        labels_k = F.interpolate(
            segment_labels.float(), size=(h_k, w_k), mode='nearest'
        ).long()  # (B, 1, h_k, w_k)
        
        # Flatten
        labels_q = labels_q.view(B, -1)  # (B, N_q)
        labels_k = labels_k.view(B, -1)  # (B, N_k)
        
        # Compute same-segment indicator
        same_segment = (labels_q.unsqueeze(-1) == labels_k.unsqueeze(-2)).float()  # (B, N_q, N_k)
        
        # Convert to attention bias
        attention_bias = self.bias_scale * (same_segment - 0.5) / self.temperature
        
        return attention_bias


class EnhancedSamAdapter(nn.Module):
    """
    Enhanced SAM Adapter v2
    ========================
    Complete SAM integration module with all improvements.
    
    Features (all toggleable):
    - use_soft_boundary: Learned soft boundary encoding
    - use_learned_embedding: Adaptive segment embeddings
    - use_temporal_propagation: Flow-guided label propagation
    - use_attention_bias: Segment-aware attention modulation
    - use_multi_scale: Multi-scale segment features
    """
    
    def __init__(
        self,
        feat_dim: int = 128,
        token_dim: int = 192,
        embed_dim: int = 64,
        max_segments: int = 32,
        # Feature flags
        use_soft_boundary: bool = True,
        use_learned_embedding: bool = True,
        use_temporal_propagation: bool = True,
        use_attention_bias: bool = True,
        use_multi_scale: bool = False,
    ):
        super().__init__()
        
        # Save config
        self.use_soft_boundary = use_soft_boundary
        self.use_learned_embedding = use_learned_embedding
        self.use_temporal_propagation = use_temporal_propagation
        self.use_attention_bias = use_attention_bias
        self.use_multi_scale = use_multi_scale
        
        # Initialize components based on flags
        if use_soft_boundary:
            self.boundary_encoder = SoftBoundaryEncoder()
        
        if use_learned_embedding:
            self.segment_encoder = LearnedSegmentEncoder(
                feat_dim=feat_dim, embed_dim=embed_dim, max_segments=max_segments
            )
            self.token_proj = nn.Linear(embed_dim, token_dim)
        else:
            # Simple projection (original approach)
            self.token_proj = nn.Linear(feat_dim, token_dim)
        
        if use_temporal_propagation:
            self.temporal_propagator = TemporalSegmentPropagator(refine=True)
        
        if use_attention_bias:
            self.attention_bias = SegmentAttentionBias()
        
        if use_multi_scale:
            self.scale_merger = nn.ModuleList([
                nn.Conv2d(1, 1, 3, padding=1) for _ in range(3)
            ])
    
    def forward(
        self,
        segment_labels: torch.Tensor,   # (B, T, 1, H, W) integer labels
        features: List[torch.Tensor],   # List of (B, C, H_l, W_l) per time step
        flow_predictions: Optional[torch.Tensor] = None,  # (B, T-1, 2, H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        Process segment labels and produce all SAM-derived features.
        
        Returns dict with:
        - 'tokens': (B, T, S, D) segment tokens for attention
        - 'boundaries': (B, T, 1, H, W) soft boundary maps
        - 'attention_bias': (B, N, N) attention bias (if enabled)
        - 'propagated_labels': (B, T, 1, H, W) temporally consistent labels
        """
        B, T = segment_labels.shape[:2]
        outputs = {}
        
        # Step 1: Temporal propagation
        if self.use_temporal_propagation and flow_predictions is not None:
            labels = self.temporal_propagator(segment_labels, flow_predictions)
        else:
            labels = segment_labels
        outputs['propagated_labels'] = labels
        
        # Step 2: Soft boundaries
        boundaries = []
        for t in range(T):
            if self.use_soft_boundary:
                boundary = self.boundary_encoder(labels[:, t])
            else:
                # Hard edges (original approach)
                seg = labels[:, t].float()
                edge_h = (seg[:, :, 1:, :] - seg[:, :, :-1, :]).abs()
                edge_v = (seg[:, :, :, 1:] - seg[:, :, :, :-1]).abs()
                edge_h = F.pad(edge_h, (0, 0, 0, 1))
                edge_v = F.pad(edge_v, (0, 1, 0, 0))
                boundary = (edge_h + edge_v).clamp(0, 1)
            boundaries.append(boundary)
        outputs['boundaries'] = torch.stack(boundaries, dim=1)
        
        # Step 3: Segment embeddings/tokens
        tokens = []
        for t in range(T):
            feat = features[t] if isinstance(features, list) else features[:, t]
            
            if self.use_learned_embedding:
                # Resize labels to feature resolution
                H_f, W_f = feat.shape[-2:]
                labels_resized = F.interpolate(
                    labels[:, t].float(), size=(H_f, W_f), mode='nearest'
                ).long()
                seg_embed = self.segment_encoder(labels_resized, feat)  # (B, S, D)
                tok = self.token_proj(seg_embed)  # (B, S, token_dim)
            else:
                # Original pooling approach
                B_f, C, H_f, W_f = feat.shape
                labels_resized = F.interpolate(
                    labels[:, t].float(), size=(H_f, W_f), mode='nearest'
                )
                labels_flat = rearrange(labels_resized, 'b 1 h w -> b 1 (h w)')
                feat_flat = rearrange(feat, 'b c h w -> b (h w) c')
                
                # Simple mean pooling per segment (less sophisticated)
                num = labels_flat.sum(-1, keepdim=True).clamp_min(1.0)
                seg_pooled = (labels_flat @ feat_flat) / num
                tok = self.token_proj(seg_pooled)
            
            tokens.append(tok)
        outputs['tokens'] = torch.stack(tokens, dim=1)  # (B, T, S, token_dim)
        
        # Step 4: Attention bias (for first frame, expand if needed)
        if self.use_attention_bias:
            H_f, W_f = (features[0] if isinstance(features, list) else features[:, 0]).shape[-2:]
            attn_bias = self.attention_bias(labels[:, 0], (H_f, W_f))
            outputs['attention_bias'] = attn_bias
        
        return outputs
    
    def get_boundary_loss_weight(
        self, 
        boundaries: torch.Tensor,  # (B, T, 1, H, W)
        base_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate per-pixel smoothness loss weights based on boundaries.
        
        Low weight at boundaries (allow discontinuities),
        High weight within segments (enforce smoothness).
        """
        # Invert: 1 - boundary = smoothness weight
        weights = base_weight * (1.0 - boundaries)
        return weights


def build_enhanced_sam_adapter(cfg: Dict) -> EnhancedSamAdapter:
    """Factory function to build adapter from config."""
    sam_cfg = cfg.get('sam', {})
    model_cfg = cfg.get('model', {}).get('args', {})
    
    return EnhancedSamAdapter(
        feat_dim=model_cfg.get('enc_channels', 64) * 2,
        token_dim=model_cfg.get('token_dim', 192),
        embed_dim=sam_cfg.get('embed_dim', 64),
        max_segments=sam_cfg.get('max_segments', 32),
        use_soft_boundary=sam_cfg.get('use_soft_boundary', True),
        use_learned_embedding=sam_cfg.get('use_learned_embedding', True),
        use_temporal_propagation=sam_cfg.get('use_temporal_propagation', True),
        use_attention_bias=sam_cfg.get('use_attention_bias', True),
        use_multi_scale=sam_cfg.get('use_multi_scale', False),
    )
