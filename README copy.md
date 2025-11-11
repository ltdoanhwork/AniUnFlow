flowchart LR
  subgraph IN[Input]
    A[Clip: B×T×3×H×W]
    M[Masks (optional): B×T×S×H×W]
  end

  A --> ENC
  M -- optional --> SAM

  subgraph E[Encoder Pyramid (SIM-ready cache)]
    ENC[Encoder\n(Pyramid: 1/4,1/8,1/16)]
  end

  ENC --> FEAT1[(Lvl1: 1/4)]
  ENC --> FEAT2[(Lvl2: 1/8)]
  ENC --> FEAT3[(Lvl3: 1/16)]

  subgraph TKN[Cost Tokenizer]
    CT1[Local Corr -> Tokens @1/4]
    CT2[Local Corr -> Tokens @1/8]
    CT3[Local Corr -> Tokens @1/16]
  end

  FEAT1 --> CT1
  FEAT2 --> CT2
  FEAT3 --> CT3

  CT1 --> LCM
  CT2 --> LCM
  CT3 --> LCM

  subgraph S[SAM Adapter (optional)]
    SAM[Seg tokens + edge map @1/8]
  end

  SAM --> LCM

  subgraph MEM[Latent Cost Memory]
    LCM[Temporal Transformer\n(causal, token_dim=D)]
  end

  LCM --> GTR

  subgraph REG[Global Temporal Regressor]
    GTR[Temporal Aggregation\n-> coarse flows @1/8]
  end

  GTR --> DEC

  subgraph DECODER[Multi-Scale Recurrent Decoder]
    DEC[Refine @1/8 -> @1/4]
  end

  DEC --> FLOWS[(Flows per pair: B×2×H/4×W/4)]
  FEAT2 --> OCC
  subgraph OCCHEAD[Occlusion Head]
    OCC[Occ logits @1/8]
  end

  subgraph LOSS[Unsupervised Losses]
    L1[Photometric (Charbonnier/SSIM)]
    SM[Edge-aware Smoothness]
    TMP[Temporal Composition]
    CYC[Cycle Consistency]
    FB[Forward-Backward Consistency (option)]
  end

  FLOWS --> L1
  FLOWS --> SM
  FLOWS --> TMP
  FLOWS --> CYC
  FLOWS -. optional .-> FB

  classDef k fill:#eef,stroke:#335,stroke-width:1px
  classDef opt fill:#efe,stroke:#363,stroke-width:1px
  class ENC,CT1,CT2,CT3,LCM,GTR,DEC,OCC k
  class SAM,opt opt
