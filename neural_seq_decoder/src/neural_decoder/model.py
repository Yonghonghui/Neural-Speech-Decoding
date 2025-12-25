import math
import torch
from torch import nn
from .augmentations import GaussianSmoothing


# update transformer #
# --- [Helper Class] Positional Encoding ---
# Transformers process all time steps in parallel and have no inherent concept of "order".
# We must inject positional information so the model knows which signal came first.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of [max_len, d_model] representing positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Use sine and cosine functions of different frequencies
        # This allows the model to learn to attend to relative positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer (part of state_dict but not a trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Add positional encoding to the input embedding
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# --- [Main Architecture] Transformer Decoder ---
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,   # Used as d_model for Transformer
        layer_dim,    # Number of Transformer layers (e.g., 2 or 3)
        nDays=24,
        dropout=0.2,  # Transformers typically require higher dropout (0.2-0.3)
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False, # Must be False for real-time (handled via masking)
    ):
        super(TransformerDecoder, self).__init__()

        # --- Hyperparameters ---
        self.hidden_dim = hidden_dim
        self.device = device
        self.kernelLen = kernelLen
        self.strideLen = strideLen
        self.gaussianSmoothWidth = gaussianSmoothWidth

        # --- Preprocessing (Standard Baseline Logic) ---
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        
        # Initialize day weights to identity for stability
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # Day-specific input affine layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))
            getattr(self, "inpLayer" + str(x)).weight.data += torch.eye(neural_dim)

        # --- [Transformer Core] ---
        
        # 1. Input Projection: Map flattened neural features to Transformer dimension (d_model)
        # Input size = channels * temporal_kernel_size
        input_feature_dim = neural_dim * kernelLen
        self.input_projection = nn.Linear(input_feature_dim, hidden_dim)
        
        # 2. Positional Encoding: Adds timing information to the projected features
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # 3. Transformer Encoder Layers
        # batch_first=True ensures input format is [Batch, Time, Feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,                # 4 Attention heads (hidden_dim must be divisible by 4)
            dim_feedforward=hidden_dim * 4, # Standard expansion ratio
            dropout=dropout,
            activation='gelu',      # GELU is smoother than ReLU, generally better for Transformers
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layer_dim)

        # --- [Post-Processing Stack] (Linderman Lab Approach) ---
        # We keep this stack as it proved effective in Exp 2/3/4
        self.post_transformer_stack = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )

        # Final Classification Head
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        # --- 1. Signal Preprocessing ---
        # Gaussian smoothing
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # Day-specific adaptation (calibrating signals across days)
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum("btd,bdk->btk", neuralInput, dayWeights) + \
                            torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # --- 2. Windowing (Unfold) ---
        # Convert continuous signal into sliding windows
        # Output shape: [batch, time_steps, flattened_features]
        stridedInputs = self.unfolder(torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3))
        stridedInputs = torch.permute(stridedInputs, (0, 2, 1))

        # --- 3. Transformer Logic ---
        
        # A. Project to hidden dimension
        src = self.input_projection(stridedInputs) 
        
        # Scale inputs (important trick for Transformer stability)
        src = src * math.sqrt(self.hidden_dim)     
        
        # Add positional info
        src = self.pos_encoder(src)

        # [cite_start]B. Generate Causal Mask (CRITICAL for Real-Time Requirement) [cite: 72]
        # This mask ensures that at time t, the model can only see inputs from 0 to t.
        # It creates a lower-triangular matrix of -inf and 0.
        seq_len = src.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)
        
        # C. Pass through Transformer Encoder with the mask
        # Output shape: [batch, time_steps, hidden_dim]
        output = self.transformer_encoder(src, mask=mask)

        # --- 4. Post-processing & Output ---
        output = self.post_transformer_stack(output)
        seq_out = self.fc_decoder_out(output)
        
        return seq_out


# update architecture: LSTM # 
class LSTMDecoder(nn.Module):
    """
    LSTM-based Decoder with 'Linderman Lab' architectural improvements.
    Replaces the standard GRU with an LSTM to potentially capture longer-term dependencies.
    """
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(LSTMDecoder, self).__init__()

        # --- Hyperparameters ---
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        # --- Preprocessing Layers ---
        # Nonlinearity applied after day-specific mixing
        self.inputLayerNonlinearity = torch.nn.Softsign()
        
        # Unfolder to create temporal windows (sliding window)
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        
        # Gaussian smoothing to reduce high-frequency noise in neural data
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        
        # Day-specific adaptation parameters (matrices and biases)
        # Allows the model to handle non-stationary signals across different recording days
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        # Initialize day weights to identity matrix for stability
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # --- [CORE ARCHITECTURE CHANGE] LSTM Layers ---
        # Replaced nn.GRU with nn.LSTM
        self.lstm_decoder = nn.LSTM(
            (neural_dim) * self.kernelLen, # Input size (flattened window)
            hidden_dim,                    # Hidden state size
            layer_dim,                     # Number of layers (e.g., 5)
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        # Initialize weights specifically for LSTM gates
        for name, param in self.lstm_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # --- Input Adaptation Layers ---
        # Day-specific affine transformations
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # --- [POST-RNN STACK] Linderman Lab Architecture ---
        # This stack increases model capacity and stability.
        # It processes the output of the LSTM before classification.
        
        # Determine dimension: hidden_dim (if uni-directional) or 2*hidden_dim (if bi-directional)
        lstm_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        
        self.post_lstm_stack = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim), # Project to hidden space
            nn.LayerNorm(hidden_dim),               # Normalization (prevents gradient explosion)
            nn.Dropout(self.dropout),               # Regularization
            nn.GELU()                               # Non-linearity (smoother than ReLU)
        )

        # --- Final Output Layer ---
        # Projects processed features to class logits (Phonemes + Blank)
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        # --- Preprocessing Step ---
        # Permute for Gaussian smoothing: [batch, time, channels] -> [batch, channels, time]
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # Apply day-specific adaptation weights
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # Unfold/Stride: Convert continuous time to sliding windows
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # --- [CORE ARCHITECTURE CHANGE] LSTM Forward Pass ---
        # Initialize Hidden State (h0) and Cell State (c0)
        # LSTM requires both, unlike GRU which only needs h0
        
        num_directions = 2 if self.bidirectional else 1
        
        # h0: Hidden state
        h0 = torch.zeros(
            self.layer_dim * num_directions,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()
        
        # c0: Cell state (Unique to LSTM)
        c0 = torch.zeros(
            self.layer_dim * num_directions,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()

        # Forward pass through LSTM
        # LSTM returns (output, (h_n, c_n))
        # We only need the full output sequence 'lstm_out'
        lstm_out, _ = self.lstm_decoder(stridedInputs, (h0.detach(), c0.detach()))

        # --- Apply Post-LSTM Stack ---
        feat = self.post_lstm_stack(lstm_out)
        
        # --- Final Classification ---
        seq_out = self.fc_decoder_out(feat)
        
        return seq_out

class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        
        # -----------------------------------------------------------------------
        # [BASELINE CODE] 
        # -----------------------------------------------------------------------
        # if self.bidirectional:
        #     self.fc_decoder_out = nn.Linear(
        #         hidden_dim * 2, n_classes + 1
        #     )  # +1 for CTC blank
        # else:
        #     self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank
        
        # -----------------------------------------------------------------------
        # [EXPERIMENT 2: Linderman Lab Architecture]
        # Adding a stack of Linear -> LayerNorm -> Dropout -> GELU after GRU
        # -----------------------------------------------------------------------
        gru_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        
        self.post_gru_stack = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),        # Normalization for training stability
            nn.Dropout(self.dropout),        # Regularization to prevent overfitting
            nn.GELU()                        # Non-linearity
        )

        # Final projection layer (input is now hidden_dim from the stack)
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)
        # -----------------------------------------------------------------------

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        
        # -----------------------------------------------------------------------
        # [BASELINE CODE] - Commented out for Experiment 2
        # -----------------------------------------------------------------------
        # seq_out = self.fc_decoder_out(hid)
        
        # -----------------------------------------------------------------------
        # [EXPERIMENT 2: Linderman Lab Architecture]
        # Pass GRU output through the new stack before final classification
        # -----------------------------------------------------------------------
        hid = self.post_gru_stack(hid)
        seq_out = self.fc_decoder_out(hid)
        # -----------------------------------------------------------------------
        
        return seq_out