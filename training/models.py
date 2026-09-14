"""源自仓库 data(2).zip 的 models.py（92a2f05）。
保留原Swin窗口/移位窗口注意力与Patch Merging；只开放MLP比例并校验配置。
原CNN/ViT保留作源码参照，新训练入口只允许Swin。
"""
import torch
import torch.nn as nn


MODEL_NAMES = ("cnn", "swin1", "swin2", "vit")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError("Expected input shape (B, 1, 32, 32)")
        return self.head(self.features(x))


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


def window_partition(x, window_size):
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, channels)


def window_reverse(windows, window_size, height, width):
    batch = windows.shape[0] // ((height // window_size) * (width // window_size))
    channels = windows.shape[-1]
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, channels)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attention_dropout=0.0, projection_dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)
        relative_size = (2 * window_size - 1) ** 2
        self.relative_position_bias = nn.Parameter(torch.zeros(relative_size, num_heads))
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords = coords.flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_windows, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        bias = self.relative_position_bias[self.relative_position_index.reshape(-1)]
        bias = bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = self.attention_dropout(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        out = self.projection_dropout(self.projection(x))
        if return_attention:
            return out, attention
        return out


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        resolution,
        num_heads,
        window_size=4,
        shift_size=0,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.window_size = min(window_size, *resolution)
        self.shift_size = 0 if min(resolution) <= self.window_size else shift_size
        if self.shift_size >= self.window_size:
            raise ValueError("Shift size must be smaller than window size")
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(
            dim,
            self.window_size,
            num_heads,
            attention_dropout=dropout,
            projection_dropout=dropout,
        )
        self.path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.path2 = DropPath(drop_path)
        self.register_buffer("attention_mask", self._make_mask(), persistent=False)

    def _make_mask(self):
        if self.shift_size == 0:
            return None
        height, width = self.resolution
        labels = torch.zeros((1, height, width, 1))
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        index = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                labels[:, height_slice, width_slice, :] = index
                index += 1
        windows = window_partition(labels, self.window_size).squeeze(-1)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x, return_attention=False):
        height, width = self.resolution
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError("Token count does not match the configured resolution")
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size)
        if return_attention:
            windows, attn = self.attention(windows, self.attention_mask, return_attention=True)
        else:
            windows = self.attention(windows, self.attention_mask)
        x = window_reverse(windows, self.window_size, height, width)
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(batch, tokens, channels)
        x = shortcut + self.path1(x)
        x = x + self.path2(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, resolution):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        height, width = self.resolution
        batch, tokens, channels = x.shape
        if height % 2 or width % 2 or tokens != height * width:
            raise ValueError("Patch merging requires an even spatial resolution")
        x = x.view(batch, height, width, channels)
        x = torch.cat(
            (
                x[:, 0::2, 0::2],
                x[:, 1::2, 0::2],
                x[:, 0::2, 1::2],
                x[:, 1::2, 1::2],
            ),
            dim=-1,
        )
        x = x.view(batch, -1, channels * 4)
        return self.reduction(self.norm(x))


class SwinClassifier(nn.Module):
    def __init__(
        self,
        num_classes=9,
        embed_dim=32,
        depths=(2, 2, 2),
        num_heads=(2, 4, 8),
        dropout=0.0,
        drop_path=0.1,
        mlp_ratio=4.0,
    ):
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError("Depth and head configurations must have equal length")
        if len(depths) != 3 or any(d < 1 for d in depths):
            raise ValueError("This review version requires three non-empty stages")
        if embed_dim < 1 or mlp_ratio <= 0 or int(embed_dim * mlp_ratio) < 1:
            raise ValueError("Invalid channel count or MLP ratio")
        if any(h < 1 or (embed_dim * 2**i) % h for i, h in enumerate(num_heads)):
            raise ValueError("Each stage channel count must be divisible by its head count")
        if not (0 <= dropout < 1 and 0 <= drop_path < 1):
            raise ValueError("Drop probabilities must be in [0, 1)")
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=2, stride=2)
        self.patch_norm = nn.LayerNorm(embed_dim)
        total_depth = sum(depths)
        path_rates = torch.linspace(0, drop_path, total_depth).tolist()
        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        resolution = (16, 16)
        dim = embed_dim
        path_index = 0
        for stage_index, depth in enumerate(depths):
            blocks = []
            for block_index in range(depth):
                blocks.append(
                    SwinBlock(
                        dim=dim,
                        resolution=resolution,
                        num_heads=num_heads[stage_index],
                        window_size=4,
                        shift_size=0 if block_index % 2 == 0 else 2,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        drop_path=path_rates[path_index],
                    )
                )
                path_index += 1
            self.stages.append(nn.Sequential(*blocks))
            if stage_index < len(depths) - 1:
                self.mergers.append(PatchMerging(dim, resolution))
                resolution = (resolution[0] // 2, resolution[1] // 2)
                dim *= 2
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError("Expected input shape (B, 1, 32, 32)")
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.patch_norm(x)
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            if stage_index < len(self.mergers):
                x = self.mergers[stage_index](x)
        return self.head(self.norm(x).mean(dim=1))

    def forward_attention(self, x):
        """返回 (logits, attention_map)；attention_map 来自最后一个 Swin block（全局 16 token、8 头）。"""
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError("Expected input shape (B, 1, 32, 32)")
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.patch_norm(x)
        attention_map = None
        for stage_index, stage in enumerate(self.stages):
            blocks = list(stage.children())
            for block_index, block in enumerate(blocks):
                is_last = (stage_index == len(self.stages) - 1) and (block_index == len(blocks) - 1)
                if is_last:
                    x, attention_map = block(x, return_attention=True)
                else:
                    x = block(x)
            if stage_index < len(self.mergers):
                x = self.mergers[stage_index](x)
        logits = self.head(self.norm(x).mean(dim=1))
        return logits, attention_map


class ViTClassifier(nn.Module):
    def __init__(
        self,
        num_classes=9,
        embed_dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=4, stride=4)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, 65, embed_dim))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError("Expected input shape (B, 1, 32, 32)")
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((token, x), dim=1)
        x = self.dropout(x + self.position)
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


def build_model(name, num_classes=9):
    if name == "cnn":
        return CNNClassifier(num_classes=num_classes)
    if name == "swin1":
        return SwinClassifier(
            num_classes=num_classes,
            embed_dim=32,
            depths=(2, 2, 2),
            num_heads=(2, 4, 8),
            dropout=0.0,
            drop_path=0.1,
        )
    if name == "swin2":
        return SwinClassifier(
            num_classes=num_classes,
            embed_dim=48,
            depths=(2, 2, 4),
            num_heads=(3, 6, 12),
            dropout=0.05,
            drop_path=0.15,
        )
    if name == "vit":
        return ViTClassifier(num_classes=num_classes)
    choices = ", ".join(MODEL_NAMES)
    raise ValueError(f"Unknown model '{name}'. Available models: {choices}")
