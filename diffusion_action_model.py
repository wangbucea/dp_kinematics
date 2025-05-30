import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
from tqdm import tqdm

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """残差块，用于U-Net架构"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 修改GroupNorm的组数，确保能被通道数整除
        # 使用更小的组数或者使用LayerNorm替代
        num_groups = min(8, in_channels)  # 确保组数不超过通道数
        while in_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
            
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels) if num_groups > 1 else nn.LayerNorm(in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout)
        )
        
        num_groups_out = min(8, out_channels)
        while out_channels % num_groups_out != 0 and num_groups_out > 1:
            num_groups_out -= 1
            
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups_out, out_channels) if num_groups_out > 1 else nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout)
        )
        
        self.res_conv = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb):
        # x shape: [batch_size, seq_len, channels]
        batch_size, seq_len, channels = x.shape
        
        # 对于GroupNorm，需要转换维度
        if isinstance(self.block1[0], nn.GroupNorm):
            # 转换为 [batch_size, channels, seq_len] 用于GroupNorm
            x_norm = x.transpose(1, 2)  # [batch_size, channels, seq_len]
            h = self.block1[0](x_norm).transpose(1, 2)  # 转回 [batch_size, seq_len, channels]
            h = self.block1[1:](h)  # 应用剩余层
        else:
            # LayerNorm可以直接应用
            h = self.block1(x)
            
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, None, :]
        
        if isinstance(self.block2[0], nn.GroupNorm):
            h_norm = h.transpose(1, 2)
            h = self.block2[0](h_norm).transpose(1, 2)
            h = self.block2[1:](h)
        else:
            h = self.block2(h)
            
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        # 使用LayerNorm替代GroupNorm以避免维度问题
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        
        # LayerNorm可以直接应用于最后一个维度
        h = self.norm(x)
        
        # 自注意力
        attn_out, _ = self.attention(h, h, h)
        
        return x + attn_out

class ConditionEncoder(nn.Module):
    """条件编码器：编码机器人状态和目标位置"""
    def __init__(self, joint_dim=7, position_dim=3, hidden_dim=512):
        super().__init__()
        # 输入维度：关节角度 + 关节位置 + 目标位置
        input_dim = joint_dim + joint_dim * position_dim + position_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, joint_angles, joint_positions, target_position):
        batch_size = joint_angles.size(0)
        joint_positions_flat = joint_positions.reshape(batch_size, -1)
        condition = torch.cat([joint_angles, joint_positions_flat, target_position], dim=1)
        return self.encoder(condition)

class DiffusionUNet(nn.Module):
    """扩散模型的U-Net架构"""
    def __init__(self, 
                 action_dim=7, 
                 seq_len=30, 
                 hidden_dims=[128, 256, 512, 1024],
                 condition_dim=512,
                 time_emb_dim=128,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        self.action_dim = action_dim
        self.seq_len = seq_len
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # 输入投影
        self.input_proj = nn.Linear(action_dim, hidden_dims[0])
        
        # 条件投影
        self.condition_proj = nn.Linear(condition_dim, hidden_dims[0])
        
        # 编码器（下采样）
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(
                ResidualBlock(in_dim, hidden_dim, time_emb_dim, dropout)
            )
            self.encoder_attns.append(
                AttentionBlock(hidden_dim, num_heads)
            )
            in_dim = hidden_dim
            
        # 中间层
        self.middle_block1 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim, dropout)
        self.middle_attn = AttentionBlock(hidden_dims[-1], num_heads)
        self.middle_block2 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim, dropout)
        
        # 解码器（上采样）
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        
        reversed_dims = list(reversed(hidden_dims))
        for i, hidden_dim in enumerate(reversed_dims):
            in_dim = hidden_dim * 2 if i > 0 else hidden_dim  # 跳跃连接
            out_dim = reversed_dims[i+1] if i < len(reversed_dims)-1 else hidden_dims[0]
            
            self.decoder_blocks.append(
                ResidualBlock(in_dim, out_dim, time_emb_dim, dropout)
            )
            self.decoder_attns.append(
                AttentionBlock(out_dim, num_heads)
            )
            
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims[0]),  # 使用LayerNorm替代GroupNorm
            nn.SiLU(),
            nn.Linear(hidden_dims[0], action_dim)
        )
        
    def forward(self, x, timesteps, condition):
        """
        Args:
            x: 噪声动作序列 [batch_size, seq_len, action_dim]
            timesteps: 时间步 [batch_size]
            condition: 条件编码 [batch_size, condition_dim]
        """
        # 时间嵌入
        time_emb = self.time_embed(timesteps)
        
        # 输入投影
        h = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # 添加条件信息
        condition_emb = self.condition_proj(condition).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        h = h + condition_emb
        
        # 编码器
        encoder_outputs = []
        for block, attn in zip(self.encoder_blocks, self.encoder_attns):
            h = block(h, time_emb)
            h = attn(h)
            encoder_outputs.append(h)
            
        # 中间层
        h = self.middle_block1(h, time_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, time_emb)
        
        # 解码器
        for i, (block, attn) in enumerate(zip(self.decoder_blocks, self.decoder_attns)):
            if i > 0:
                # 跳跃连接
                skip = encoder_outputs[-(i+1)]
                h = torch.cat([h, skip], dim=-1)
            h = block(h, time_emb)
            h = attn(h)
            
        # 输出投影
        return self.output_proj(h)

class DiffusionActionModel(nn.Module):
    """完整的扩散动作生成模型"""
    def __init__(self, 
                 action_dim=7,
                 seq_len=30,
                 joint_dim=7,
                 position_dim=3,
                 hidden_dims=[128, 256, 512, 1024],
                 condition_dim=512,
                 time_emb_dim=128,
                 num_heads=8,
                 dropout=0.1,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02):
        super().__init__()
        
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.num_timesteps = num_timesteps
        
        # 条件编码器
        self.condition_encoder = ConditionEncoder(joint_dim, position_dim, condition_dim)
        
        # U-Net去噪网络
        self.unet = DiffusionUNet(
            action_dim=action_dim,
            seq_len=seq_len,
            hidden_dims=hidden_dims,
            condition_dim=condition_dim,
            time_emb_dim=time_emb_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 扩散过程参数
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps, beta_start, beta_end))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
    def _cosine_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """余弦噪声调度"""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
        
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, joint_angles, joint_positions, target_position, t, noise=None):
        """计算扩散损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 前向扩散
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 编码条件
        condition = self.condition_encoder(joint_angles, joint_positions, target_position)
        
        # 预测噪声
        predicted_noise = self.unet(x_noisy, t, condition)
        
        # 计算损失
        loss = F.mse_loss(noise, predicted_noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, condition):
        """单步去噪采样"""
        betas_t = self.betas[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1)
        
        # 预测噪声
        predicted_noise = self.unet(x, t, condition)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, joint_angles, joint_positions, target_position, num_samples=1):
        """完整采样过程"""
        device = next(self.parameters()).device
        batch_size = joint_angles.size(0) if len(joint_angles.shape) > 1 else 1
        
        # 编码条件
        condition = self.condition_encoder(joint_angles, joint_positions, target_position)
        
        # 从纯噪声开始
        shape = (batch_size, self.seq_len, self.action_dim)
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, condition)
            
        return x
    
    def forward(self, joint_angles, joint_positions, target_position, action_sequence=None):
        """前向传播"""
        if self.training and action_sequence is not None:
            # 训练模式：计算损失
            batch_size = joint_angles.size(0)
            device = joint_angles.device
            
            # 随机采样时间步
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
            
            # 计算损失
            loss = self.p_losses(action_sequence, joint_angles, joint_positions, target_position, t)
            return loss
        else:
            # 推理模式：生成动作序列
            return self.sample(joint_angles, joint_positions, target_position)

# 训练函数
def train_diffusion_model(model, train_dataloader, val_dataloader=None, num_epochs=1000, lr=1e-4, device='cuda', 
                          patience=50, min_delta=1e-6, save_path='model_results/best_diffusion_model.pth'):
    """训练扩散模型，包含早停策略
    
    Args:
        model: 扩散模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        num_epochs: 最大训练轮数
        lr: 学习率
        device: 设备
        patience: 早停耐心值（验证损失不改善的最大轮数）
        min_delta: 最小改善阈值
        save_path: 最佳模型保存路径
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.to(device)
    
    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"开始训练，最大轮数: {num_epochs}, 早停耐心值: {patience}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for batch in tqdm(train_dataloader):
            joint_angles, joint_positions, target_positions, _, action_sequences = batch
            joint_angles = joint_angles.to(device)
            joint_positions = joint_positions.to(device)
            target_positions = target_positions.to(device)
            action_sequences = action_sequences.to(device)
            
            optimizer.zero_grad()
            
            # 计算损失
            loss = model(joint_angles, joint_positions, target_positions, action_sequences)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    joint_angles, joint_positions, target_positions, _, action_sequences = batch
                    joint_angles = joint_angles.to(device)
                    joint_positions = joint_positions.to(device)
                    target_positions = target_positions.to(device)
                    action_sequences = action_sequences.to(device)
                    
                    # 计算验证损失
                    loss = model(joint_angles, joint_positions, target_positions, action_sequences)
                    total_val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), save_path)
                print(f'Epoch [{epoch+1}/{num_epochs}] - 新的最佳模型已保存！验证损失: {avg_val_loss:.6f}')
            else:
                patience_counter += 1
            
            # 检查是否需要早停
            if patience_counter >= patience:
                print(f'早停触发！在第 {epoch+1} 轮停止训练，最佳验证损失: {best_val_loss:.6f}')
                break
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}, '
                      f'最佳验证损失: {best_val_loss:.6f}, 耐心计数: {patience_counter}/{patience}, LR: {scheduler.get_last_lr()[0]:.6f}')
        else:
            # 没有验证集时的输出
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], 训练损失: {avg_train_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        scheduler.step()
    
    # 加载最佳模型
    if val_dataloader is not None and best_val_loss < float('inf'):
        model.load_state_dict(torch.load(save_path))
        print(f"训练完成，已加载最佳模型（验证损失: {best_val_loss:.6f}）")
    
    return model, train_losses, val_losses
