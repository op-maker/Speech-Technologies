from typing import List

import torch
from torch import nn


class QuartzNetBlock(torch.nn.Module):
    def __init__(
        self,
        feat_in: int,
        filters: int,
        repeat: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        residual: bool,
        separable: bool,
        dropout: float,
    ):

        super().__init__()
        
        if residual:
            self.res = nn.Sequential(
                nn.Conv1d(
                    feat_in,
                    filters,
                    kernel_size=1),
                nn.BatchNorm1d(filters)
                )
        else:
            self.res = None

        self.conv = nn.ModuleList()

        for i in range(repeat):
            self.conv += self.build_block(
                            separable,
                            feat_in,
                            filters,
                            kernel_size,
                            stride,
                            dilation,
                         )
            
            if i != repeat - 1 and residual:
                self.conv += [
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]

            feat_in = filters
        
        #self.conv = nn.ModuleList(self.conv)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def build_block(
        self,
        separable,
        feat_in,
        filters,
        kernel_size,
        stride,
        dilation,
    ):
        
        block = []
        padding = dilation * (kernel_size - 1) // 2
        
        if separable:
            block += [
                nn.Conv1d(
                        feat_in,
                        feat_in,
                        kernel_size,
                        groups=feat_in,
                        stride=stride,
                        dilation=dilation,
                        padding=padding,
                    ),
                nn.Conv1d(
                    feat_in,
                    filters,
                    kernel_size=1),
                nn.BatchNorm1d(filters)
            ]
            
        else:
            block += [
                nn.Conv1d(
                    feat_in,
                    filters,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                ) ,
                nn.BatchNorm1d(filters)
            ]
        
        
        return block #nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input_ = x
        for block in self.conv:
            x = block(x)
        
        if self.res is not None:
            x += self.res(input_)
            
        output = self.out(x)
        
        return output


class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride**block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.layers(features)
        encoded_len = (
            torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len
