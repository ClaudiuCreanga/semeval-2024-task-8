# flake8: noqa: W503
import torch
import torch.nn as nn


class StatisticsCNN1D(nn.Module):
    def __init__(
        self,
        input_size: int,
        out_size: int = 1,
        dropout_p: float = 0.5,
        conv_blocks: [dict] = [],
        fc: [int] = [],
        out_activation: str | None = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.conv_blocks = conv_blocks
        self.fc = fc

        self.conv1d = nn.Sequential()
        for i, conv_block_setting in enumerate(conv_blocks):
            conv1d_settings = conv_block_setting["conv1d"]
            max_pool1d_settings = conv_block_setting["max_pool_1d"]

            self.conv1d.add_module(f"conv1d_{i}", nn.Conv1d(**conv1d_settings))

            if "batch_norm_1d" in conv_block_setting:
                num_features = conv1d_settings["out_channels"]
                batch_norm_1d_settings = conv_block_setting["batch_norm_1d"]

                self.conv1d.add_module(
                    f"batchnorm_{i}",
                    nn.BatchNorm1d(
                        num_features=num_features,
                        **batch_norm_1d_settings
                    )
                )

            self.conv1d.add_module(f"relu_{i}", nn.ReLU())
            self.conv1d.add_module(f"dropout_{i}", nn.Dropout(dropout_p))
            self.conv1d.add_module(f"maxpool_{i}", nn.MaxPool1d(**max_pool1d_settings))

        self.conv1d.add_module("flatten", nn.Flatten())

        conv1d_out_channels = self.__get_conv_block_out_channels()
        conv1d_out_size = self.__get_conv_block_out_size()

        self.out = nn.Sequential()
        self.out.add_module(
            "fc",
            nn.Linear(
                in_features=conv1d_out_channels * conv1d_out_size,
                out_features=out_size,
            ),
        )

        self.out_activation = None
        if out_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # [batch_size, input_size] -> [batch_size, 1, input_size]
        input_ids = torch.unsqueeze(input_ids, 1)

        conved_output = self.conv1d(input_ids)
        output = self.out(conved_output)

        if self.out_activation is not None:
            output = self.out_activation(output)

        return output

    def freeze_transformer_layer(self):
        pass

    def unfreeze_transformer_layer(self):
        pass

    def get_predictions_from_outputs(self, outputs):
        if self.out_activation is None:
            return outputs.flatten().tolist()
        else:
            return torch.round(outputs).flatten().tolist()

    def __get_conv_block_out_channels(self):
        return self.conv_blocks[-1]["conv1d"]["out_channels"]

    def __get_conv_block_out_size(self):
        output_size = self.input_size

        for conv_block_settings in self.conv_blocks:
            conv1d_stride = 1
            conv1d_padding = 0
            conv1d_dilation = 1

            if "stride" in conv_block_settings["conv1d"]:
                conv1d_stride = conv_block_settings["conv1d"]["stride"]
            if "padding" in conv_block_settings["conv1d"]:
                conv1d_padding = conv_block_settings["conv1d"]["padding"]
            if "dilation" in conv_block_settings["conv1d"]:
                conv1d_dilation = conv_block_settings["conv1d"]["dilation"]

            conv1d_kernel_size = conv_block_settings["conv1d"]["kernel_size"]

            # Conv1D output size calculation
            output_size = (
                (
                    output_size +
                    2 * conv1d_padding -
                    conv1d_dilation * (conv1d_kernel_size - 1) - 1
                ) // conv1d_stride + 1
            )

            # MaxPool1D output size calculation
            max_pool1d_kernel_size = conv_block_settings["max_pool_1d"]["kernel_size"]
            max_pool1d_stride = max_pool1d_kernel_size
            max_pool1d_padding = 0
            max_pool1d_dilation = 1

            if "stride" in conv_block_settings["max_pool_1d"]:
                max_pool1d_stride = conv_block_settings["max_pool_1d"]["stride"]
            if "padding" in conv_block_settings["max_pool_1d"]:
                max_pool1d_padding = conv_block_settings["max_pool_1d"]["padding"]
            if "dilation" in conv_block_settings["max_pool_1d"]:
                max_pool1d_dilation = conv_block_settings["max_pool_1d"]["dilation"]

            output_size = (
                (
                    output_size +
                    2 * max_pool1d_padding -
                    max_pool1d_dilation * (max_pool1d_kernel_size - 1) - 1
                ) // max_pool1d_stride + 1
            )

        return output_size
