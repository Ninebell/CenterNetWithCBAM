from torch_model.layers import BottleNeckBlock, Hourglass
import torch
import torch.nn as nn


class CenterNet(nn.Module):

    def __init__(self, feature, out_channel, out_activation, n_layer=5, n_stack=2):
        super(CenterNet, self).__init__()
        self.feature = feature
        self.n_layer = n_layer
        self.n_stack = n_stack
        self.out_ch = out_channel
        self.out_activation = out_activation
        self.__build__()

    def __build__(self):
        feature = self.feature
        out_count = len(self.out_activation)
        n_s = self.n_stack
        self.conv7 = nn.Conv2d(3, feature, 7, stride=2, padding=3)
        self.conv3 = nn.Conv2d(feature, feature, 3, stride=1, padding=1)

        self.hours = nn.ModuleList([Hourglass(feature, feature, self.n_layer) for _ in range(n_s)])

        self.bottles_1 = nn.ModuleList([BottleNeckBlock(feature, feature) for _ in range(n_s-1)])
        self.bottles_2 = nn.ModuleList([BottleNeckBlock(feature, feature) for _ in range(n_s-1)])

        self.bottles_3 = nn.ModuleList([BottleNeckBlock(sum(self.out_ch), feature) for _ in range(n_s-1)])



        total_inter = []
        for j in range(n_s-1):
            inter = []
            for i in range(out_count):
                inter.append(nn.Conv2d(feature, self.out_ch[i], 1, stride=1, padding=0))
            total_inter.append(nn.ModuleList(inter))

        self.intermediate_out = nn.ModuleList(total_inter)

        self.out_front = []
        self.out = []
        self.out_batch = []
        for i in range(out_count):
            self.out_front.append(nn.Conv2d(feature, feature, 3, stride=1, padding=1))
            self.out.append(nn.Conv2d(feature, self.out_ch[i], 1, stride=1, padding=0))
            self.out_batch.append(nn.BatchNorm2d(feature))

        self.out_front = nn.ModuleList(self.out_front)
        self.out = nn.ModuleList(self.out)
        self.out_batch = nn.ModuleList(self.out_batch)

        self.batch1 = nn.BatchNorm2d(feature)
        self.batch2 = nn.BatchNorm2d(feature)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        output = []
        x = torch.selu(self.batch1(self.conv7(x)))
        x = self.max_pool(torch.selu(self.batch2(self.conv3(x))))

        for i in range(self.n_stack-1):
            init = x
            x = self.hours[i](x)
            x = self.bottles_1[i](x)

            inter_out = []
            for j in range(len(self.out_ch)):
                intermediate_out = self.out_activation[i](self.intermediate_out[i][j](x))
                inter_out.append(intermediate_out)
            output.append(inter_out)
            x = self.bottles_2[i](x)
            intermediate_out = torch.cat([inter for inter in inter_out], dim=1)
            inter = self.bottles_3[i](intermediate_out)
            x = init + inter + x

        last_hour = self.hours[-1](x)

        last_out = []
        for i in range(len(self.out_ch)):
            out = torch.selu(self.out_batch[i](self.out_front[i](last_hour)))
            out = self.out_activation[i](self.out[i](out))
            last_out.append(out)
        output.append(last_out)

        return output

    def info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print("{0:^40s}".format('CenterNet Information'))
        print("{0:^40s}".format("{0:22s}: {1:10d}".format('hourglass repeat', self.n_stack)))
        print("{0:^40s}".format("{0:22s}: {1:10d}".format('hourglass layer count', self.n_layer)))
        print("{0:^40s}".format("{0:22s}: {1:10,d}".format('total parameter', total_params)))


if __name__ == "__main__":
    net = CenterNet(256, 5, torch.selu, n_layer=3, n_stack=4)
    lr = 1e-4

    optim = torch.optim.Adam(net.parameters(), lr)

    pytorch_total_params = sum(p.numel() for p in net.parameters())

    print(pytorch_total_params)

    input = torch.zeros((1,3,256,256))
    criterion = torch.nn.CrossEntropyLoss()
    output = net(input)
    print(len(output), output[-1].shape)


