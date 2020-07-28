import torch


def focal_loss(output, target, alpha=2, beta=4):

    ones = torch.ones((64,64)).cuda()
    zeros = torch.zeros((64,64)).cuda()

    ones_board = torch.where(target == 1, output, ones)
    zeros_board = torch.where(target != 1, output, zeros)

    N = torch.sum(torch.where(target == 1, target, zeros))

    epsilon = 1e-10

    ones_board = torch.pow(1-ones_board, alpha) * torch.log(ones_board+epsilon)

    zeros_board = torch.pow(1-target, beta) * torch.pow(zeros_board, alpha) * torch.log(1-zeros_board+epsilon)

    return -(ones_board+zeros_board).sum()/N


def size_loss(output, target, center):

    epsilon = 1e-10
    zeros = torch.zeros((64,64)).cuda()
    ones = torch.ones((64,64)).cuda()

    N = torch.where(center == 1, center, zeros)
    N = torch.sum(N)

    l1_output = torch.where(center == 1, output, zeros)
    # log_output = torch.where(center == 1, output, ones)

    l1_loss = torch.abs(l1_output-target)
    # log_loss = torch.abs(torch.log(log_output+epsilon)-torch.log(target+epsilon))

    l1 = l1_loss.sum()/N * 0.1
    logl1 = torch.log(1-torch.tanh(l1_loss)+epsilon).sum()/N

    return l1 - logl1


def center_loss(output, target):
    o_heat = output[0]
    o_size = output[1]
    t_heat = torch.from_numpy(target[0]).type(torch.FloatTensor).cuda()
    t_size = torch.from_numpy(target[1]).type(torch.FloatTensor).cuda()

    fl = focal_loss(o_heat, t_heat)

    sz = size_loss(o_size, t_size, t_heat)
    return fl + sz


def custom_center_loss(output, target):
    o_heat = output[0]
    o_size = output[1]
    o_seg = output[2]
    t_heat = torch.from_numpy(target[0]).type(torch.FloatTensor).cuda()
    t_size = torch.from_numpy(target[1]).type(torch.FloatTensor).cuda()
    t_seg = torch.from_numpy(target[2]).type(torch.FloatTensor).cuda()

    fl = focal_loss(o_heat, t_heat)

    sz = size_loss(o_size, t_size, t_heat)

    l1 = torch.mean(torch.abs(o_seg - t_seg))
    return fl + sz + l1



