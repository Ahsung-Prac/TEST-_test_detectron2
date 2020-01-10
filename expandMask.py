import torch


def edgeSearch(mask, x_s, x_d, flag):
    maskArry = torch.where(mask == True)[0]
    if maskArry.size() == torch.Size([0]):
        return -1
    if flag:
        src = maskArry[0] + 1
    else:
        src = maskArry[-1] - 1
    return src


def expand(masks, size, y_s, y_d, x_s, x_d):
    mask = masks.clone()
    masktemp = masks.clone()
    y_s, y_d, x_s, x_d = int(y_s), int(y_d), int(x_s), int(x_d)
    for i in range(y_s, y_d):
        leftx = edgeSearch(masktemp[i], x_s, x_d, True)
        rightx = edgeSearch(masktemp[i], x_s, x_d, False)
        if leftx == -1:
            continue
        mleft = max(leftx - size, 0)
        mright = min(rightx + size, mask.shape[1])
        mask[i][mleft:leftx] = True  # 왼쪽으로 쭉
        mask[i][rightx:mright] = True  # 오른쪽으로 쭉

    masktemp = masks.clone()
    for i in range(x_s, x_d):
        upy = edgeSearch(masktemp.T[i], y_s, y_d, True)
        downy = edgeSearch(masktemp.T[i], y_s, y_d, False)

        if upy == -1:
            continue
        mup = max(upy - size, 0)
        mdown = min(downy + size, mask.shape[0])
        mask.T[i][mup:upy] = True  # 위로 쭉
        mask.T[i][downy:mdown] = True  # 아래로 쭉

    return mask


