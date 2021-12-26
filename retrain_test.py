

import retrain
import torch

LR = 0.00001
reer = retrain.Retrainer('/mnt/4/models/gpt2-ptmap', '/mnt/4/models/gpt2-ptmap16')
print('Measuring differences ...')
mods = reer.mods_by_difference(shallowness=2)
loss, (src_name, src_block), (dst_name, dst_block) = mods[0]
print(src_name, loss.item())
datagen = reer.datagen_for_mod(src_block)
dst_block.train()
dropout_origps = {
    mod : mod.p
    for mod in dst_block.modules()
    if type(mod) is torch.nn.Dropout
}
for dropout, orig_p in dropout_origps.items():
    dropout.p = 0
optim = torch.optim.Adam(dst_block.parameters(), lr=LR)
while True:
    optim.zero_grad()
    loss_sum = 0
    for epoch_item in range(8):
        #for dropout_scale in (0, 1):
        #    for dropout, orig_p in dropout_origps.items():
        #        dropout.p = orig_p * dropout_scale
        data = datagen()
        label, pred = reer.compare_mods(src_block, dst_block)
        loss = (label - pred).abs().mean()
        print(loss.item(), end=' ', flush=True)
        loss_sum += loss.detach()
        loss.backward()
    loss_sum /= 8
    print('->', loss_sum.item())
    optim.step()

    mods[0] = (loss_sum, (src_name, src_block), (dst_name, dst_block))
    mods.sort(key = lambda elem: elem[0], reverse = True)
    if mods[0][0] != loss_sum:
        loss, (src_name, src_block), (dst_name, dst_block) = mods[0]
        print(src_name, loss.item())
        datagen = reer.datagen_for_mod(src_block)
        dst_block.train()
        dropout_origps = {
            mod : mod.p
            for mod in dst_block.modules()
            if type(mod) is torch.nn.Dropout
        }
        for dropout, orig_p in dropout_origps.items():
            dropout.p = 0
        optim = torch.optim.Adam(dst_block.parameters(), lr=LR)
