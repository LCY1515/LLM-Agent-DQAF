import random
train = []
test = []
val=[]
with open('u.data') as f:
    for line in f:
        items = line.strip().split('\t')
        if int(items[2])<4:
            continue
        num=random.random()
        if num > 0.2:
            # train保留四列（含时间戳）
            new_line = ' '.join(items) + '\n'
            train.append(new_line)
        elif num > 0.1:
            # val只保留前三列
            new_line = ' '.join(items[:3]) + '\n'
            val.append(new_line)
        else:
            # test只保留前三列
            new_line = ' '.join(items[:3]) + '\n'
            test.append(new_line)

with open('train.txt','w') as f:
    f.writelines(train)

with open('test.txt','w') as f:
    f.writelines(test)

with open('val.txt','w') as f:
    f.writelines(val)
