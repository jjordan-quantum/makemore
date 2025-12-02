from pprint import pprint
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

words = open('./src/names.txt', 'r').read().splitlines()
print(words[:10])
print(len(words))
print(min(len(w) for w in words))
print(max(len(w) for w in words))

b = {}
# for w in words[:3]:
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        # print(ch1, ch2)

pprint(sorted(b.items(), key = lambda kv: -kv[1]))

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1


p = N[0].float()
p = p / p.sum()
print(p)
# ===========================

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(ix)
print(itos[ix])
# exit()

# ===========================

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p/p.sum()
print(p)

# ===========================

m = torch.multinomial(p, num_samples=100, replacement=True, generator=g)
print(m)
# exit()

# ===========================

P = (N+1).float()
P /= P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)

for i in range(50):
    out = []
    ix = 0
    while True:
        p = P[ix]
        p = N[ix].float()
        p = p / p.sum()
        # p = torch.ones(27) / 27.0

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print(''.join(out))

# ===========================
# CHK FREQ.
# ===========================

log_likelihood = 0.0
n = 0

# for w in words[:3]:
for w in ["andrejq"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

# ===============================
# Create training set of bigrams (x, y)
# ===============================
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f'number of examples: {num}')
# print(xs)
# print(ys)

xenc = F.one_hot(xs, num_classes=27).float()
print(xenc)

# DOT-PROD example
# ==================
# sample = logits[3, 13]
# print(sample)
# inp = xenc[3]
# print(inp)
# col = W[:, 13]
# print(col)
# dot_prod = (inp * col).sum()
# print(dot_prod)

# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# forward pass
# ===================
xenc = F.one_hot(xs, num_classes=27).float() # create input to network: one-hot encoding
logits = xenc @ W # predict log-counts
print(logits)
print(logits.shape)

counts = logits.exp() # equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
# btw: the last 2 lines here are together called a 'softmax'

print(probs)

loss = -probs[torch.arange(num), ys].log().mean()
print(f'loss: {loss.item()}')

# backward pass
# ===================

W.grad = None # set gradient to zero
loss.backward()

print(W.grad)

W.data += -0.1 * W.grad

xenc = F.one_hot(xs, num_classes=27).float() # create input to network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(num), ys].log().mean()
print(f'loss: {loss.item()}')


# gradient decent
# ===================

for k in range(100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()  # create input to network: one-hot encoding
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # equivalent to N
    probs = counts / counts.sum(1, keepdims=True)  # probabilities for next character
    # loss = -probs[torch.arange(num), ys].log().mean()
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean() # add gradient gravity

    # backward pass
    W.grad = None  # set gradient to zero
    loss.backward()

    # update
    W.data += -50.0 * W.grad
    print(f'loss: {loss.item()}')


# ===========================
# SAMPLE FROM NN MODEL
# ===========================

g = torch.Generator().manual_seed(2147483647)

for i in range(50):
    out = []
    ix = 0
    while True:
        # -- BEFORE
        # p = P[ix]
        # p = N[ix].float()
        # p = p / p.sum()
        # p = torch.ones(27) / 27.0

        # -- NOW:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W # predict log counts
        counts = logits.exp() # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True) # probabilities for next character
        # ---

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print(''.join(out))


exit()


# ===========================
# TRAINING
# ===========================

nlls = torch.zeros(5)

for i in range(5):
    # i-th bigram:
    x = xs[i].item() # input character index
    y = ys[i].item() # label character index
    print('----------')
    print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y}')
    print('input to the neural net: ', y)
    p = probs[i, y]
    print('probability assigned by the net to the correct character: ', p.item())
    logp = torch.log(p)
    print('log likelihood: ', logp.item())
    nll = -logp
    print('negative log likelihood: ', nll.item())
    nlls[i] = nll

print('===========')
print('average negative log likelihood, i.e. loss = ', nlls.mean().item())


exit()
# ===========================
# PLOT
# ===========================

# print(N[0])
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontsize=6)
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray', fontsize=6)

plt.axis('off')
plt.show()

