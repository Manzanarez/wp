# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import logging


import data
import model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('info.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)



# GAM- All arguments needed for the neural network
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
#Original
##parser.add_argument('--data', type=str, default='/home/gerardo/AI/wlm/data/wikitext-2',
## GAM - Activate 6may19
#parser.add_argument('--data', type=str, default='/home/gerardo/code/wp/data/train',
## GAM ~ server Paris
##parser.add_argument('--data', type=str, default='/users/gerardo.aleman/wp/data/train',
##parser.add_argument('--data', type=str, default='/opt/wp/data/txt/clean',

## GAM ~ server Paris
##parser.add_argument('--data', type=str, default='/users/aleman/wp/data/train',
parser.add_argument('--data', type=str, default='/home/gerardo/code/wp/data/train',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
##parser.add_argument('--emsize', type=int, default=200,
## GAM ~ Change from 200 to 128
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
##parser.add_argument('--nhid', type=int, default=200,
## GAM ~ Change from 200 to 100 to make it faster
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
##parser.add_argument('--lr', type=float, default=20,
## GAM ~ Change from 20 to 5 to make it faster
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
## Change to 16
##parser.add_argument('--batch_size', type=int, default=20, metavar='N',
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
##parser.add_argument('--bptt', type=int, default=35,
## GAM ~ Change from 35 to 10 to make it faster
parser.add_argument('--bptt', type=int, default=10,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
#parser.add_argument('--log-interval', type=int, default=200, metavar='N',
## GAM ~ Change from 200 to 16 to make it faster
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
#GAM-To seed the Random Number Generator (RNG) for all devices (both CPU and CUDA):
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#GAM - Device where is going to be executed CUDA or CPU
##device = torch.device("cuda" if args.cuda else "cpu")
## How to specify the GPU to use, in this case number 2
#device = torch.device(2)
device = torch.device('cuda:2')


###############################################################################
# Load data
###############################################################################

#GAM - Load the corpus data using data.py
corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz #GAM~ Size of the tensor 'data' // (floor division - retuns the integer of the quotient) diveded by 'bsz' (batch size)
    # Trim off any extra elements that wouldn't cleanly fit (remainders). ##GAM~ The ones that are excluded from the // (flor division)
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches. #GAM~ Makes a matix size nbatch x bzs (batch size, default 20)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
##Train data (prompts and stories
##GAM May619 Activate
train_datap = batchify(corpus.trainp, args.batch_size) #GAM~ Tensor with word indexes size corpus.train.size(0)//arg.batch_size x arg.batch_size
train_datas = batchify(corpus.trains, args.batch_size) #GAM~ Tensor with word indexes size corpus.train.size(0)//arg.batch_size x arg.batch_size
nbatch1 = corpus.trains.size(0) // args.batch_size
file = open("ejemplo.txt","w")
for i in range(nbatch1):
#    for j in range(args.batch_size):
        file.write(str(train_datas[i]) + "\n")
##        file
        print(train_datas[i])
        logger.info('Train data story[i] %s', str(i))
print(train_datas)
logger.info('Train data story')
##Validate data (prompts and stories)
##GAM May619 Activate
val_datap = batchify(corpus.validp, eval_batch_size) #GAM~ Tensor with word indexes size corpus.valid.size(0)//arg.batch_size x arg.batch_size
val_datas = batchify(corpus.valids, eval_batch_size) #GAM~ Tensor with word indexes size corpus.valid.size(0)//arg.batch_size x arg.batch_size
##Test data (prompts and stories)
##GAM May619 Activate
test_datap = batchify(corpus.testp, eval_batch_size) #GAM~ Tensor with word indexes size corpus.test.size(0)//arg.batch_size x arg.batch_size
test_datas = batchify(corpus.tests, eval_batch_size) #GAM~ Tensor with word indexes size corpus.test.size(0)//arg.batch_size x arg.batch_size

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

#GAM ~ This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
#GAM ~ Intializes 'criterion' as CrossEntropyLoss() with ignore_index = -100
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i) ## (Original script)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
 ## GAM           print('\n evaluate {:10d} | data source size {:10d} | bptt {:4d}'.format(i, data_source.size(0), args.bptt))
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden,data,0)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


##def train(): #Original
def train(trainx,ps,hp,st): #GAM ~ Recieve what is it going to train on, hp = hidden from prompt

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
#    hidden = model.init_hidden(args.batch_size) #GAM ~ Original
##    hidden = model.init_hidden(args.batch_size,args.bptt) #GAM ~ 15may19


    hiddenp = model.init_hidden(args.batch_size)
    hidden = model.init_hidden(args.batch_size)
#    print(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr) ## Original ~ 15abr19
##    optimizer = torch.optim.Adam(model.parameters(),lr) ##GAM ~ 15apr19 default lr= 1e-3
#    for batch, i in enumerate(range(0, train_datas.size(0) - 1, args.bptt)): #Original
    for batch, i in enumerate(range(0, trainx.size(0) - 1, args.bptt)): ##Original
##    for batch, i in enumerate(range(0, trainx.size(0), args.bptt)): #Changed because it doesn't takes when first dimension = 1
##        data, targets = get_batch(train_datap, i) #Assign data and target (original program)
        if ps == 0:
            data, targetp = get_batch(trainx, i ) #Assign data to Prompt and target to Stories


        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to the start of the dataset.
            hiddenp = repackage_hidden(hiddenp)
            model.zero_grad() #GAM~ """Sets gradients of all model parameters to zero.""" ## In the original is active
            output, hiddenp = model(data, hiddenp,data,ps) #GAM ~ module.py _call_ -- forward()
            loss = criterion(output.view(-1, ntokens), targetp)
        else:

        # feed the prompt (train_datap) into stories (train_datas) ## GAM ~ 13may19
##        datax = torch.empty(35,32, dtype=torch.float)
            data, targets = get_batch(trainx, i )
##ojo            datax = data.float()
##ojo            datax.resize_as_(output)
##GAM~ Concatenates the result of the "prompt" (output) last hidden state with the storie "train_datas" (data)  that corresponds
##        outputx = torch.cat((output,datax),0)
            hidden = repackage_hidden(hidden)
            model.zero_grad()  # GAM~ """Sets gradients of all model parameters to zero.""" ## In the original is active
            outputy, hidden = model(data, hidden,hp[1][1],1) #GAM ~ module.py _call_ -- forward. Feed the last hidden state
            outputz = torch.split(outputy,args.batch_size,0) ## 35 size of sequence length
##            (outputy.size(0)-hidden[1].size(0),outputy.size(1),outputy.size(2))

#GAM~ outputy 2 more dimension because of the concatenation of data and hidden[1] from the line above

##        loss = criterion(output.view(-1, ntokens), targets) #GAM ~ initializes 'loss as a 'criterion' (CrossEntropyLoss()) Original
##        loss = criterion(outputy.view(args.bptt, ntokens), targets) #GAM ~ 16may19
            loss = criterion(outputy.view(-1, ntokens),targets)  # GAM ~ initializes 'loss as a 'criterion' (CrossEntropyLoss()) Original

##        optimizer.zero_grad()  ## GAM ~ 15abr19 ##GAM~ May6-19 (in the original is not in the code)
##GAM -        print("loss - Criterion: ", loss)
        loss.backward(retain_graph=True) #GAM ~ Initializes 'loss' as a backward() in __init__.py, loos has the value given when calling CrossEntropyLoss
##GAM -        print("loss backward: ", loss)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #GAM ~ Clips gradient norm of an iterable parameters.
        # The norm is computed over all gradients together, as if they were concatenated into a single vector.
        # Gradients are modified in-place.
        #Gradient clipping limits the magnitude of the gradient and can make
        # stochastic gradient descent (SGD) behave better in the vicinity of steep cliffs:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) ##GAM~ May6-19 in the original is active

        optimizer.step()  ## GAM ~ 15abr19
        for p in model.parameters(): #GAM~ 23april19 in the original is active?
            p.data.add_(-lr, p.grad.data) #GAM~ 23april19 in the original is active?

        total_loss += loss.item()
## Delete        print("|Batch {:5f} |Total loss {:5.2f} | ", batch , total_loss)

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            ##GAM~ ppl = perplexity is calculated using cur_loss which is the total_loss (using cross entropy)
            ## divided by the log_interval, in other words the average loss by log_interval or batch
            print('|st {:3d}| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(st,
                epoch, batch, len(train_datas) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return hiddenp


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        hp = model.init_hidden(args.batch_size) #result of hidden from prompt
        st = 0
        pr = 0
        pathp = os.path.join(args.data, 'prompts.tokens.alligned_train.24042019.txt')
        with open(pathp, 'r', encoding="utf8") as fp:
            for x in fp:
                xp = data.Corpusline(x,corpus.dictionary.word2idx) #Tokenize the line
                xpb = batchify(xp.linesinfile,args.batch_size)
                hp=train(xpb,0,hp,pr)
 ##               print ('End of train()  Prompt: ', pr)
                paths = os.path.join(args.data, 'stories.tokens.alligned_train.24042019.txt')
                with open(paths, 'r', encoding="utf8") as fs:
                    for y in fs:
                        if st == pr:
                            ys = data.Corpusline(y,corpus.dictionary.word2idx)
                            ysb = batchify(ys.linesinfile,args.batch_size)
                            hp=train(ysb,1,hp,st)
##                            print ('End of train() Story: ',st)
                            break
                        else:
                            st+= 1
                pr += 1
        val_loss = evaluate(val_datas)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_datas)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)