import sys
sys.path.append('./python')
sys.path.append('./apps')
import needle as ndl
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=16, device=device, dtype="float32")
model = LanguageModel(30, len(corpus.dictionary), hidden_size=10, num_layers=2, seq_model='rnn', device=device)
train_acc, train_loss = train_ptb(model, train_data, seq_len=1, n_epochs=1, device=device)
print(f'train_acc={train_acc}, train_loss={train_loss}')
test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=40, device=device)
print(f'test_acc={test_acc}, train_loss={train_loss}')
# train_acc=0.04267539631994768, train_loss=0.44389123265362873
# test_acc=0.035492174851011055, train_loss=0.44389123265362873