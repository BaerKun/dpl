import RNN
import utils
import torch
import os

def main():
    corpus = utils.load_wiki_text()
    vocab = corpus.build_vocab(8)
    loader = corpus.get_loader(256, 16, random_sample=True)
    model = RNN.SeqRNN(len(vocab), 256, 2, 512, rnn_mode="lstm")

    mm = utils.ModelManager(model, seq2seq=True, output_state=True)
    mm.train(loader, torch.nn.CrossEntropyLoss(), 10, warmup_steps=4)
    mm.save(os.path.join(utils.project_root, "weights", "gru.pt"))

    state = None
    while True:
        question = input("You: ")
        if not question:
            break
        q_tensor = vocab.encode_from_str(question)
        answer_tensor, state = mm.predict_seq(q_tensor, 40, state)
        answer = vocab.decode2str(answer_tensor[0])
        print("Bot: ", answer)


if __name__ == '__main__':
    main()