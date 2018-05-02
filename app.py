import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle
import jsonpickle
import os
from torch.autograd import Variable 
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import cv2

import flask
from flask import Flask, request, render_template, Response

app = Flask(__name__)

@app.route("/")
@app.route("/index", methods=['GET', 'POST'])
def index():
    return flask.render_template('index.html')


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def load_image(image, transform=None):
    #image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

@app.route('/predict', methods=['POST'])    
def main():
    if request.method=='POST':
    
        # Args
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--encoder_path', type=str, default='./models/encoder-5-3000.pkl',
                            help='path for trained encoder')
        parser.add_argument('--decoder_path', type=str, default='./models/decoder-5-3000.pkl',
                            help='path for trained decoder')
        parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                            help='path for vocabulary wrapper')
        
        # Model parameters (should be same as paramters in train.py)
        parser.add_argument('--embed_size', type=int , default=256,
                            help='dimension of word embedding vectors')
        parser.add_argument('--hidden_size', type=int , default=512,
                            help='dimension of lstm hidden states')
        parser.add_argument('--num_layers', type=int , default=1 ,
                            help='number of layers in lstm')
        args = parser.parse_args()
        # Image preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
        
        # Load vocabulary wrapper
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # Build Models
        encoder = EncoderCNN(args.embed_size)
        encoder.eval()  # evaluation mode (BN uses moving mean/variance)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                            len(vocab), args.num_layers)
        

        # Load the trained model parameters
        encoder.load_state_dict(torch.load(args.encoder_path))
        decoder.load_state_dict(torch.load(args.decoder_path))

        r = request
        img = r.data
        nparr = np.fromstring(img, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pil_im = Image.fromarray(img)
        image = load_image(pil_im, transform)
        image_tensor = to_var(image, volatile=True)
        
        # If use gpu
        if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()
        
        # Generate caption from image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids.cpu().data.numpy()
        
        # Decode word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption[1:-1])

        # build a response dict to send back to client
        response = {'message': sentence}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)

        return Response(response=response_pickled, status=200, mimetype="application/json")

        
        
        
        '''image = Image.open(args.image)
        plt.imshow(np.asarray(image))'''
    
if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=8000, debug=True)
